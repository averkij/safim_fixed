# SAFIM review: empty-answer scores, post-processing, and other correctness issues

Scope: `evaluate.py`, `generate.py`, `prompt_utils.py`, `model_utils.py`, `ast_utils.py`, `exec_utils.py`,
`data_utils.py`, plus the HF datasets (`block`, `block_v2`, `control`, `control_fixed`, `api`).
Everything below was either executed (numbers) or read from code (marked **[code reading]**).
Scripts to reproduce are in `review/`.

## TL;DR

1. **"An empty answer scores >50%" does not reproduce.** I executed the empty completion on 6,515 tasks
   (C++, Java, Python, C#; block, block_v2, control) and **0 passed**. The API task cannot pass an empty answer
   by construction. So with this repo's scoring logic, the number you saw is not coming from the benchmark
   tasks themselves. Section 1 lists what could produce it in your setup and a one-command audit to find out.
2. The one real "vacuous pass" I found: in **Python block tasks a bare `pass` passes 11%** of tasks
   (92/805 block, 127/1160 block_v2), because 83% of tasks have a single unit test.
3. Post-processing is justified and mostly sound for block/control, but **`truncate_api_call` destroys
   correct answers** (4.8% of API tasks can never pass), and long prompts are **silently turned into empty
   answers** in `model_utils.py` **[code reading]**.
4. Largest correctness bug found: **Python API scoring ignores the function name** (`totally_wrong_function(args)`
   passes 180/180 Python call tasks).
5. `evaluate.py` crashes on an empty answers file and mis-handles missing tasks. If your harness wraps it, this
   may be where bad numbers come from.

---

## 1. Does an empty answer pass? (No, not in this repo's logic)

### Method
`evaluate.py` builds `eval_prompt.replace("{{completion}}", completion)`, runs it on the task's unit tests through
ExecEval, and passes iff all tests pass. Docker is unavailable here, so I wrote `review/localexec.py`, a local
stand-in with the same contract (compile, run every test, token-wise stdout compare). I validated it on the
**ground truth** first (section 1.3). Then I ran `completion = ""` on stratified random samples
(500 per language, or the whole set for C#).

### Results: empty completion

| set | C++ | Java | Python | C# | outcome of the failures |
|---|---|---|---|---|---|
| block (8,781) | 0/500 | 0/500 | 0/500 | 0/529 (all) | C++/Java mostly WRONG_ANSWER (92%/78%), Python 100% COMPILATION_ERROR |
| block_v2 (4,571) | 0/500 | 0/500 | 0/500 | 0/452 (all) | same pattern |
| control (8,629) | 0/500 | 0/500 | 0/500 | 0/534 (all) | 100% COMPILATION_ERROR (`if ()`, `for ()`, `if :` are syntax errors) |
| control_fixed, Python | | | 0/500 | | COMPILATION_ERROR |
| api (310) | by construction: `syntax_match("", ground_truth)` is False for all 310 | | | | |

6,515 executed, 0 passed (95% upper bound on the true rate: about 0.05%). A hole in C++/Java/C# block tasks is
syntactically optional, yet removing it changes the output in every sampled case. That looks like the dataset
having been filtered against exactly this trivial baseline (I did not verify that against the paper).

### 1.2 The one vacuous-pass path: Python `pass`
Empty Python blocks are syntax errors, but a stub is valid:

| set | `pass` stub passes |
|---|---|
| block, Python (all 805) | **92 (11.4%)** |
| block_v2, Python (all 1160) | **127 (10.9%)** |

The block is simply invisible to the single test. It is not >50%, but it is an exploitable floor. A model that
learns to emit `pass` (or a trivial statement) gets about 11% on Python for free.

### 1.3 Runner validation (so the zeros are trustworthy)
Ground truth on 360 sampled block/control tasks: 336 pass (93.3%). Of the 24 failures, 22 are
Codeforces-checker differences (`yes` vs `YES`, `0` vs `0.000000`; see issue I-3), 1 is a compile error and 1 is a
CPython timeout (ExecEval uses PyPy). A runner that passes real solutions and fails empty ones is not
rejecting everything.

### 1.4 What could produce ">50%" in your setup (hypotheses, not verified)
I cannot see your run. In descending likelihood:

1. **A different scoring path than `evaluate.py`.** Any wrapper where "no result / ExecEval error / empty
   response list" counts as pass will do it, e.g. `all([])` is `True` in Python. In this repo that case is
   only avoided by accident (`result[0]` raises first in `exec_utils.run_test`).
2. **ExecEval unhealthy or misconfigured** (wrong image/version, tests silently not executed). Negative control:
   evaluate an all-empty file. If it is not about 0%, the executor is the problem.
3. **Not actually empty:** the "empty" run still contains answers for many tasks (e.g. leftover or cached outputs),
   or only the tasks that happen to be Python `pass`-like.
4. A different dataset variant than the ones I tested.

To settle it, run on your files:

```bash
python review/audit_empty_passes.py outputs_X.jsonl results_X.jsonl
```

It prints how many tasks have an empty completion and how many of those are marked `passed`. If that number
is ~0 and your score is still >50%, the score came from non-empty completions. Send me the two files and I
will go through them.

I recommend adding the two controls below to the README as a mandatory pre-flight check for any new
executor/setup: **empty file ⇒ about 0%** and **ground-truth file ⇒ about 93-100%**.

---

## 2. Post-processing: what it does, why, and is it correct

Base FIM models rarely emit a clean end-of-middle: generation (`max_new_tokens=128`, stop = EOS only) usually
runs past the hole. The post-processors cut the raw text down to "the hole". Applied in the order given by
`--post_processors`:

| processor | what it does | verdict |
|---|---|---|
| `extract_code` | chat models: pulls the first fenced block (or inline `` `code` ``), demoting blocks that contain the placeholder or the few-shot answer | OK as heuristic; an unclosed ``` fence is dropped and the raw text is returned |
| `truncate_fewshot` | cuts at `\nComplete the code in ` (model re-starting the few-shot prompt) | OK |
| `truncate_line_until_block` | pops trailing lines until the *whole program* parses (tree-sitter, no ERROR nodes) **and** the AST event trace equals the empty-hole program's trace plus pure insertions | **Mostly sound**, see below |
| `truncate_control` | Python: first non-empty line. Others: cut at first unmatched `)` | OK; breaks on `')'` char literals |
| `truncate_api_call` | cut at the first balanced `)` | **Wrong**, see below |

**Round trip test** (a perfect model's output *is* the ground truth, so a post-processor must return it
unchanged; `review/ppcheck.py`):

| processor | ground truth returned unchanged |
|---|---|
| `truncate_line_until_block`, block (8,781) and block_v2 (4,571) sample | 99.7-99.8% (0.1-0.4% of C++/Python tasks collapse to `""`) |
| `truncate_control` | 99.9% (fails only on `')'` char literals, e.g. `s[i] == ')'`) |
| `truncate_api_call` | **95.2%: 15/310 ground truths are cut**, e.g. `torch.argmax(x, dim=1).item()` becomes `torch.argmax(x, dim=1)`, `Caffeine.newBuilder().maximumSize(100)` becomes `Caffeine.newBuilder()`, `Vector3d x = mat.colPivHouseholderQr().solve(b)` becomes `...colPivHouseholderQr()` |

So **4.8% of API tasks are unattainable** with the documented `--post_processors truncate_api_call`.

**Over-generation test** (`review/overrun.py`): correct block followed by N further lines copied from the suffix:

| model over-generates by | truncation restores the exact ground truth |
|---|---|
| 1 line | 96% |
| 3 lines | 93% |
| 8 lines | 84% |

(Using current tree-sitter wheels, not the repo's pinned commits.) The rest keep the junk: truncation enforces
"parses and only inserts nodes", not "ends where the hole ends", so a continuation that is itself a valid statement
survives and fails at runtime. That is an acceptable, documented trade-off, but it does mean the score mixes
"content correctness" with "stops at the right place", unevenly across models.

Further points:
* When the post-processor rejects everything it returns `""`, which is then scored as an empty answer. Given
  section 1 that means a clean fail, not an inflated pass. Fine, but it is not flagged anywhere.
* `truncate_line_until_block` has `assert visitor.error_cnt == 0` on the *empty* program. If a grammar version
  disagrees on one program, `generate.py` dies and (see I-8) **writes no output at all**. It did not trigger in
  my samples.
* `ErrorCheckVisitor` counts only `ERROR` nodes (and ignores `;` errors). tree-sitter `MISSING` nodes pass
  silently. Only affects how aggressively lines are cut.

**Answer:** truncation is the right idea and the block/control versions are reasonable. `truncate_api_call`
should be fixed (section 4), and the pipeline should report how many answers were altered/emptied per task type.

---

## 3. Other issues

Severity = effect on measuring models correctly.

| id | sev | issue | evidence |
|---|---|---|---|
| I-1 | **High** | **Python API scoring ignores the callee.** `function_calls_match` compares only `args` and `keywords`, never `node.func`. Whitespace is stripped first. | `totally_wrong_function(<same args>)` passes **180/180** Python call tasks (`review/apicheck.py`); 22 of them have no args, so any `f()` passes; `os.remove(p)` ≡ `shutil.rmtree(p)`; in chains `a.b(c).d(e)` ≡ `z.q(1).d(e)`. Python is 181/310 = 58% of API. |
| I-2 | **High** | **Silent empty/garbled answers from long prompts** **[code reading + token count]**. Classes that set `tokenizer.model_max_length` call `tokenizer(prompt, truncation=True)`. Truncation is on the **right**, which cuts the FIM sentinel (`<fim_middle>`) and the end of the suffix, and `max_length=min(len+128, max_length)` leaves no room to generate. DeepSeek also returns `""` on any `RuntimeError` (e.g. OOM). OpenAI wrapper returns `""` on context errors or after 999 retries. | Prompts ≥ 2048 tokens: **6.8% block, 7.2% control, 11.8% block_v2** (DeepSeek tokenizer as proxy for the 2048-context classes: StarCoder, Incoder, CodeGen, Phi, Wizard, SantaCoder); ≥ 4096: 0.6-1.0%. Not run end-to-end (no model/GPU here), please confirm with a 2048-context model. |
| I-3 | **High?** | **Strict output comparison vs Codeforces checkers.** My strict token compare fails the *reference solution* on `YES/yes` and `0/0.000000`. If your ExecEval compares strictly too, those tasks cannot be passed. | 22/360 sampled ground-truth runs (6.1%), all explained by case/float tolerance. Needs a check against the real ExecEval: run the ground-truth file through it. |
| I-4 | **High** | **Tests are too weak.** Median 1 unit test per task; 83% have exactly one; max 7. Python `pass` passes 11%. Any plausible-but-wrong block that happens to match that one input scores as correct. | counts in section 1.2 and `unit_tests` stats |
| I-5 | **High** | **`evaluate.py` fragility.** Empty answers file ⇒ `UnboundLocalError: completion` (reproduced). A task missing from the file is labelled `EMPTY`, but `completion` then refers to the *previous* task's answer for the later `EMPTY`/syntax-check logic. Extra/duplicate `task_id`s are ignored silently (dict, last wins). Nothing checks the answer file covers all tasks, so a partial file gives a pass rate computed over the full set but misleading `EMPTY` counts. | run with an empty file |
| I-6 | Med-High | **Python `control` uses the broken variant.** `load_dataset("control")` has ground truth ending in `:` while `eval_prompt` has no colon (750/750 Python tasks). A model that writes the natural condition without `:` is a syntax error. The HF `control_fixed` config fixes this (colon moved into the suffix) but the repo never references it, and `control_fixed` would `KeyError` in `COMPLETION_TYPE_REPR`/few-shot tables for `instructed`/`prefix_feeding`/`fewshot` modes. | datasets diff: 750 tasks change `prompt`, `eval_prompt`, `ground_truth` |
| I-7 | Medium | **128-token generation cap** vs ground truth length: 243 block (2.8%) and 109 block_v2 (2.4%) ground truths exceed 128 tokens, so are unreachable. | DeepSeek tokenizer count |
| I-8 | Medium | **`generate.py` writes nothing on any exception** (only the cache is saved) and the cache key is the prompt only, so changing `--block_comments`, sampling or model reuses stale completions. | code |
| I-9 | Medium | **Noise:** `do_sample=True, temperature=0.2`, no seed, one sample. Standard error about ±0.5 pp at 8.7k tasks, so differences below ~1 pp between models are not meaningful. | arithmetic |
| I-10 | Low | `whitespace removed everywhere` in `syntax_match` also removes spaces inside string literals (`foo("a b")` ≡ `foo("ab")`), and `f(a in b)` becomes `f(ainb)`. | `review/apicheck.py` |
| I-11 | Low | `show_detailed_results.py`: `list(r_set)[0]` raises `IndexError` if a failed row has no failing outcome; `EMPTY` is only counted when the empty answer failed, so empty-but-passed rows are invisible in the summary. | code |
| I-12 | Low | `check_result: 0` is written but unused; README typos (`DeekSeek`, `-fim-tb.jsonl` for API). | code |

Not issues, checked: `prompt` and `eval_prompt` agree on prefix/suffix for all 30,920 rows; the placeholder
occurs exactly once everywhere; tree-sitter grammars are pinned in `setup_tree_sitter.bash`; no ground truth is empty.

---

## 4. Recommended fixes (in priority order)

1. **Add the pre-flight controls** (empty file ⇒ ~0%, ground-truth file ⇒ high) to the README and CI. Run
   `review/audit_empty_passes.py` on every submission.
2. **API matcher (I-1):** for Python compare the full normalised expression (`ast.dump` of the whole `Call`
   including `func`, recursively through chains). Do not delete whitespace inside string literals for any language.
3. **`truncate_api_call` (section 2):** do not stop at the first `)` when the next non-space token is `.`, `->`,
   `[` or `(`. Re-run the round-trip test; it should return 100% of the ground truths.
4. **Prompt budgeting (I-2):** never truncate on the right. Shrink the prefix from the left and the suffix from the
   right around the hole, keep the sentinels, and **count and report skipped/failed generations** instead of
   returning `""`.
5. **`evaluate.py` (I-5):** initialise `completion` from `completions.get(task_id)`, treat missing as failed with
   an explicit `MISSING` label, fail loudly when coverage < 100%, and report empty-and-passed separately.
6. **Executor comparison (I-3):** verify ground truth against the real ExecEval; if it is strict, either adopt
   a case-insensitive and float-tolerant comparator or exclude tasks whose reference solution fails.
7. **Test strength (I-4):** add tests (fuzz/differential against the reference solution) and drop or re-hole
   tasks where `pass` / a no-op stub passes.
8. **Datasets (I-6):** switch Python control to `control_fixed` and register it in the prompt tables.
9. Report ±CI with results, seed the sampling, and make `generate.py` write partial outputs.

---

## 5. Limits of this review

* Not run through the real ExecEval (no Docker daemon here). My runner uses CPython (not PyPy), the sandbox's JDK, `g++ 13`, and .NET Roslyn for C# (not Mono), and a strict token-wise comparison. The zeros for empty answers are
  robust to these differences (they are compile errors or wrong outputs); the ground-truth ceiling (I-3) is not.
* Empty-answer runs: stratified random samples of 500 per language for C++/Java/Python and the full C# sets, not
  every task. 95% upper bound on the empty-answer rate across the 6,515 runs is about 0.05%.
* No model was run, so I-2 and the effect of post-processing on real model outputs are inferred from code and
  synthetic tests (ground truth and ground truth + over-generation).
* Token counts use the DeepSeek-Coder tokenizer as a proxy (StarCoder's is gated).

## Reproduce

```bash
bash review/fetch_data.sh                      # downloads the datasets into review/data
pip install tree-sitter tree-sitter-python tree-sitter-java tree-sitter-cpp tree-sitter-c-sharp tokenizers
python review/apicheck.py                      # I-1, I-10
python review/ppcheck.py api_completion 1000 api
python review/ppcheck.py block_completion 2000 block
python review/ppcheck.py control_completion 1000 control
python review/overrun.py
python review/localexec.py review/data/block_completion.jsonl.gz empty out.jsonl --limit 500
python review/summ.py out.jsonl
```

`localexec.py` expects a C++ precompiled header directory at `review/pch` (`bits/stdc++.h` plus `.gch`) and the
paths to the .NET `csc.dll` and reference assemblies at the top of the file; edit them for your machine.
