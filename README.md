# SAFIM Benchmark


Syntax-Aware Fill-in-the-Middle (SAFIM) is a benchmark for evaluating Large Language Models (LLMs) on
the code Fill-in-the-Middle (FIM) task. SAFIM has three subtasks: Algorithmic Block Completion,
Control-Flow Expression Completion, and API Function Call Completion. SAFIM is sourced from code
submitted from April 2022 to January 2023 to minimize the impact of data contamination on evaluation
results.

![Three splits in the SAFIM benchmark illustrated with code examples.](assets/safim.png)

- Authors: [Linyuan Gong](https://gonglinyuan.com), Sida Wang, Mostafa Elhoushi, Alvin Cheung
- Paper: [https://arxiv.org/abs/2403.04814](https://arxiv.org/abs/2403.04814)
- Leaderboard: [https://safimbenchmark.com](https://safimbenchmark.com)
- Huggingface
  Dataset: [https://huggingface.co/datasets/gonglinyuan/safim](https://huggingface.co/datasets/gonglinyuan/safim)

## Environment Setup

For the maintained Python 3.12 profile, create a separate virtual environment
and install `python -m pip install -r requirements-modern.txt`. It pins the
primary dependencies, including compatible Transformers/Hub/tokenizers versions
and the parser wheels. Record the complete installed environment and PyTorch
CPU/CUDA build for each experiment; this is a tested primary-dependency profile,
not a lock of every transitive package. The historical installation below is a
separate reproduction profile. Do not combine its old SDK/parser pins with the
modern profile.

Native base-model FIM adapters are registered for `Qwen/Qwen2.5-Coder-3B`
(revision `09d9bc5d376b0cfa0100a0694ea7de7232525803`, prefix/suffix/middle order)
and `JetBrains/Mellum-4b-base` (revision
`83cce2605fbdf6a3868627e9b0a5924e0072b94d`, suffix/prefix/middle order).
These are distinct model identities; they do not alias similarly named models.
The tokenizer and weights use the same immutable revision, with remote code
disabled and marker validation. Native adapters use temperature0.2, top-p0.95,
top-k0, repetition penalty1, and the configured output cap. Their explicit
protocol and resolved identities are saved under `model` in generation metadata.

For these adapters, `--model-revision` accepts a 40-character commit;
`--context-limit` may reduce the configured model context; `--device` defaults
to cuda and `--dtype` to float16. CPU plumbing can use `--device cpu --dtype float32`.
The native contexts are32768 (Qwen) and8192 (Mellum); use a shared declared cap
such as8192 for controlled comparisons. Native adapters reject reverse_infilling
and comment-token banning because those alternative protocols are not defined
here. Native-only options on historical adapters fail rather than being ignored.
No score equivalence with historical adapter decoding defaults is implied.

Example (run separately for each registered model with separate outputs/cache):

```bash
python generate.py Qwen/Qwen2.5-Coder-3B block_v2 cache-qwen.json qwen-seed0.jsonl infilling \
  --context-limit 8192 --max-new-tokens 512 --seed 0 --post_processors truncate_block
```

This requests pretrained weight downloads and inference when run. Adapter checks
in the correction study used real pinned tokenizers/configs and random tiny CPU
models, not pretrained model evaluation. Keep each seed replicate independent
and record hardware, complete environment, actual truncation and output budget.

The fixed loader reads UTF-8 JSONL directly from dataset commit
`be132cc15372e90b6f03a608e77f2d940e384edb` using `huggingface_hub`.
It does not execute a dataset loading script or follow a moving repository head.
Generation/evaluation metadata records the configuration, revision, compressed
file SHA256 and row count. Evaluation rejects conflicting recorded input
identities; legacy prediction files without identity still need manual provenance.

Generation accepts `--max-new-tokens` (positive integer, historical default 128).
The requested cap is saved in the output `.run.json` configuration and applies to
all wrappers, including the OpenAI wrapper. Use the same declared budget when
comparing models; token counts still depend on each model's tokenizer. Increasing
the cap is a protocol change, not evidence of an improved model or a guaranteed
score increase. Context-window handling can further limit output and must also
be controlled. Failed/interrupted generation returns nonzero and preserves
completed JSONL rows; a `.run.json` status other than `complete` is not a full run.

Raw completion cache keys include the model, wrapper source hash, generation
configuration, available resolved revisions, context limit, output cap and prompt.
Legacy prompt-only entries are retained but never reused. Switching settings or
models in the same cache therefore triggers fresh inference. Use immutable model
revisions: a remote alias can change without exposing a revision to the client.

`--seed` defaults to 0. A stable SHA256-derived seed for each prompt is applied
before uncached inference, so cache hits and task order do not shift other tasks'
random draws. The same seed is sent to the remote API. This changes the formerly
unseeded generation protocol. Record several independent base seeds to measure
sampling variability; one seed is not an uncertainty estimate. GPU kernels,
dependency/hardware changes and remote service behavior can still prevent exact
repeatability. Cache reuse is not an independent replicate.

`show_detailed_results.py --format json` reports exact passed/total counts,
percentage, the percentage-point size of one task, and explicit unmeasured
sampling variability. The default CSV layout/precision is preserved for scripts;
its decimal digits do not express statistical confidence. Scores are empirical
pass@1 with one completion per task. Report independent seed replicates and
paired task outcomes when comparing model differences.

Use `block_v2` for newer block tasks. `control` and `control_fixed` are distinct
protocols: Python's header colon belongs to the answer in `control`, to the suffix
in `control_fixed`. Both have explicit prompt/few-shot support; do not pool their
scores. With `truncate_control`, a duplicate trailing colon is removed only for
the fixed variant when its evaluation suffix already supplies one. Evaluation
rejects a conflicting variant declared in a prediction `.run.json`; legacy files
without metadata still require the researcher to supply the correct variant.

Evaluation accepts `--exec-profile profile.json` with schema_version 1 and all
four language entries (`cpp`, `java`, `python`, `csharp`), each declaring a runtime
and complete ExecEval resource limits. Partial limit objects are rejected because
this engine bypasses language defaults whenever a limits object is supplied.
The profile/hash, runtime mapping and request timeout are saved in `.run.json`.
Without a profile, historical server defaults remain in use and metadata labels
them as uncaptured. Capture actual image/compiler/runtime/configuration versions
for comparisons: a runtime label is not a version check, and profile provenance
is a declaration rather than automatic attestation of a remote server.

`--force-execution` bypasses the exact-reference shortcut for tasks with tests.
Use it to validate references against the actual engine and when measuring
execution-based scores. Default shortcut behavior remains available, but each
result now labels `reference_match`, `execution`, or `syntax_match`, and the run
metadata counts them. An exact reference match alone does not verify engine
health. API tasks without execution tests retain their syntax-matching protocol.

For the separately reported augmented-test protocol, add
`--test-overlay test_overlays/block_v2_2026_10_23.json`. It adds independently
reasoned counterexamples to two block_v2 tasks whose original tests accept a
no-op. The original dataset remains unchanged. Overlays require an exact input
identity, append tests, force execution, and record their hash/affected IDs.
Report original-test and augmented-test scores separately; the two-task overlay
does not establish complete test coverage. The counterexamples and their oracles
are documented in the overlay. Other datasets or revisions are rejected.

FIM prompts reserve the requested output budget and retain the model's marker
sequence. If necessary, context is removed from the beginning of the prefix and
the end of the suffix, reducing the longer character span by one eighth per
step and checking assembled token length each time. Per-row `context` metadata
records input/output limits and removed characters. This is an explicit protocol
change for long prompts; short prompts are unchanged. Unstructured prompts that
exceed the declared context fail instead of silently losing instructions. Remote
chat limits retain the historical 32-token overhead allowance; configure/use a
model with a compatible context and treat server rejection as an incomplete run.

The remote wrapper uses the OpenAI 2.x client with explicit temperature 0.2,
top-p 0.95 and `max_completion_tokens`. `OPENAI_API_KEY` is read by the SDK.
`--api-timeout` defaults to 120 seconds per attempt and `--api-max-retries` to 2.
Exhausted requests or missing text fail generation instead of becoming empty
predictions. Models that reject these decoding parameters fail visibly; they
require a separately declared protocol. No model identifier is substituted.

Historical local environment: Python 3.8.12. The remote client requires Python
3.9 or newer; SDK interface checks use Python 3.12.3 and openai 2.29.0. Install
`openai==2.29.0` instead of the historical 0.28.1 pin below for remote generation.

CUDA version: 11.7

Docker CE version: 24.0.7

Install dependencies:

```bash
python -m pip install \
torch==1.13.0+cu117 torchvision==0.14.0+cu117 torchaudio==0.13.0 \
--extra-index-url https://download.pytorch.org/whl/cu117
python -m pip install \
Jinja2==3.1.2 openai==0.28.1 tiktoken==0.5.2 transformers==4.36.0 \
tqdm==4.64.1 tree-sitter==0.20.4 requests==2.28.1 datasets==2.18.0
```

If you encounter issues with `libstd++`, and you are using a conda environment, you can try this solution:

```bash
conda install -n [ENV_NAME] libstdcxx-ng=12.2.0 -c conda-forge
```
Build Tree-Sitter parsers:

The fixed benchmark defaults to the pinned wheel profile in
`requirements-parser.txt` (install with `python -m pip install -r requirements-parser.txt`).
Parser imports are lazy and independent of the working directory. These grammar
versions differ from the historical sources: record the profile and hold it fixed
in paired comparisons. For historical grammar reproduction, use an isolated
tree-sitter 0.20.4 environment, run the setup script below, and explicitly set
`SAFIM_PARSER_BACKEND=legacy`. Modern and legacy profiles must not be conflated.

```bash
bash setup_tree_sitter.bash
```

In another terminal, build and run ExecEval daemon:

```bash
git clone https://github.com/ntunlp/ExecEval
cd ExecEval
docker build . -t exec-eval:1.0
docker run -it -p 5000:5000 -e NUM_WORKERS=2 exec-eval:1.0
```

## Reproduce Results in Our Paper

### GPT-3.5 + One-Shot Prompt + Algorithmic Block Completion

Generate:

```bash
export OPENAI_API_KEY=YOUR_OPENAI_API_KEY_HERE
mkdir -p cache outputs_block
python generate.py \
  gpt-3.5-turbo-0301 \
  block \
  cache/gpt-3.5-turbo-0301.json \
  outputs_block/gpt-3.5-turbo-0301-few-ex-tb.jsonl \
  fewshot \
  --post_processors extract_code truncate_fewshot truncate_line_until_block  # syntax-aware truncation
```

Evaluation:

```bash
mkdir -p results_block
python evaluate.py \
  block \
  outputs_block/gpt-3.5-turbo-0301-few-ex-tb.jsonl \
  results_block/gpt-3.5-turbo-0301-few-ex-tb.jsonl
```

Show results:

```bash
python show_detailed_results.py \
  block \
  results_block/gpt-3.5-turbo-0301-few-ex-tb.jsonl
```

The expected outcome is:

```
gpt-3.5-turbo-0301-few-ex-tb,28.381643,39.975797,27.701863,22.495274,31.237900,308,1184,196,94,258,3291,707
```

So the pass@1 is 31.24%. For the interpretation of the other numbers, please refer to `show_detailed_results.py`

### DeekSeek-Coder-1.3B + PSM Prompt + Control-Flow Completion

This examples shows how to do generation using DeepSeek-Coder-1.3B using Prefix-Suffix-Middle infilling prompt on the
control-flow completion task. This experiment requires a single GPU.

Generate:

```bash
mkdir -p cache outputs_control
python generate.py \
  deepseek-ai/deepseek-coder-1.3b-base \
  control \
  cache/deepseek-coder-1.3b-base.json \
  outputs_control/deepseek-coder-1.3b-base-fim-tc.jsonl \
  infilling \
  --post_processors truncate_control  # syntax-aware truncation for control-flow expressions
```

Evaluation:

```bash
mkdir -p results_control
python evaluate.py \
  control \
  outputs_control/deepseek-coder-1.3b-base-fim-tc.jsonl \
  results_control/deepseek-coder-1.3b-base-fim-tc.jsonl
```

Show results:

```bash
python show_detailed_results.py \
  control \
  results_control/deepseek-coder-1.3b-base-fim-tc.jsonl
```

The expected outcome is:

```
deepseek-coder-1.3b-base-fim-tc,51.813153,56.777597,56.533333,59.176030,54.096651,8,190,168,47,100,2884,564
```

So the Pass@1 is 54.10%

## Evaluate Your Model and Submit the Results

To evaluate your own model with the SAFIM benchmark, follow these steps:

### Model Integration

Implement a new class in `model_utils.py` that inherits from `ModelWrapper`. This class should handle model loading,
define sentinel tokens for Fill-in-the-Middle (FIM) tasks, and set relevant hyperparameters.
Refer to existing class implementations for details.

Next, integrate your model into the framework by adding it to the `build_model` function within `model_utils.py`.

### Generate

Use `generate.py` to produce predictions for each of the three subtasks.
Replace `NAME_OF_YOUR_MODEL` with your model's identifier:

For Algorithmic Block Completion:

```bash
mkdir -p cache outputs_block
python generate.py \
  NAME_OF_YOUR_MODEL \
  block \
  cache/NAME_OF_YOUR_MODEL.json \
  outputs_block/NAME_OF_YOUR_MODEL-fim-tb.jsonl \
  infilling \
  --post_processors truncate_line_until_block
```

For Control-Flow Expression Completion:

```bash
mkdir -p cache outputs_control
python generate.py \
  NAME_OF_YOUR_MODEL \
  control \
  cache/NAME_OF_YOUR_MODEL.json \
  outputs_control/NAME_OF_YOUR_MODEL-fim-tc.jsonl \
  infilling \
  --post_processors truncate_control
```

For API Function Call Completion:

```bash
mkdir -p cache outputs_api
python generate.py \
  NAME_OF_YOUR_MODEL \
  api \
  cache/NAME_OF_YOUR_MODEL.json \
  outputs_api/NAME_OF_YOUR_MODEL-fim-ta.jsonl \
  infilling \
  --post_processors truncate_api_call
```

### Prompts and Post-Processing

Select a prompt strategy based on your model's design and the task specifics, for details, please refer to code or our
paper:

- `infilling`: Prefix-Suffix-Middle (PSM)
- `reverse_infilling`: Suffix-Prefix-Middle (SPM)
- `left_to_right`: Left-to-Right (L2R)
- `prefix_feeding`: Instructed Prefix Feeding (IPF)
- `fewshot`: One-Shot (1S)

For chat models that outputs Markdown-style outputs, prepend `extract_code` to the list of `--post_processors` for code
extraction from Markdown outputs.

For `prefix_feeding` mode, use `--block_comments` to mask the logits to prevent the model from generating comments,
which usually improve output quality by avoiding comment repetition.

### Evaluate and Upload

Evaluate your model's performance using `evaluate.py` and submit the results for leaderboard inclusion:

```bash
mkdir -p results_block
python evaluate.py \
  block \
  outputs_block/NAME_OF_YOUR_MODEL-fim-tb.jsonl \
  results_block/NAME_OF_YOUR_MODEL-fim-tb.jsonl
```

```bash
mkdir -p results_control
python evaluate.py \
  control \
  outputs_control/NAME_OF_YOUR_MODEL-fim-tc.jsonl \
  results_control/NAME_OF_YOUR_MODEL-fim-tc.jsonl
```

```bash
mkdir -p results_api
python evaluate.py \
  api \
  outputs_api/NAME_OF_YOUR_MODEL-fim-ta.jsonl \
  results_api/NAME_OF_YOUR_MODEL-fim-ta.jsonl
```

The result files `results_block/NAME_OF_YOUR_MODEL-fim-tb.jsonl`, `results_control/NAME_OF_YOUR_MODEL-fim-tc.jsonl`,
and `results_api/NAME_OF_YOUR_MODEL-fim-tb.jsonl` can be submitted to
[https://safimbenchmark.com/submit](https://safimbenchmark.com/submit) to put your model on the leaderboard. Note that
this page requires login using your Google account.


## Impact of Data Contamination

SAFIM is sourced from code submitted from April 2022 to January 2023. The training data of some LLMs evaluated in our
paper has overlapping date ranges with SAFIM. To assess the impact of potential data contamination on our evaluation
results, we collected a new test dataset for the algorithmic block completion task based on Codeforces contests,
indicated as `block_v2` in our [Huggingface dataset](https://huggingface.co/datasets/gonglinyuan/safim). Here is an example of running
DeepSeek-Coder-1.3B (with training data cutoff Feb 2023, overlapping with the date range of SAFIM) using
Prefix-Suffix-Middle infilling prompt on this new test set. This experiment requires a single GPU.

Generate:

```bash
mkdir -p cache outputs_block_v2
python generate.py \
  deepseek-ai/deepseek-coder-1.3b-base \
  block_v2 \
  cache/deepseek-coder-1.3b-base.json \
  outputs_block_v2/deepseek-coder-1.3b-base-fim-tb.jsonl \
  infilling \
  --post_processors truncate_line_until_block
```

Evaluation:

```bash
mkdir -p results_block_v2
python evaluate.py \
  block_v2 \
  outputs_block_v2/deepseek-coder-1.3b-base-fim-tb.jsonl \
  results_block_v2/deepseek-coder-1.3b-base-fim-tb.jsonl
```

Show results:

```bash
python show_detailed_results.py \
  block_v2 \
  results_block_v2/deepseek-coder-1.3b-base-fim-tb.jsonl
```

The expected outcome is:

```
deepseek-coder-1.3b-base-fim-tb,41.194030,53.745928,48.534483,54.424779,46.051192,76,161,136,30,54,1659,350
```

So the Pass@1 is 46.05%. In [our paper](https://arxiv.org/abs/2403.04814), we discussed these experimental results in Appendix A.9 "Further Analysis on Data Contamination".

## Citation

```
@article{
    safim,
    title={Evaluation of {LLM}s on Syntax-Aware Code Fill-in-the-Middle Tasks},
    url={http://arxiv.org/abs/2403.04814},
    note={arXiv:2403.04814 [cs]},
    number={arXiv:2403.04814},
    publisher={arXiv},
    author={Gong, Linyuan and Wang, Sida and Elhoushi, Mostafa and Cheung, Alvin},
    year={2024},
    month=mar
}
```

## Acknowledgement

The SAFIM benchmark is partially derived from problem descriptions and code solutions from
[https://codeforces.com](https://codeforces.com). According to the license of CodeForces, you may publish the texts of
Codeforces problems in any open sources, but you must preserve a direct link to the site.

The evaluation of SAFIM is partially based on [ExecEval](https://github.com/ntunlp/ExecEval) framework, detailed in paper
[xCodeEval: A Large Scale Multilingual Multitask Benchmark for Code Understanding, Generation, Translation and Retrieval](https://arxiv.org/abs/2303.03004).
