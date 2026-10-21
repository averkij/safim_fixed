"""Round-trip test of SAFIM post-processors: feed the GROUND TRUTH through them; a correct
post-processor must return it unchanged (a perfect model must not be penalised).

Uses the repo's own function bodies (exec'd from prompt_utils.py / ast_utils.py) on top of
modern tree-sitter wheels (the repo builds grammars from source; versions differ slightly).
"""
import gzip, json, os, random, re, sys, collections
HERE = os.path.dirname(os.path.abspath(__file__))
import tree_sitter_python, tree_sitter_java, tree_sitter_cpp, tree_sitter_c_sharp
from tree_sitter import Language, Parser

REPO = os.path.join(HERE, "..")
LANGS = {"python": tree_sitter_python.language(), "java": tree_sitter_java.language(),
         "cpp": tree_sitter_cpp.language(), "csharp": tree_sitter_c_sharp.language()}
_parsers = {k: Parser(Language(v)) for k, v in LANGS.items()}


def get_parser(lang):
    return _parsers[lang]


# --- reuse the repo's visitor classes verbatim ---
au = open(f"{REPO}/ast_utils.py").read()
cls_src = au[au.index("class ASTVisitor"):au.index("def get_parser")]
ns = {}
exec(cls_src, ns)
ErrorCheckVisitor = ns["ErrorCheckVisitor"]

# --- reuse the repo's post-processors verbatim ---
pu = open(f"{REPO}/prompt_utils.py").read()
fn_src = pu[pu.index("def truncate_to_first_line"):pu.index("COMPLETION_PLACEHOLDER")]
fn_src += pu[pu.index("def truncate_line_until_parsable"):pu.index("def apply_prompt")]
g = {"get_parser": get_parser, "ErrorCheckVisitor": ErrorCheckVisitor}
exec(fn_src, g)

data = sys.argv[1]
N = int(sys.argv[2])
kind = sys.argv[3]  # block | control | api
rows = [json.loads(l) for l in gzip.open(os.path.join(HERE, "data", f"{data}.jsonl.gz"), "rt")]
random.Random(1).shuffle(rows)
per = collections.Counter()
sel = []
for r in rows:
    if per[r["lang"]] < N:
        per[r["lang"]] += 1
        sel.append(r)

stat = collections.defaultdict(collections.Counter)
bad = collections.defaultdict(list)
for r in sel:
    lang, gt = r["lang"], r["ground_truth"]
    try:
        if kind == "block":
            out = g["truncate_line_until_block"](r, gt)
        elif kind == "control":
            out = g["truncate_control"](r, gt)
        else:
            out = g["truncate_api_call"](gt)
    except AssertionError:
        stat[lang]["ASSERT_FAIL(empty-program has parse error)"] += 1
        continue
    stat[lang]["n"] += 1
    if out == gt:
        stat[lang]["identical"] += 1
    elif out.strip() == gt.strip():
        stat[lang]["identical_mod_ws"] += 1
    elif out.strip() == "":
        stat[lang]["BECAME_EMPTY"] += 1
        bad[lang].append((r["task_id"], gt[:120], out[:60]))
    else:
        stat[lang]["MODIFIED"] += 1
        bad[lang].append((r["task_id"], gt[:120], out[:120]))
print(data, kind)
for lang in stat:
    print(" ", lang, dict(stat[lang]))
for lang in bad:
    for b in bad[lang][:3]:
        print("   e.g.", lang, b)
