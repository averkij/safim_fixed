import ast, re, gzip, json, collections, sys, os
HERE = os.path.dirname(os.path.abspath(__file__))
EVAL = os.path.join(HERE, '..', 'evaluate.py')
# verbatim copies of evaluate.py functions (evaluate.py itself needs tree-sitter at import time)
src = open(EVAL).read()
ns = {'ast': ast, 're': re}
start = src.index('def get_function_call_params'); end = src.index('def main')
exec(src[start:end], ns)
syntax_match = ns['syntax_match']

rows = [json.loads(l) for l in gzip.open(os.path.join(HERE, 'data', 'api_completion.jsonl.gz'), 'rt')]
py = [r for r in rows if r['lang'] == 'python']
print('api tasks', len(rows), 'python', len(py))

# 1) wrong function name, same args
def parse_call(s):
    try:
        t = ast.parse(re.sub(r'\s+', '', s), mode='eval').body
        return t if isinstance(t, ast.Call) else None
    except Exception:
        return None
n_call = n_noargs = n_swapped = 0
examples = []
for r in py:
    c = parse_call(r['ground_truth'])
    if c is None: continue
    n_call += 1
    if not c.args and not c.keywords: n_noargs += 1
    # replace the callee by a bogus name, keep arguments textually
    gt = r['ground_truth']
    i = gt.index('(')   # naive: first paren
    bogus = 'totally_wrong_function' + gt[i:]
    if parse_call(bogus) is not None and syntax_match(bogus, gt, 'python'):
        n_swapped += 1
        if len(examples) < 6: examples.append((gt, bogus))
print('python GT that parse as a Call (ast, whitespace-stripped):', n_call)
print('  ... with zero args/kwargs (any f() matches):', n_noargs)
print('  ... where replacing the callee with a bogus name still PASSES:', n_swapped)
for e in examples: print('   ', e)

# 2) whitespace-insensitivity
print(syntax_match('foo("a b")', 'foo("ab")', 'java'), '<- java string-literal "a b" vs "ab" judged equal?')
print(syntax_match('x.y(a)', 'x.y(a)', 'python'), syntax_match('os.remove(p)', 'shutil.rmtree(p)', 'python'), '<- os.remove(p) vs shutil.rmtree(p)')
print(syntax_match('a.b(c).d(e)', 'z.q(1).d(e)', 'python'), '<- chained call, inner chain ignored')
# non-python: check whitespace collapse hazards
nonpy = [r for r in rows if r['lang'] != 'python']
print('non-python api tasks:', len(nonpy))
# how many GTs rely on whitespace for meaning (e.g. 'new X', 'a instanceof b', 'x as y')
hz = [r['ground_truth'] for r in rows if re.search(r'\b(new|in|is|as|not|and|or|instanceof)\s+\w', r['ground_truth'])]
print('GT containing keyword+space+identifier (whitespace-removal hazard):', len(hz))
print('empty completion vs any GT:', any(syntax_match('', r['ground_truth'], r['lang']) for r in rows))
