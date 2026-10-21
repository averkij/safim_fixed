"""If a model writes the correct block and then keeps going (copying the next lines of the suffix, as
FIM models commonly do when they fail to emit EOS), does truncate_line_until_block cut it back?"""
import sys, json, gzip, random, collections, os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.argv = ['x', 'block_completion', '0', 'block']  # reuse ppcheck's loader without running its loop
src = open(os.path.join(HERE, 'ppcheck.py')).read()
src = src[:src.index('data = sys.argv[1]')]
exec(src)
rows = [json.loads(l) for l in gzip.open(os.path.join(HERE, 'data', 'block_completion.jsonl.gz'), 'rt')]
random.Random(7).shuffle(rows)
per = collections.Counter(); sel = []
for r in rows:
    if per[r['lang']] < 120:
        per[r['lang']] += 1; sel.append(r)
res = collections.defaultdict(collections.Counter)
for k in (1, 3, 8):
    for r in sel:
        ep, es = r['eval_prompt'].split('{{completion}}')
        extra = ''.join(es.splitlines(keepends=True)[:k])
        # lines of the suffix right after the hole (the hole usually ends mid-line; start from the next line)
        nxt = es.split('\n', 1)[1] if '\n' in es else ''
        extra = ''.join(nxt.splitlines(keepends=True)[:k])
        comp = r['ground_truth'] + '\n' + extra
        out = g['truncate_line_until_block'](r, comp)
        key = 'restored_exact_GT' if out.strip() == r['ground_truth'].strip() else ('kept_overrun' if out.strip().startswith(r['ground_truth'].strip()) and len(out.strip()) > len(r['ground_truth'].strip()) else 'other')
        res[k][key] += 1
for k in res: print(f'overrun of next {k} suffix line(s):', dict(res[k]), 'of', sum(res[k].values()))
