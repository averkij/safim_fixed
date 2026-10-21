import json,sys,collections,math
for f in sys.argv[1:]:
    r=[json.loads(l) for l in open(f)]
    print(f.split('/')[-1], len(r))
    tot=0;tp=0
    for l in ('cpp','java','python'):
        n=[x for x in r if x['lang']==l]
        if not n: continue
        p=sum(x['result']=='PASSED' for x in n); tot+=len(n); tp+=p
        se=math.sqrt(p/len(n)*(1-p/len(n))/len(n))
        c=collections.Counter(x['result'] for x in n)
        print(f"  {l:7s} n={len(n):4d} PASSED={100*p/len(n):5.1f}% (±{196*se:.1f})  ", dict(c))
