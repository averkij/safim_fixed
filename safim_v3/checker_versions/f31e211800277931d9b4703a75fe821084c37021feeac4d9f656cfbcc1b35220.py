"""Named constructive-output checkers; no execution of supplied code."""


def memory_game_turns(cards):
    """Literal greedy memory-game simulation, linear in number of cards."""
    known = {}
    ready = []
    removed = set()
    position = turns = 0
    while position < len(cards) or ready:
        if ready:
            value = ready.pop()
            removed.add(value)
            known.pop(value, None)
            turns += 1
            continue
        first = cards[position]; position += 1
        if first in known:
            known.pop(first); removed.add(first)
        else:
            known[first] = 1
            second = cards[position]; position += 1
            if second == first:
                known.pop(first); removed.add(first)
            elif second in known:
                ready.append(second)
            else:
                known[second] = 1
        turns += 1
    return turns


def removable_tree(a, adjacency):
    """Existence criterion, separately checked against exhaustive small games."""
    odd = [x % 2 for x in a]
    for v in range(len(a)):
        if not odd[v] and not any(odd[u] for u in adjacency[v]):
            return False
    seen = set()
    for v in range(len(a)):
        if not odd[v] or v in seen:
            continue
        component = [v]; seen.add(v)
        for u in component:
            for w in adjacency[u]:
                if odd[w] and w not in seen:
                    seen.add(w); component.append(w)
        if len(component) % 2 == 0:
            return False
    return True


def triangle_interiors_overlap(a, b):
    # Exact separating-axis test. Equality permits shared edges/vertices.
    for tri in (a,b):
        for i in range(3):
            x,y=tri[i];u,v=tri[(i+1)%3];nx,ny=y-v,u-x
            pa=[nx*px+ny*py for px,py in a];pb=[nx*px+ny*py for px,py in b]
            if max(pa)<=min(pb) or max(pb)<=min(pa):return False
    return True


def digit_chain_matches(original, answer):
    """Literal S(x) witness check, without solving the rearrangement problem."""
    from collections import Counter
    if not answer or answer[0] == '0' or any(c not in '0123456789' for c in answer):
        return False
    if Counter(original) != Counter(answer):
        return False
    if len(answer) == 1:
        return True
    # |s| <= 100000: the first digit sum is <=900000, then <=54,
    # then <=13, then <=9. Its entire appended chain uses at most 11 digits.
    for cut in range(max(2, len(answer)-11), len(answer)):
        value = sum(ord(c)-48 for c in answer[:cut])
        tail = str(value)
        while value > 9:
            value = sum(int(c) for c in str(value))
            tail += str(value)
        if answer[cut:] == tail:
            return True
    return False


CHECKERS = {'cf_2192B', 'cf_2201B', 'cf_2193H', 'cf_2215E', 'cf_2204E'}


def check_constructive(name, input_text, output_text, expected_text=None):
    if name not in CHECKERS:
        raise ValueError('Unknown constructive checker: ' + str(name))
    inp = iter(input_text.split()); out = iter(output_text.split())
    expected=iter(expected_text.split()) if expected_text is not None else None
    try:
        for _ in range(int(next(inp))):
            if name == 'cf_2204E':
                if not digit_chain_matches(next(inp), next(out)):
                    return False
                continue
            n = int(next(inp))
            if name == 'cf_2192B':
                s = next(inp)
                count = int(next(out))
                possible = n % 2 == 0 or s.count('1') % 2 == 0
                if count == -1:
                    if possible: return False
                    continue
                if not possible or not 0 <= count <= n: return False
                chosen = [int(next(out))-1 for j in range(count)]
                if len(set(chosen)) != count or any(not 0 <= i < n for i in chosen): return False
                selected = set(chosen)
                if any((int(s[i]) ^ (count % 2) ^ (i in selected)) != 0 for i in range(n)): return False
            elif name == 'cf_2201B':
                k = int(next(inp)); response = next(out).lower()
                possible = n <= k <= 2*n-1
                if response == 'no':
                    if possible: return False
                    continue
                if response != 'yes' or not possible: return False
                cards = [int(next(out)) for j in range(2*n)]
                counts = [0]*(n+1)
                for value in cards:
                    if not 1 <= value <= n: return False
                    counts[value] += 1
                if any(count != 2 for count in counts[1:]): return False
                if memory_game_turns(cards) != k: return False
            elif name == 'cf_2215E':
                points=[(int(next(inp)),int(next(inp))) for j in range(n)]
                if expected is None:return False
                optimal=int(next(expected))
                for j in range(3*optimal):next(expected)
                count=int(next(out))
                if count!=optimal or not 0<=count<=2*n:return False
                triangles=[];seen=set()
                for j in range(count):
                    ids=tuple(int(next(out))-1 for k in range(3))
                    if len(set(ids))!=3 or any(not 0<=i<n for i in ids):return False
                    key=tuple(sorted(ids))
                    if key in seen:return False
                    seen.add(key);tri=tuple(points[i] for i in ids)
                    ordered=sorted(tri)
                    if min(ordered[0][1],ordered[2][1])<ordered[1][1]<max(ordered[0][1],ordered[2][1]):return False
                    if any(triangle_interiors_overlap(tri,other) for other in triangles):return False
                    triangles.append(tri)
            else:
                a = [int(next(inp)) for j in range(n)]
                adj = [[] for j in range(n)]
                for j in range(n-1):
                    u,v = int(next(inp))-1,int(next(inp))-1
                    adj[u].append(v); adj[v].append(u)
                response = next(out).lower()
                if response == 'no':
                    if removable_tree(a,adj): return False
                    continue
                if response != 'yes': return False
                order = [int(next(out))-1 for j in range(n)]
                if set(order) != set(range(n)): return False
                parity = [sum(a[u] for u in adj[v]) % 2 for v in range(n)]
                alive = [True]*n
                for v in order:
                    if a[v] % 2 == parity[v]: return False
                    alive[v] = False
                    for u in adj[v]:
                        if alive[u]: parity[u] ^= a[v] % 2
        return next(inp,None) is None and next(out,None) is None and (name!='cf_2215E' or next(expected,None) is None)
    except (ValueError, StopIteration, IndexError):
        return False
