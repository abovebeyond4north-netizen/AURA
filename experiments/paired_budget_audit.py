"""Audit matched public trials; failures charged the full interaction cap."""
import json
from collections import defaultdict
from pathlib import Path

def audit(rows, cap=65):
    groups=defaultdict(dict)
    for r in rows:
        key=(r['seed'],r['size'],r['column'],tuple(r['before']),tuple(r['after']),r['noise'],r['when'],r['reset_start'])
        name=r['controller']
        if name in groups[key] or len(r['actions'])>cap or type(r['success']) is not bool:
            raise ValueError('invalid trial')
        groups[key][name]=r
    if not groups or any(set(g)!={'Budget6','EntropyProbeCamera'} for g in groups.values()):
        raise ValueError('unpaired trials')
    def summarize(pairs):
        out={}
        for name in ('Budget6','EntropyProbeCamera'):
            rs=[g[name] for g in pairs]
            out[name]=dict(n=len(rs),success=sum(r['success'] for r in rs),actions=sum(len(r['actions']) for r in rs),success_actions=sum(len(r['actions']) for r in rs if r['success']),failure_adjusted=sum(len(r['actions']) if r['success'] else cap for r in rs))
        out['discordant']=sum(g['Budget6']['success']!=g['EntropyProbeCamera']['success'] for g in pairs)
        return out
    strata=defaultdict(list)
    for k,g in groups.items(): strata[(k[1],k[5],k[6])].append(g)
    return {'overall':summarize(list(groups.values())), 'strata':{str(k):summarize(v) for k,v in sorted(strata.items())}}

if __name__=='__main__':
    import sys
    print(json.dumps(audit(json.loads(Path(sys.argv[1]).read_text())),indent=2))
