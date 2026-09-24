"""Inventory and preserve original scalar reporting metrics from JSONL logs."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from analyze_residuals import metadata

FAMILIES = ('rb_fit/', 'rb_novelty/', 'coverage_novelty/', 'bonus_propagation/',
            'prior_stats/', 'state_density/', 'policy_eval/', 'coverage/',
            'rb_next/', 'sac/q_', 'sac/cal_', 'sac/actor_obj_')


def collect(root, out):
    rows=[];inventory=[]
    for run in sorted(root.iterdir()):
        if not (run/'policy_eval/mc.npz').exists():continue
        meta=metadata(run);path=run/'metrics.jsonl';counts={k:0 for k in FAMILIES}
        if path.exists():
            with path.open() as file:
                for line in file:
                    item=json.loads(line)
                    for key,value in item.items():
                        family=next((x for x in FAMILIES if key.startswith(x) or '/'+x in key),None)
                        if family and isinstance(value,(float,int)):
                            rows.append({**meta,'step':item.get('step'),'metric':key,'value':value})
                            counts[family]+=1
        for family,count in counts.items():
            inventory.append({**meta,'family':family,'logged_values':count,'log_present':path.exists()})
    pd.DataFrame(inventory).to_csv(out/'reporting_metric_inventory.csv',index=False)
    if rows:
        df=pd.DataFrame(rows);df.to_csv(out/'original_reporting_metrics.csv.gz',index=False)
        summary=df.groupby(['env','method','fc','pls','metric']).value.agg(['count','first','last','min','max','mean','std'])
        summary.to_csv(out/'original_reporting_summary.csv')
    print(f'Original logged scalar values: {len(rows)}')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();collect(a.root,a.out)
