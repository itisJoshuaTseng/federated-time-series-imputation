#!/usr/bin/env python3
"""Extract key metrics from all experiment JSONs."""
import json, glob, os

files = sorted(glob.glob('logs/saits/E*.json'))
for f in files:
    d = json.load(open(f))
    name = d.get('experiment_name','?')
    mode = d.get('mode','?')
    rounds = d.get('num_rounds','')
    best = d.get('best_mae','')
    cfg = d.get('config',{})
    agg = cfg.get('federation',{}).get('aggregation','')
    nc = cfg.get('federation',{}).get('num_clients','')
    mu = cfg.get('federation',{}).get('mu','')
    split = cfg.get('data',{}).get('split_strategy','')
    
    if mode == 'federated':
        final = d.get('final_eval',{}).get('global',{})
        mae = final.get('mae', best)
        fair = d.get('final_eval',{}).get('fairness',{})
        cv = fair.get('mae_cv','N/A')
        gap = fair.get('mae_gap','N/A')
        std = fair.get('mae_std','N/A')
        cv_str = f"{cv:.3f}" if isinstance(cv, float) else str(cv)
        gap_str = f"{gap:.4f}" if isinstance(gap, float) else str(gap)
        std_str = f"{std:.4f}" if isinstance(std, float) else str(std)
        print(f"{name:<35s} agg={agg:<8s} split={split:<14s} nc={nc} mu={mu} rounds={rounds} MAE={mae:.4f} std={std_str} gap={gap_str} CV={cv_str}")
    elif mode == 'centralized':
        mae = d.get('test_metrics',{}).get('mae','')
        print(f"{name:<35s} mode=centralized MAE={mae:.4f}")
    elif mode == 'local':
        mae = d.get('summary',{}).get('avg_test_mae','')
        std = d.get('summary',{}).get('std_test_mae','')
        print(f"{name:<35s} mode=local split={split:<14s} MAE={mae:.4f} std={std:.4f}")
