"""Separate known original labels from human-adjudicated final-claim validity."""
import csv
import json
from collections import Counter,defaultdict
from pathlib import Path

def load_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]

def score(predictions, cases, annotations=None):
    gold={c.case_id:c for c in cases}
    ann={} if annotations is None else {(a["case_id"],a["policy"]):a for a in annotations}
    by=defaultdict(list)
    for p in predictions:
        if p["case_id"] not in gold: raise ValueError("Prediction references unknown case")
        by[p["policy"]].append(p)
    report={}
    for policy,items in by.items():
        n=len(items);accepted=[p for p in items if p.get("final_claim")]
        result={"n":n,"coverage":len(accepted)/n if n else None,
                "actions":dict(Counter(p["action"] for p in items)),
                "total_model_calls":sum(p.get("model_calls",0) for p in items)}
        verified=[p for p in items if p.get("verdict")]
        # Binary original-label check only. SciFact has no gold INSUFFICIENT label for these cases.
        if verified:
            result["original_binary_verifier_accuracy"] = sum(p["verdict"]["label"]==gold[p["case_id"]].gold_label for p in verified)/len(verified)
            result["original_binary_verifier_n"]=len(verified)
        reviewed=[(p,ann[(p["case_id"],policy)]) for p in items if (p["case_id"],policy) in ann]
        if reviewed:
            valid=[(p,a) for p,a in reviewed if p.get("final_claim")]
            result["human_reviewed_n"]=len(reviewed)
            result["human_supported_accepted_n"]=sum(a["final_support"]=="SUPPORTED" for p,a in valid)
            result["human_unsupported_accepted_n"]=sum(a["final_support"]!="SUPPORTED" for p,a in valid)
            result["human_unsupported_acceptance_rate"]=(sum(a["final_support"]!="SUPPORTED" for p,a in valid)/len(valid) if valid else None)
            result["human_useful_coverage"]=(sum(a["final_support"]=="SUPPORTED" and a["meaning_preservation"]=="MEANINGFUL" for p,a in valid)/len(reviewed))
        else: result["human_metrics_status"]="NOT_AVAILABLE: no independent annotations supplied"
        report[policy]=result
    return report

def read_annotations(path):
    with open(path,newline="") as f:
        rows=list(csv.DictReader(f))
    for a in rows:
        if a["final_support"] not in ("SUPPORTED","CONTRADICTED","INSUFFICIENT","NOT_APPLICABLE"):
            raise ValueError("Invalid human final_support label")
    return rows
