import argparse
import json
import platform
import sys
from pathlib import Path
from .data import download,build_cases,install_archive,audit_splits
from .model import Ollama
from .policies import run_policy
from .evaluate import score,load_jsonl,read_annotations

def write_jsonl(path, rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in rows),encoding='utf-8')

def main():
    p=argparse.ArgumentParser(description="SciFact recovery experiment (no fabricated metrics)")
    sub=p.add_subparsers(dest="command",required=True)
    d=sub.add_parser("download");d.add_argument("--data",default="data/scifact")
    i=sub.add_parser("import-archive");i.add_argument("--archive",required=True);i.add_argument("--data",default="data/scifact")
    q=sub.add_parser("audit-splits");q.add_argument("--data",default="data/scifact")
    s=sub.add_parser("sample");s.add_argument("--data",default="data/scifact");s.add_argument("--split",choices=["train","dev"],default="dev");s.add_argument("--limit",type=int,default=60);s.add_argument("--seed",type=int,default=42);s.add_argument("--out",default="runs/sample.jsonl")
    r=sub.add_parser("run");r.add_argument("--sample",default="runs/sample.jsonl");r.add_argument("--out",default="runs/predictions.jsonl");r.add_argument("--model",default="llama3.2:3b");r.add_argument("--url",default="http://127.0.0.1:11434");r.add_argument("--initial-k",type=int,default=3);r.add_argument("--expanded-k",type=int,default=8);r.add_argument("--threshold",type=float,default=0.7)
    e=sub.add_parser("evaluate");e.add_argument("--sample",default="runs/sample.jsonl");e.add_argument("--predictions",default="runs/predictions.jsonl");e.add_argument("--annotations");e.add_argument("--out",default="runs/metrics.json")
    a=p.parse_args()
    if a.command=="download": print(json.dumps(download(Path(a.data)),indent=2))
    elif a.command=="import-archive": print(json.dumps(install_archive(Path(a.archive),Path(a.data)),indent=2))
    elif a.command=="audit-splits": print(json.dumps(audit_splits(Path(a.data)),indent=2))
    elif a.command=="sample":
        cases=build_cases(Path(a.data),split=a.split,limit=a.limit,seed=a.seed)
        write_jsonl(Path(a.out),[c.model_dump() for c in cases]);print(f"Sampled {len(cases)} labeled claim-document pairs -> {a.out}")
    elif a.command=="run":
        from .data import Case
        cases=[Case.model_validate(c) for c in load_jsonl(a.sample)]
        if not (0<a.initial_k<=a.expanded_k and 0<=a.threshold<=1):p.error("Invalid k or threshold")
        llm=Ollama(model=a.model,url=a.url)
        outputs=[]
        for case in cases:
            for policy in ("B0","B1","B2","B3","B4"):
                try:outputs.append(run_policy(policy,case,llm,a.initial_k,a.expanded_k,a.threshold))
                except Exception as ex:
                    outputs.append({"case_id":case.case_id,"policy":policy,"final_claim":None,"action":"ERROR","reason":str(ex),"model_calls":0})
            write_jsonl(Path(a.out),outputs)  # resumable partial results on interruption; rerun overwrites
        meta={"model":a.model,"url":a.url,"initial_k":a.initial_k,"expanded_k":a.expanded_k,"threshold":a.threshold,
              "python":sys.version,"platform":platform.platform(),"cases":len(cases),"outputs":len(outputs),"error_outputs":sum(o["action"]=="ERROR" for o in outputs)}
        Path(a.out+".meta.json").write_text(json.dumps(meta,indent=2)+"\n")
        print(json.dumps(meta,indent=2))
    elif a.command=="evaluate":
        from .data import Case
        cases=[Case.model_validate(c) for c in load_jsonl(a.sample)]
        anns=read_annotations(a.annotations) if a.annotations else None
        metrics=score(load_jsonl(a.predictions),cases,anns)
        Path(a.out).parent.mkdir(parents=True,exist_ok=True)
        Path(a.out).write_text(json.dumps(metrics,indent=2)+"\n")
        print(json.dumps(metrics,indent=2))
if __name__=="__main__":main()
