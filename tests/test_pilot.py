import json
from pathlib import Path
import pytest
from recovery.data import Claim,Paper,build_cases
from recovery.model import Verification,Repair
from recovery.policies import run_policy,retrieval_order
from recovery.evaluate import score

class FakeLLM:
    def __init__(self,answers):self.answers=iter(answers);self.calls=0
    def ask(self,schema,prompt):
        self.calls+=1
        return schema.model_validate(next(self.answers))

def fixture(tmp_path):
    corpus={"doc_id":7,"title":"Study","abstract":["Treatment reduces risk.","No increase was observed."],"structured":False}
    claim={"id":1,"claim":"Treatment increases risk.","evidence":{"7":[{"label":"CONTRADICT","sentences":[0]}]},"cited_doc_ids":[7]}
    (tmp_path/"corpus.jsonl").write_text(json.dumps(corpus)+'\n')
    (tmp_path/"claims_dev.jsonl").write_text(json.dumps(claim)+'\n')
    return build_cases(tmp_path,limit=10)[0]

def test_real_schema_fixture(tmp_path):
    case=fixture(tmp_path)
    assert case.gold_label=="CONTRADICT" and case.rationale_indices==[0]

def test_invalid_rationale_rejected(tmp_path):
    fixture(tmp_path)
    path=tmp_path/"claims_dev.jsonl"
    x=json.loads(path.read_text());x["evidence"]["7"][0]["sentences"]=[99]
    path.write_text(json.dumps(x)+'\n')
    with pytest.raises(ValueError,match="Out-of-range"):build_cases(tmp_path)

def test_bad_label_rejected():
    with pytest.raises(Exception): Claim.model_validate({"id":1,"claim":"c","evidence":{"1":[{"label":"UNKNOWN","sentences":[0]}]}})

def test_retain(tmp_path):
    case=fixture(tmp_path);out=run_policy("B0",case,FakeLLM([]))
    assert out["action"]=="RETAIN" and out["model_calls"]==0

def test_abstain(tmp_path):
    case=fixture(tmp_path)
    llm=FakeLLM([{"label":"CONTRADICT","confidence":.9,"reason":"no","evidence_sentence_ids":[0]}])
    assert run_policy("B1",case,llm)["final_claim"] is None

def test_repair(tmp_path):
    case=fixture(tmp_path)
    llm=FakeLLM([{"label":"CONTRADICT","confidence":.9,"reason":"no","evidence_sentence_ids":[0]},
                 {"claim":"Treatment reduces risk.","reason":"corrected","evidence_sentence_ids":[0]},
                 {"label":"SUPPORT","confidence":.95,"reason":"yes","evidence_sentence_ids":[0]}])
    out=run_policy("B2",case,llm)
    assert out["action"]=="REPAIR" and out["model_calls"]==3

def test_retrieve(tmp_path):
    case=fixture(tmp_path)
    llm=FakeLLM([{"label":"INSUFFICIENT","confidence":.8,"reason":"no","evidence_sentence_ids":[]},
                 {"label":"SUPPORT","confidence":.9,"reason":"yes","evidence_sentence_ids":[0]}])
    assert run_policy("B3",case,llm)["action"]=="RETRIEVE"

def test_adaptive(tmp_path):
    case=fixture(tmp_path)
    llm=FakeLLM([{"label":"CONTRADICT","confidence":.9,"reason":"no","evidence_sentence_ids":[0]},
                 {"claim":"Treatment reduces risk.","reason":"yes","evidence_sentence_ids":[0]},
                 {"label":"SUPPORT","confidence":.9,"reason":"yes","evidence_sentence_ids":[0]}])
    assert run_policy("B4",case,llm)["action"]=="REPAIR"

def test_metrics_do_not_invent_human_truth(tmp_path):
    case=fixture(tmp_path)
    out=run_policy("B0",case,FakeLLM([]))
    metrics=score([out],[case])
    assert metrics["B0"]["human_metrics_status"].startswith("NOT_AVAILABLE")
    assert "human_unsupported_acceptance_rate" not in metrics["B0"]

def test_bad_evidence_id(tmp_path):
    case=fixture(tmp_path)
    llm=FakeLLM([{"label":"SUPPORT","confidence":.9,"reason":"yes","evidence_sentence_ids":[99]}])
    with pytest.raises(ValueError,match="Invalid evidence index"):run_policy("B1",case,llm)


def test_support_without_evidence_abstains(tmp_path):
    case=fixture(tmp_path)
    llm=FakeLLM([{"label":"SUPPORT","confidence":.99,"reason":"unsupported","evidence_sentence_ids":[]},
                 {"label":"INSUFFICIENT","confidence":.91,"reason":"no support","evidence_sentence_ids":[]}])
    out=run_policy("B1",case,llm)
    assert out["action"]=="ABSTAIN"

def test_support_missing_ids_retry_can_recover_with_evidence(tmp_path):
    case=fixture(tmp_path)
    llm=FakeLLM([{"label":"SUPPORT","confidence":.9,"reason":"forgot IDs","evidence_sentence_ids":[]},
                 {"label":"SUPPORT","confidence":.9,"reason":"cites sentence","evidence_sentence_ids":[0]}])
    out=run_policy("B1",case,llm)
    assert out["action"]=="RETAIN" and out["model_calls"]==2


def test_threshold_changes_retention(tmp_path):
    case=fixture(tmp_path)
    answer={"label":"SUPPORT","confidence":.8,"reason":"test","evidence_sentence_ids":[0]}
    assert run_policy("B1",case,FakeLLM([answer]),threshold=.7)["action"]=="RETAIN"
    assert run_policy("B1",case,FakeLLM([answer]),threshold=.9)["action"]=="ABSTAIN"


def test_official_archive_import(tmp_path):
    import tarfile, io
    from recovery.data import install_archive, audit_splits
    corpus={"doc_id":7,"title":"Study","abstract":["A sentence."],"structured":False}
    claim={"id":1,"claim":"A claim.","evidence":{"7":[{"label":"SUPPORT","sentences":[0]}]},"cited_doc_ids":[7]}
    archive=tmp_path/"data.tar.gz"
    with tarfile.open(archive,"w:gz") as tar:
        for name,obj in [("data/corpus.jsonl",corpus),("data/claims_train.jsonl",claim),("data/claims_dev.jsonl",claim)]:
            raw=(json.dumps(obj)+"\n").encode(); info=tarfile.TarInfo(name);info.size=len(raw);tar.addfile(info,io.BytesIO(raw))
    dest=tmp_path/"imported";manifest=install_archive(archive,dest)
    assert manifest["corpus.jsonl"]["records"]==1
    assert audit_splits(dest)["shared_source_papers"]==1

def test_missing_archive_member_rejected(tmp_path):
    import tarfile, io
    from recovery.data import install_archive
    archive=tmp_path/"bad.tar.gz"
    with tarfile.open(archive,"w:gz") as tar:
        raw=b'{}\n';info=tarfile.TarInfo("corpus.jsonl");info.size=len(raw);tar.addfile(info,io.BytesIO(raw))
    with pytest.raises(ValueError,match="Missing SciFact"):
        install_archive(archive,tmp_path/"out")


def test_repair_final_verification_distinguishes_initial_verdict(tmp_path):
    case=fixture(tmp_path)
    llm=FakeLLM([
        {"label":"CONTRADICT","confidence":.9,"reason":"original wrong","evidence_sentence_ids":[0]},
        {"claim":"Treatment reduces risk.","reason":"corrected","evidence_sentence_ids":[0]},
        {"label":"SUPPORT","confidence":.95,"reason":"repaired supported","evidence_sentence_ids":[0]},
    ])
    out=run_policy("B2",case,llm)
    assert out["verdict"]["label"]=="CONTRADICT"
    assert out["repair_verdict"]["label"]=="SUPPORT"
    assert out["final_verification"]==out["repair_verdict"]

def test_repair_rejected_if_second_verification_is_not_support(tmp_path):
    case=fixture(tmp_path)
    llm=FakeLLM([
        {"label":"CONTRADICT","confidence":.9,"reason":"original wrong","evidence_sentence_ids":[0]},
        {"claim":"Treatment reduces risk.","reason":"candidate","evidence_sentence_ids":[0]},
        {"label":"INSUFFICIENT","confidence":.8,"reason":"not established","evidence_sentence_ids":[]},
    ])
    out=run_policy("B2",case,llm)
    assert out["action"]=="ABSTAIN"
    assert out["final_claim"] is None
    assert out["final_verification"] is None
    assert out["repair_candidate"]=="Treatment reduces risk."

def test_retain_has_explicit_final_verification(tmp_path):
    case=fixture(tmp_path)
    llm=FakeLLM([{"label":"SUPPORT","confidence":.9,"reason":"yes","evidence_sentence_ids":[0]}])
    out=run_policy("B1",case,llm)
    assert out["final_verification"]["label"]=="SUPPORT"

def test_b0_has_no_model_verification(tmp_path):
    case=fixture(tmp_path)
    out=run_policy("B0",case,FakeLLM([]))
    assert out["final_verification"] is None
