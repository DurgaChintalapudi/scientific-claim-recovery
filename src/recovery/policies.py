"""Five policy baselines. Model verdicts are NOT human ground truth."""
from .model import verify, repair, Choice

def retrieval_order(claim, sentences):
    # Deterministic lexical retrieval over the *same cited abstract* (not external sources).
    words=set(w.strip('.,:;()[]').lower() for w in claim.split() if len(w)>3)
    return sorted(range(len(sentences)),key=lambda i:(-len(words.intersection(
        w.strip('.,:;()[]').lower() for w in sentences[i].split())),i))

def run_policy(policy, case, llm, initial_k=3, expanded_k=8, threshold=0.7):
    evidence=case.evidence
    ids=retrieval_order(case.claim,evidence)
    first=ids[:initial_k]
    initial=[evidence[i] for i in first]
    output={"case_id":case.case_id,"policy":policy,"original_claim":case.claim,
        "final_claim":None,"action":"ABSTAIN","verdict":None,"reason":"", "evidence_ids":[],"model_calls":0,
        "final_verification":None}
    before=llm.calls
    if policy=="B0":
        output.update(final_claim=case.claim,action="RETAIN",reason="Unverified retention baseline")
    else:
        v=verify(llm,case.claim,initial)
        output["verdict"]=v.model_dump()
        if v.label=="SUPPORT" and bool(v.evidence_sentence_ids) and v.confidence>=threshold:
            output.update(final_claim=case.claim,action="RETAIN",reason="Initial verifier accepted",
                evidence_ids=[first[i] for i in v.evidence_sentence_ids],final_verification=v.model_dump())
        elif policy=="B1":
            output["reason"]="Initial verifier rejected or uncertain"
        elif policy=="B2":
            attempt_repair(output,llm,case,initial,first,threshold)
        elif policy=="B3":
            expanded=ids[:expanded_k]
            ev=[evidence[i] for i in expanded]
            v2=verify(llm,case.claim,ev)
            if v2.label=="SUPPORT" and bool(v2.evidence_sentence_ids) and v2.confidence>=threshold:
                output.update(final_claim=case.claim,action="RETRIEVE",reason="Expanded evidence supports original",
                    evidence_ids=[expanded[i] for i in v2.evidence_sentence_ids],final_verification=v2.model_dump())
            else: output["reason"]="Expanded evidence did not support original"
        elif policy=="B4":
            # Pre-registered heuristic: contradiction -> repair; insufficiency/low confidence -> expand then repair.
            if v.label=="CONTRADICT" and v.confidence>=threshold:
                attempt_repair(output,llm,case,initial,first,threshold)
            else:
                expanded=ids[:expanded_k];ev=[evidence[i] for i in expanded]
                v2=verify(llm,case.claim,ev)
                if v2.label=="SUPPORT" and bool(v2.evidence_sentence_ids) and v2.confidence>=threshold:
                    output.update(final_claim=case.claim,action="RETRIEVE",reason="Expanded evidence supports original",
                    evidence_ids=[expanded[i] for i in v2.evidence_sentence_ids],final_verification=v2.model_dump())
                else: attempt_repair(output,llm,case,ev,expanded,threshold)
        else: raise ValueError(f"Unknown policy: {policy}")
    output["model_calls"]=llm.calls-before
    return output

def attempt_repair(output,llm,case,ev,ids,threshold):
    candidate=repair(llm,case.claim,ev)
    if not candidate.claim.strip():
        output["reason"]="No candidate repair";return
    checked=verify(llm,candidate.claim,ev)
    output["repair_candidate"]=candidate.claim
    output["repair_verdict"]=checked.model_dump()
    if checked.label=="SUPPORT" and bool(checked.evidence_sentence_ids) and checked.confidence>=threshold:
        output.update(final_claim=candidate.claim,action="REPAIR",reason="Model-verified repair; human validation pending",
            evidence_ids=[ids[i] for i in checked.evidence_sentence_ids],final_verification=checked.model_dump())
    else: output["reason"]="Repair failed model verification"
