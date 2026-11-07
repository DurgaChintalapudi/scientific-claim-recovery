"""Strict structured JSON responses from Ollama. No network calls at import time."""
import json
from typing import Literal
import requests
from pydantic import BaseModel, Field, ValidationError

class Verification(BaseModel):
    label: Literal["SUPPORT", "CONTRADICT", "INSUFFICIENT"]
    confidence: float = Field(ge=0,le=1)
    evidence_sentence_ids: list[int] = Field(default_factory=list)
    reason: str

class Repair(BaseModel):
    claim: str
    evidence_sentence_ids: list[int] = Field(default_factory=list)
    reason: str

class Choice(BaseModel):
    action: Literal["REPAIR", "RETRIEVE", "ABSTAIN"]
    reason: str

class Ollama:
    def __init__(self, model="llama3.2:3b",url="http://127.0.0.1:11434",timeout=180):
        self.model,self.url,self.timeout=model,url.rstrip("/"),timeout
        self.calls=0
    def ask(self, schema, prompt):
        response=requests.post(self.url+"/api/chat",json={"model":self.model,"stream":False,
            "format":schema.model_json_schema(),"options":{"temperature":0,"seed":42},
            "messages":[{"role":"system","content":"Return only valid JSON matching the schema. Do not fabricate evidence."},
                        {"role":"user","content":prompt}]},timeout=self.timeout)
        response.raise_for_status();self.calls+=1
        body=response.json()
        return schema.model_validate_json(body["message"]["content"])

def evidence_text(sentences):
    return "\n".join(f"[{i}] {s}" for i,s in enumerate(sentences))

def verify(llm, claim, evidence):
    """Verify a claim; retry once if support lacks the required evidence pointers.

    The retry does not relax the acceptance criterion. A verifier must return
    a non-empty, in-range sentence index to support an accepted output.
    """
    prompt=("Decide whether the provided source evidence SUPPORTS, CONTRADICTS, or is "
        "INSUFFICIENT for the claim. Do not infer beyond the text. "
        "Evidence sentences have zero-based [index] labels. For SUPPORT, list at least "
        "one exact supporting index in evidence_sentence_ids; if no sentence supports "
        "the claim, choose INSUFFICIENT or CONTRADICT.\nClaim: "+claim+
        "\nEvidence:\n"+evidence_text(evidence))
    v=llm.ask(Verification,prompt)
    if v.label=="SUPPORT" and not v.evidence_sentence_ids and evidence:
        v=llm.ask(Verification,
            "Your previous SUPPORT decision omitted mandatory supporting sentence IDs. "
            "Re-examine the evidence carefully. Output SUPPORT only if you can identify "
            "one or more *explicitly supporting* zero-based [index] sentence IDs. "
            "Otherwise output CONTRADICT or INSUFFICIENT and leave IDs empty. "
            "Do not invent IDs.\nClaim: "+claim+"\nEvidence:\n"+evidence_text(evidence))
    if any(i<0 or i>=len(evidence) for i in v.evidence_sentence_ids):
        raise ValueError("Invalid evidence index")
    return v

def repair(llm, claim, evidence):
    r=llm.ask(Repair,"Revise this claim to be supported by the provided source evidence. Preserve useful scientific specificity; "
        "do not invent measurements or populations. Return an empty claim if no defensible correction is possible.\nOriginal: "
        +claim+"\nEvidence:\n"+evidence_text(evidence))
    if any(i<0 or i>=len(evidence) for i in r.evidence_sentence_ids): raise ValueError("Invalid repair evidence index")
    return r
