"""SciFact release downloader, validation and deterministic case construction."""
import hashlib
import json
import random
from pathlib import Path
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator
import requests

RELEASE_URL = "https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz"
FILES = ("claims_train.jsonl", "claims_dev.jsonl", "corpus.jsonl")

class Evidence(BaseModel):
    model_config = ConfigDict(extra="allow")
    label: Literal["SUPPORT", "CONTRADICT"]
    sentences: list[int]
    @field_validator("sentences")
    @classmethod
    def valid_sentences(cls, v):
        if any(i < 0 for i in v): raise ValueError("negative rationale index")
        return v

class Claim(BaseModel):
    model_config = ConfigDict(extra="allow")
    id: int
    claim: str = Field(min_length=1)
    evidence: dict[str, list[Evidence]] = Field(default_factory=dict)
    cited_doc_ids: list[int] = Field(default_factory=list)

class Paper(BaseModel):
    model_config = ConfigDict(extra="allow")
    doc_id: int
    title: str
    abstract: list[str]
    structured: bool | None = None

class Case(BaseModel):
    case_id: str
    claim_id: int
    doc_id: int
    claim: str
    title: str
    evidence: list[str]
    rationale_indices: list[int]
    gold_label: Literal["SUPPORT", "CONTRADICT"]
    source_split: str

def read_jsonl(path: Path, model):
    if not path.exists(): raise FileNotFoundError(f"Missing {path}; run download first")
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            try: yield model.model_validate_json(line)
            except Exception as exc: raise ValueError(f"{path}:{line_no}: {exc}") from exc

def install_archive(archive: Path, dest: Path, source: str | None = None):
    """Import the official SciFact release from a local tar.gz, without unsafe extraction."""
    import tarfile
    dest.mkdir(parents=True, exist_ok=True)
    payloads = {}
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar:
            if not member.isfile():
                continue
            if Path(member.name).name not in FILES:
                continue
            name = Path(member.name).name
            if name in payloads:
                raise ValueError(f"Duplicate archive member: {name}")
            if member.size > 100_000_000:
                raise ValueError(f"Oversized dataset member: {name}")
            stream = tar.extractfile(member)
            if stream is None:
                raise ValueError(f"Unreadable archive member: {name}")
            payloads[name] = stream.read()
    missing = set(FILES) - set(payloads)
    if missing:
        raise ValueError(f"Missing SciFact release members: {sorted(missing)}")
    manifest = {}
    for name, raw in payloads.items():
        # Validate every record before installing any files.
        cls = Paper if name == "corpus.jsonl" else Claim
        for line_no, line in enumerate(raw.splitlines(), 1):
            if line.strip():
                try:
                    cls.model_validate_json(line)
                except Exception as exc:
                    raise ValueError(f"{name}:{line_no}: {exc}") from exc
        manifest[name] = {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw), "records": sum(bool(line.strip()) for line in raw.splitlines())}
    for name, raw in payloads.items():
        (dest / name).write_bytes(raw)
    manifest["provenance"] = {"source": source or str(archive), "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest()}
    (dest / "download_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest

def download(dest: Path, timeout=40):
    import tempfile
    response = requests.get(RELEASE_URL, timeout=timeout)
    response.raise_for_status()
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / "data.tar.gz"
        archive.write_bytes(response.content)
        return install_archive(archive, dest, source=RELEASE_URL)

def audit_splits(dest: Path):
    """Report source-paper overlap, not an assertion of independent splits."""
    ids = {}
    for split in ("train", "dev"):
        claims = list(read_jsonl(dest / f"claims_{split}.jsonl", Claim))
        ids[split] = set(int(d) for c in claims for d in c.evidence)
    return {"train_source_papers": len(ids["train"]), "dev_source_papers": len(ids["dev"]),
            "shared_source_papers": len(ids["train"] & ids["dev"]),
            "shared_ids": sorted(ids["train"] & ids["dev"])}

def build_cases(dest: Path, split="dev", seed=42, limit=60, balanced=True):
    if split not in ("train", "dev"): raise ValueError("Use labeled train/dev only")
    papers = {p.doc_id:p for p in read_jsonl(dest / "corpus.jsonl", Paper)}
    cases=[]
    for claim in read_jsonl(dest / f"claims_{split}.jsonl", Claim):
        for doc_id, annotations in claim.evidence.items():
            paper = papers.get(int(doc_id))
            if paper is None: raise ValueError(f"Unknown document {doc_id}")
            for idx, ann in enumerate(annotations):
                if any(i>=len(paper.abstract) for i in ann.sentences):
                    raise ValueError(f"Out-of-range rationale index for claim {claim.id}")
                cases.append(Case(case_id=f"{split}:{claim.id}:{doc_id}:{idx}",claim_id=claim.id,
                    doc_id=paper.doc_id,claim=claim.claim,title=paper.title,evidence=paper.abstract,
                    rationale_indices=ann.sentences,gold_label=ann.label,source_split=split))
    # Deduplicate at claim-paper-label level, while retaining original evidence rationale for each case.
    unique={}
    for c in cases: unique.setdefault((c.claim_id,c.doc_id,c.gold_label),c)
    cases=list(unique.values())
    rng=random.Random(seed)
    if balanced:
        pos=[c for c in cases if c.gold_label=="SUPPORT"]
        neg=[c for c in cases if c.gold_label=="CONTRADICT"]
        rng.shuffle(pos);rng.shuffle(neg)
        each=min(len(pos),len(neg),limit//2)
        chosen=pos[:each]+neg[:each]
        remaining=[c for c in cases if c not in chosen]
        rng.shuffle(remaining)
        chosen+=remaining[:max(0,limit-len(chosen))]
    else:
        rng.shuffle(cases);chosen=cases[:limit]
    rng.shuffle(chosen)
    return chosen
