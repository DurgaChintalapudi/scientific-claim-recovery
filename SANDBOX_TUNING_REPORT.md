# Sandbox tuning run — 2026-11-04

## What actually ran
- Existing Python prototype's test suite executed in sandbox: **12 tests passed**.
- Added a safety rule: a verifier's SUPPORT label cannot result in accepting a claim unless it also returns at least one in-range evidence sentence identifier. This is a procedural guard, not proof that the sentence entails the claim.
- Added regression tests for empty support evidence and threshold sensitivity.
- A synthetic/mock test at confidence 0.80 returns RETAIN at threshold 0.70 and ABSTAIN at threshold 0.90. These are programmed test inputs, **not model-generated predictions or measured model calibration**.

## What did not run
- No actual SciFact records were loaded; sandbox outbound HTTPS to GitHub failed with a connection error.
- No Ollama service/model is available here.
- No GPT inference, human annotation, or scientific claim-repair benchmark was executed.
- Consequently, no factuality, repair success, or risk–coverage research result can be reported.

## Next tuning settings (hypotheses, not optimized values)
- Start with threshold 0.70, 0.80, 0.90, chosen using a *development* subset only; never tune on final held-out test data.
- Preserve all raw model outputs, evidence sentence indices, and error cases.
- Use SciFact train for development and dev for held-out pilot evaluation; note potential paper overlap and group leakage checks.
- Check for unsupported repairs with independent annotators before calculating accepted-claim risk.
- B3 currently expands retrieval only within the same abstract; describe it as evidence expansion, not cross-document retrieval.

## Second sandbox run — official archive import hardening
- Corrected the downloader to use the official SciFact `data.tar.gz` release URL documented by AllenAI.
- Added `import-archive` for user-supplied release archives, avoiding the sandbox's blocked outbound DNS.
- Added archive member allowlisting, full schema validation before installation, SHA-256 provenance manifest, and `audit-splits` for overlapping source papers.
- Executed test suite: **14 passed**, including archive import and missing-member rejection.
- Re-attempted official release download: **DNS resolution failed**; no actual SciFact records loaded and no model inference conducted.
- Important: SciFact original annotations contain SUPPORT and CONTRADICT rationales; no native third INSUFFICIENT class for annotated pairs. Do not claim three-class empirical accuracy without additional labeling.

### Run after obtaining official archive
```bash
pip install -e '.[test]'
recovery-pilot import-archive --archive /path/to/data.tar.gz --data data/scifact
recovery-pilot audit-splits --data data/scifact
recovery-pilot sample --data data/scifact --split train --limit 60 --out runs/train_sample.jsonl
pytest -q
```
For a genuine LLM run, start a local Ollama service and use `recovery-pilot run --sample runs/train_sample.jsonl --model llama3.2:3b`. Do not interpret synthetic fixture tests as scientific findings.

## Local development verification — November 2026

Environment: Apple Silicon MacBook Pro, macOS.

- Automated test suite: 19 passed.
- Installed local models: qwen2.5:7b and llama3.2:3b.
- Qwen recovery pilot: 60 SciFact cases, five policies, 300 outputs, zero recorded execution errors.
- Exploratory human author review: 35 selected claims.
- AI-assisted evaluation: Gemini review, separately identified.
- Independent human validation and representative factuality estimates remain unavailable.

The automated tests verify software behavior, not the scientific correctness
of the model outputs. The Qwen pilot was run separately from the sandbox
development tests.
