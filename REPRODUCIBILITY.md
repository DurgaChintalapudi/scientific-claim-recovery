# Reproducibility protocol — Scientific Claim Recovery v0.1.1

**Status:** exploratory research prototype; this protocol does not constitute evidence of independent replication or verified factual accuracy.

## 1. Software and models

Clone the repository at https://github.com/DurgaChintalapudi/scientific-claim-recovery and check out the published `v0.1.1` tag. Use Python >=3.10, a working Ollama service, and sufficient memory for the local model.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[test]'
pytest -q
ollama pull qwen2.5:7b
ollama pull llama3.2:3b
ollama list
```

The Python client sends Ollama `/api/chat` requests with `temperature=0`, `seed=42`, `stream=false`, and a JSON schema. The client timeout is 180 seconds per request. Record the exact Ollama version, model manifest/digest, operating system, Python version, and hardware. Model outputs may differ across hardware, runtime versions, and model updates.

## 2. Dataset and sampling

SciFact's official release is obtained via `recovery-pilot download` (or `import-archive` for a separately acquired official archive). Record the archive hash and the generated data manifest. Follow SciFact's dataset license and attribution requirements.

```bash
recovery-pilot download --data data/scifact
recovery-pilot audit-splits --data data/scifact
recovery-pilot sample --data data/scifact --split dev --limit 60 --seed 42 --out runs/sample.jsonl
shasum -a 256 runs/sample.jsonl
```

**Historical pilot sample:** The uploaded 60-record pilot sample has SHA-256 `a06247de0bad17557a383ec79c977af926d68c29d9d48c04971d125a7f419528`, and comprises 30 SUPPORT / 30 CONTRADICT cases, all tagged `dev`. The sampling command above uses the *current code's* defaults; **do not assert it reproduces this particular historical file until its checksum matches**. For exact replication, publish the original sample file (if redistribution is permitted), or its complete IDs/order and sampling provenance, with verification hashes. Split overlap audit is diagnostic, not a guarantee of document-disjoint evaluation.

## 3. Five-policy Qwen recovery pilot

```bash
recovery-pilot run \
  --sample runs/sample.jsonl \
  --model qwen2.5:7b \
  --url http://127.0.0.1:11434 \
  --initial-k 3 \
  --expanded-k 8 \
  --threshold 0.7 \
  --out runs/qwen_recovery_predictions.jsonl

recovery-pilot evaluate \
  --sample runs/sample.jsonl \
  --predictions runs/qwen_recovery_predictions.jsonl \
  --out runs/qwen_recovery_metrics.json
```

The `run` command executes policies B0, B1, B2, B3, B4 in order for each case, and writes a JSONL prediction file and `<predictions>.meta.json` metadata file. Be aware that `run` **overwrites** its output during a new run; archive historical artifacts and hashes first. B3 **expands sentence context within the same scientific abstract**, not cross-document retrieval. B4 is an experimental policy heuristic, not proof of optimal adaptation.

## 4. Observed pilot results (historical output, not a predicted reproduction)

| Policy | Returned / 60 | Coverage | Model calls |
|---|---:|---:|---:|
| B0 | 60 | 100.0% | 0 |
| B1 | 21 | 35.0% | 60 |
| B2 | 41 | 68.3% | 119 |
| B3 | 29 | 48.3% | 99 |
| B4 | 47 | 78.3% | 134 |

The original-claim binary verifier's agreement with the SciFact labels was 32/60 (53.3%) in this recovery pilot. This is **not** independent scientific accuracy of the final claims. The pilot metadata reports 60 cases, 300 policy predictions and zero execution errors.

## 5. Interpreting evaluation and annotations

`evaluate` without `--annotations` reports output coverage, action frequencies, model-call counts, and original-label agreement; it correctly reports no independent human final-claim metrics. The published prototype's annotation input expects a CSV keyed by `(case_id, policy)` with fields `final_support` and `meaning_preservation`; `final_support` must be one of `SUPPORTED`, `CONTRADICTED`, `INSUFFICIENT`, or `NOT_APPLICABLE`. Inspect `src/recovery/evaluate.py` before using this option: its `human_useful_coverage` currently counts `meaning_preservation == MEANINGFUL`, which is **not the same vocabulary** as the author review's PRESERVED/PARTIAL/LOST. Do not directly feed the author review as human ground truth or claim that metric is comparable without a documented, justified mapping and independent review.

The 35 reviewed outputs were a purposefully selected **author-conducted** subset, with posthoc guided adjudications. Gemini judgments are AI-assisted evaluations, **not** independent human labels. Do not extrapolate subset counts to unbiased whole-population accuracy or report guided adjudications as blinded independent assessment.

## 6. Known limitations and release checklist

- Validate that the historical sample SHA-256 matches and publish its provenance.
- Archive complete model predictions, `.meta.json`, metrics, evidence references, prompts, code tag/commit and runtime versions; distinguish raw and adjudicated assessments.
- Add an end-to-end smoke test using a real Ollama model; unit-test success alone is not scientific replication.
- Never treat an LLM `SUPPORT` verdict as independent evidence of factual correctness; previous author review found a contradictory accepted repair.
- Separate original verifier verdict (`verdict`) from post-repair verification (`repair_verdict` / final-verification fields) in all downstream analyses.
- Determine whether public release of model outputs, dataset-derived text and annotations complies with the original dataset and any applicable third-party licenses.
- No matched-budget, independent-human validated superiority conclusion is supported by the pilot.
