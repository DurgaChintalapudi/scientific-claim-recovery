# Evidence-Constrained Recovery Policies for Scientific Claims

Reproducible **research prototype**, not a completed empirical study. No SciFact data, benchmark outcomes, or human labels are fabricated or bundled. The SciFact files are downloaded directly from the public AllenAI repository on the researcher's machine. A SHA-256 download manifest is written at download time.

## Setup (Python 3.10+)

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -e '.[test]'
pytest -q
```

Install Ollama from https://ollama.com and pull a model:

```bash
ollama pull llama3.2:3b
ollama serve  # only if the Ollama service is not already running
```

Run from the project root:

```bash
recovery-pilot download --data data/scifact
recovery-pilot sample --data data/scifact --split dev --limit 60 --seed 42 --out runs/sample.jsonl
recovery-pilot run --sample runs/sample.jsonl --model llama3.2:3b --out runs/predictions.jsonl
recovery-pilot evaluate --sample runs/sample.jsonl --predictions runs/predictions.jsonl --out runs/metrics.json
```

**Important:** SciFact train/dev has SUPPORT and CONTRADICT evidence annotations, not a native third INSUFFICIENT class. The 60-case sample is approximately balanced across the two annotated labels; the pilot **does not** silently invent a third class. The verifier may output INSUFFICIENT, but three-class accuracy is not reported. A later three-class study requires independently adjudicated insufficient cases.

## Methods

- B0: retain the original claim (no model call)
- B1: verify then abstain unless supported with confidence >= threshold
- B2: verify, try one constrained repair, verify repaired claim; otherwise abstain
- B3: verify, expand retrieval over the *same cited abstract*, verify again; otherwise abstain
- B4: fixed policy: high-confidence contradiction -> repair; otherwise expand evidence and then repair or abstain

All policies operate on identical sampled cases and the same model. Initial retrieval uses deterministic lexical overlap over the source abstract. Expanded retrieval increases the number of sentences from the **same abstract**, not an external search engine. For a true cross-document retrieval study, a new retrieval stage and associated budget controls must be added. B0 is intentionally unverified and should not be presented as a fair accuracy baseline for all tasks.

**Limitations:** The model verifies its own repaired text. This is a *model-based screening step*, not independent validation. Gold SciFact labels apply to original claim/document pairs only. Automated repair acceptance and useful-claim coverage cannot be claimed as research findings without human review. The same evidence sentence may be used in multiple pairs; group train/test by source paper and audit duplicates before final results. Output records and errors are saved rather than silently dropped. A run that contains errors must be flagged incomplete.

## Independent annotation

Use `configs/annotation_template.csv` as a template, filling one row for each `(case_id, policy)` pair. `final_support` must be one of `SUPPORTED`, `CONTRADICTED`, `INSUFFICIENT`, `NOT_APPLICABLE`; `meaning_preservation` is `MEANINGFUL`, `PARTIAL`, `LOST`, or `NOT_APPLICABLE`. Use `NOT_APPLICABLE` for abstentions. Reviewers must see original claim, original source abstract, output claim, and evidence identifiers. Use two blinded annotators and adjudication. Never label an abstention as a supported final claim.

```bash
recovery-pilot evaluate --sample runs/sample.jsonl --predictions runs/predictions.jsonl --annotations runs/adjudicated.csv
```

The evaluation script reports coverage, model calls, original-label verifier accuracy (for model-verdict cases only), and human-grounded metrics **only if annotations are provided**. These are pilot descriptive measures; confidence intervals, document-clustered bootstrap, and model-token/latency measurements remain future work.

## Data provenance and licensing

Source: https://github.com/allenai/scifact . Review original dataset and abstract licensing terms before redistribution. Downloaded data and generated outputs are excluded from this archive. No employer data or internal systems are used.

## Reproducibility caveats

Ollama and model versions, hardware, prompts, seed, dataset hashes, and any runtime errors should be preserved with each run. Temperature zero and seed 42 do not guarantee identical outputs across model runtimes. For rigorous benchmarking, run multiple seeds and report variance. This prototype has not been executed against a real Ollama server in the authoring environment.

## v3: evidence-pointer validation pilot
The original 60-case run had 29/30 B1 model SUPPORT decisions without any
sentence pointers. v3 retains the strict evidence-pointer acceptance gate and
adds one *additional verifier model call* when the verdict is SUPPORT with
empty evidence_sentence_ids. The retry must cite at least one valid passage
index or the policy still abstains. Extra calls increase measured inference
cost. No change was made to the human-grounded validation requirement.

**Run the identical sample again:**

```bash
recovery-pilot run --sample runs/sample.jsonl --model llama3.2:3b --out runs/predictions_v3.jsonl
recovery-pilot evaluate --sample runs/sample.jsonl --predictions runs/predictions_v3.jsonl --out runs/metrics_v3.json
```

Never overwrite the original predictions.jsonl or metrics.json. v3 remains
within-cited-abstract retrieval only. No final scientific validity metrics are
available without blinded human annotation.

## v0.1.1: clarify original-claim and final-claim judgments

`verdict` is the model's **initial verification of the original claim**, not the
verification of a repaired output. `repair_verdict` is the **post-repair**
verification; returned `REPAIR` outputs require this label to be `SUPPORT`
with in-range evidence pointers and confidence at least the threshold.
`final_verification` is a convenience field for the verifier decision that
caused the final output to be emitted (null for B0 and abstentions). It does
**not** provide independent factual ground truth and must not be presented
as human validation. Existing prediction files remain valid; old runs do
not contain `final_verification`, but `repair_verdict` is available.

Audit of the previously uploaded Qwen 2.5 7B run: 41 `REPAIR` outputs
were emitted (B2: 20; B4: 21); all 41 have `repair_verdict.label` equal to
`SUPPORT`. The 41 corresponding initial `verdict` labels were 24
`CONTRADICT` and 17 `INSUFFICIENT`, as expected prior to repair. These
counts describe *model decisions*, not actual repair correctness.

This change is metadata clarification and regression coverage only, **not**
a modification to the accept/reject decision rule. Do not silently merge
results from different models, evidence conditions, or protocols.
