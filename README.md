# Semantic genericness detection for knowledge-graph drug repurposing

Code for *Detecting Generic Concepts in Knowledge-Graph Drug Repurposing: Expert Validation and
Query-Level Filtering* (Hodges, Bazarkulov, Koslicki, Roach, Ramsey).

The repository validates ontology information content (IC) against a blinded expert panel, builds a
genericness score from concept text alone (a label-free generality axis in SapBERT space plus a
fine-tuned BioLORD-2023 classifier, averaged over three seeds), and compares answer filters on the
complete answer lists of ARAX `treats` queries for 200 development and 200 replication diseases.

This is a code-only release. Data, intermediate files, model checkpoints, raw rating forms and ARAX
responses are not included; they are available from the corresponding author (see *Inputs*).

## Layout

| Path | Contents |
|---|---|
| `paths.py` | Repository and external-input locations shared by every script |
| `build_nodes.py`, `semantic_ic.py`, `generality_axis.py` | Node table, text-mined parent/child pairs, label-free generality axis |
| `axis_pair_ablation.py`, `combine.py`, `report_tables.py`, `seal_sample_a.py` | Axis robustness, degree-quintile AUCs, sealed Sample A axis scores |
| `figures.py` | Manuscript figures 1 to 4 and supplementary figure S1, to Frontiers specifications |
| `panel/`, `rq1_report/` | Expert-panel Sample A (draw, rating forms, combined ratings) and the IC-versus-panel analysis |
| `sample_b/` | Sample B (concepts without IC): draw and panel analysis |
| `v1/` | The parts of the earlier linear score that v2 depends on: training set, BioLORD embeddings, Sample A reference table, Sample B sealed scores |
| `v2/` | The semantic score: DrugCentral split, variant selection, fine-tuning, per-seed and ensemble evaluation, answer scoring, panel and filter numbers |
| `rq3/` | Query-level evaluation: local ARAX client, block-list replica, disease draws, answer scoring, filter arms, replication and DrugCentral split |
| `llm_panel/` | LLM audit: expert items, stratified sample, judge inference, validation and audit analyses |

## Environments

Python 3.12. `requirements-analysis.txt` covers the CPU steps; `requirements-gpu.txt` (torch,
transformers, vLLM) is needed for embedding, fine-tuning, answer scoring and the LLM judges.

```
python -m venv .venv && . .venv/bin/activate
pip install -r requirements-analysis.txt      # add -r requirements-gpu.txt for the GPU steps
```

## Inputs

External locations are read from environment variables in `paths.py`:

| Variable | Input |
|---|---|
| `SEMSCORE_TIER0_DIR` | Translator tier-0 knowledge graph, build 2026-06-21 (`nodes.jsonl`, `edges.jsonl`) |
| `SEMSCORE_EMBEDDINGS` | SapBERT (`cambridgeltl/SapBERT-from-PubMedBERT-fulltext`, CLS pooling) embeddings of `"name: description"` for all 1,754,754 nodes, in `nodes.jsonl` order |
| `SEMSCORE_RTX_DIR` | The ARAX code base (RTX, commit `2d1fb2d1`) with its production block-list |
| `SEMSCORE_TESTS` | Translator test suite `sprint_6_tests.json` |
| `SEMSCORE_PREDICTED_IC_DIR` | Node degree and predicted-IC tables used by `report_tables.py` |

Files placed in the repository tree before running (available from the corresponding author):

- `data/positive_generic.parquet`, `data/llm_confirmed_generics.txt`, `data/llm_hard_negatives.txt`,
  `data/confirmed_generics.txt`, `data/hard_negatives.txt`: curated and LLM-reviewed label files.
- `panel/data/nodes_table.parquet`: node table used to draw Sample A.
- The Sample A rating workbook (passed to `panel/sample_a_combine.py`) and `sample_b/data/sampleB.xlsx`.

The query-level steps need a local ARAX server at `http://localhost:5001` running the commit above.

## Reproducing the results

Run from the repository root, in this order. Each script's module docstring lists its inputs and outputs.

**1. Generality axis and label-free checks**
```
python build_nodes.py
python semantic_ic.py
python generality_axis.py           # held-out pairwise accuracy
python axis_pair_ablation.py        # agreement between disjoint halves of the pairs
python combine.py
python report_tables.py ic_axis ic_zsum   # degree-quintile AUCs
python seal_sample_a.py
```

**2. Expert panel, Sample A, and IC versus the panel**
```
python panel/sample_a_build.py
python panel/sample_a_forms_build.py
python panel/sample_a_combine.py <sample_a_workbook.xlsx>
python rq1_report/analysis_rq1.py   # rq1_report/numbers.json
```

**3. v1 dependencies** (GPU for embedding)
```
python v1/build_training_set.py
python v1/embed_biolord.py
python v1/train_v1.py
```

**4. Sample B**
```
python sample_b/sample_b_build.py
python v1/seal_sample_b.py
python sample_b/analysis_sample_b.py
```

**5. Development query set** (local ARAX server)
```
python rq3/extract_drugcentral.py
python rq3/draw_queries.py
python rq3/fidelity_run.py
python rq3/run_wide.py
RQ3_RESPONSES=responses_wide RQ3_SCORES=answer_scores_wide.parquet python rq3/score_answers.py
RQ3_SET=wide python rq3/arms_wide.py 0 50        # hand-list removals used by v2 selection
```

**6. Semantic score (v2)** (GPU)
```
python v2/split_drugcentral.py
python v2/train_v2.py                              # training-set variant selection
FT_SEED=2654 FT_TAG=       python v2/finetune_v2.py
FT_SEED=2    FT_TAG=_seed2 python v2/finetune_v2.py
FT_SEED=3    FT_TAG=_seed3 python v2/finetune_v2.py
python v2/ft_eval_ab.py && python v2/ft_eval_ab.py _seed2 && python v2/ft_eval_ab.py _seed3
python v2/score_ensemble.py rq3/data/answer_scores_wide.parquet v2/wide_answers_ft_ens.parquet
python v2/query_features.py
python v2/panel_numbers.py                         # v2/numbers_v2.json
```

**7. Filters and replication**
```
python rq3/draw_queries_heldout.py
RQ3_QUERIES=rq3/data/query_set_heldout.csv RQ3_RESPONSES=rq3/responses_heldout \
  RQ3_LOG=rq3/logs/run_heldout.log python rq3/run_wide.py
cp rq3/data/query_set_wide.csv rq3/data/query_set_wide_rerun.csv
RQ3_QUERIES=rq3/data/query_set_wide_rerun.csv RQ3_RESPONSES=rq3/responses_wide_rerun \
  RQ3_LOG=rq3/logs/run_wide_rerun.log python rq3/run_wide.py
for s in heldout wide_rerun; do
  RQ3_RESPONSES=responses_$s RQ3_SCORES=answer_scores_$s.parquet python rq3/score_answers.py
  python v2/score_ensemble.py rq3/data/answer_scores_$s.parquet v2/${s}_answers_ft_ens.parquet
done
for s in wide heldout wide_rerun; do for ic in 50 80; do
  RQ3_SET=$s python rq3/arms_wide.py 0 $ic v2/${s}_answers_ft_ens.parquet score_ft_ens
done; done
python rq3/collateral_split.py wide heldout wide_rerun   # rq3/results_heldout/HELDOUT_RESULTS.md
```
The re-run of the development diseases must not reuse ARAX's KP query cache: timed-out knowledge-provider
calls are cached and replayed, so clear them before running.

**8. LLM audit** (GPU)
```
python llm_panel/build_expert_items.py
python llm_panel/draw_sample.py
for j in gemma nemo phi; do python llm_panel/infer.py $j; done
python llm_panel/analyze_validation.py
python llm_panel/analyze_audit.py
```

**9. Manuscript figures**
```
python figures.py                   # results/figures
python figures.py <output directory>
```
Each figure is written as a vector PDF and as a 300 dpi RGB LZW TIFF, 180 mm wide (Frontiers'
two-column width), with figure text at 8 pt or larger and data lines at 2 pt or heavier.

Seeds are fixed in the scripts (2654, with 2655 for the Sample B and replication draws).

## Code style

`pyproject.toml` configures ruff. Run `ruff format . && ruff check .` before committing.

## License

MIT; see `LICENSE`.

## Citation

A citation will be added on publication.
