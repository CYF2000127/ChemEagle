# Original Yufan pipeline: three-model comparison

18 November 2026. This adds original-upstream results to the existing six-PDF focused matrix. The paired table is in [EXPANDED_COMPARISON.md](EXPANDED_COMPARISON.md); machine-readable totals are in [yufan-model-comparison.json](yufan-model-comparison.json).

## Findings

The original pipeline returned 4/11 target experiments with Luna, 6/11 with Terra and 4/11 with Sol. All checked outcome and condition fields matched within those uniquely linked records; missing rows and failed exports explain the lower fixed-denominator totals. The focused workflow returned all 11 targets per model, but its supplied targets/reference context make this a comparison of two different extraction tasks, not a controlled model or architecture ranking.

The original epoxidation exports omitted 3ai/3aj with every model. The successful furan exports selected the product scheme without establishing the requested table entry. Four exports failed because prose preceded the final JSON, and Luna's cycloaddition run failed on malformed table-specialist JSON. The quaternary targets succeeded for all three models.

Graph limitations remain separate: Luna/Terra retain repeated invalid tetrahydropyran product graphs. Both Terra and Sol assign the same parseable cycloaddition graph to substrate 5a and catalyst (R)-4e. That role assignment needs source review; syntax acceptance is not evidence of correct substrate/catalyst linkage. No complete molecular set is certified here.

## What was compared

- **Original:** unmodified upstream commit `479e81232ded6bd11dbcbc934fec27a7b9e19e1b`, invoking `main.ChemEagle` once per selected page/model. Its planner chooses specialists; the actual molecule recognizer, specialist extraction, condition interpretation and synthesis run on HKUST A30 GPUs. No repair from our fork is applied to its exported results.
- **Focused:** the saved target-assisted workflow at fork `9872c6f07b2f6bdde3f335410ae9150341725dce`, with source page image/text, named targets and one reference page. The existing 18 calls are reused. Rechecking them with consistent typography/solvent normalization and graph counting leaves the reported scores unchanged.
- Models are CLI Proxy aliases `gpt-6-luna`, `gpt-6.1-terra`, `gpt-6.1-sol`. Their underlying snapshots are unknown. These are single runs, not repeated estimates of model variability.
- Both paths use identical 180dpi current-page PNGs. The original receives no externally supplied target list or reference-page packet, because that interface is absent upstream. It must discover rows. The focused path receives that assistance. Tools, decoding defaults and output scope also differ.
- This is **selected-page core extraction**, not the original `main_pdf.ChemEagle_PDF` whole-document wrapper. Its PDF segmentation, figure crops, document text and registry flow are not exercised. Six article PDFs contribute one page each; SI, scans, full-document recall and blind held-out performance remain untested. Low full-page recognizer quality here does not quantify its quality on native figure crops.

| PDF case | Physical page | Target experiments |
|---|---:|---|
| Epoxidation | 4 | Products 3ai, 3aj |
| Iodination | 4 | Products 3i, 3j |
| Furan | 5 | Table 2, entry 2 |
| Quaternary stereocenters | 3 | Table 1, entries 7 and 10 |
| Tricyclic tetrahydropyrans | 3 | Table 1, entries 8 and 9 |
| Vinylindole cycloaddition | 9 | Table 5, entries 14 and 15 |

## Scoring and source integrity

The same frozen rubric covers 11 target experiments, 22 reported outcome values and 33 condition fields. Only one uniquely identified final exported reaction can receive credit. Explicit product/entry labels establish identity. For tables, the original table specialist's documented `<row>_1` numbering can establish identity; a same-numbered scheme product cannot. Specialist artifacts establish provenance only and never supply values missing from the final export. Selection never uses the expected numeric answer.

Missing/ambiguous targets and failed exports receive no matched-field credit, with fixed denominators. An export can succeed while omitting every target. An original parser failure remains a failure even if a downstream parser could recover the raw JSON. There are no best-of-many retries. The quaternary table/prose conflict remains frozen at 89% for the table-specific entry 10 target, without silently substituting the prose value of 95%.

Normalization covers Unicode subscripts/minus signs, reported units, solvent synonyms and room-temperature wording. It does not derive ee from er or infer missing values. Graph counts cover only uniquely linked target reactants, products and catalysts; explicit nulls and solvents are excluded. Identical same-role representations within a row count once; repeats across experiments still count. Counts include invalid candidates before quarantine. **RDKit parseability does not certify chemical identity or stereochemistry.** A smaller graph count may simply mean fewer rows were delivered. No complete experiment has been fully qualified against all source molecular drawings in this comparison.

The upstream source archive is pinned by SHA-256. Each cell records before/after Python-source fingerprints; an additional check compares all 97 archived Python/text files, including prompts, against each job's source copy. All 18 actual input PNG hashes also match the manifest. Raw outputs remain immutable; projections and scores are separate artifacts. The scoring policy was recorded before final original outputs were reviewed. Later normalization corrections handle Unicode typography and distinguish absent provenance from ambiguous provenance; they do not change raw outputs or select rows by expected values.

## Runtime and cost method

Each original page has a 16-request limit and 1,800-second wall limit, with a 600-second streaming provider deadline. At most three independent GPU jobs run concurrently. Original extraction defaults are retained; network compound-name lookup and LLM confidence scoring are disabled. Model usage is collected at the HTTP boundary for all planning, specialist and synthesis calls, including runs whose final export fails. Provider inference is not automatically retried.

Authenticated SSH carries queued requests to the already configured local CLI Proxy. Credentials stay local. The adapter serializes SDK requests, collects upstream streaming responses and delivers the same completed messages to the original client. No original prompt or extraction code is edited. Buffered delivery means timings are not native streaming latency. The prior focused calls used a 180-second nonstreaming deadline, so these timings are not a controlled architecture comparison.

Input/output tokens are API-reported; cached input is a subset of input. Summed request time, pipeline wall time and allocated GPU time are different quantities. Dollar cost is unknown for both API and HPC. No public tariff is substituted for the user's unknown proxy pricing.

## Reproduction and artifacts

Local artifacts: `/Users/mathus/Documents/ChemEagle-runs/2026-11-18-yufan-model-comparison`.
HPC artifacts: `/scratch/mjab/chemeagle/yufan-models-20261118`.

- `manifest.json`, `source-pin.json`, `upstream-source.tar.gz`: source/version, PDF/page hashes, case selection and limits.
- `run_hpc.py`, `run.sbatch`, `relay_local.py`, `relay_remote.py`: isolated original runs and credential-free HPC transport.
- `results/<case>--<model>/`: final export when available, raw stage artifacts, status and request IDs.
- `transport/`: complete raw provider responses, usage, request hashes and elapsed times, including failed-export runs.
- `ground-truth.json`, `SCORING_POLICY.md`, `score.py`, `scores.json`: fixed rubric and deterministic final-export scoring.
- `verify_focused.py`, `focused-rescore.json`, `verify_source.py`, `source-verification.json`: consistent focused rescoring and upstream-source verification.
- `hpc-accounting.json`, `comparison-summary.json`: allocated compute and consolidated totals.

No new production extraction logic is introduced by this report. The measured limitations remain separate from the earlier fork fixes and their offline replay evidence.

## Original page-level results

| PDF | Model | Final export | Targets | Outcomes | Conditions |
|---|---|---|---:|---:|---:|
| Epoxidation | gpt-6-luna | completed | 0/2 | 0/4 | 0/6 |
| Epoxidation | gpt-6.1-terra | completed | 0/2 | 0/4 | 0/6 |
| Epoxidation | gpt-6.1-sol | completed | 0/2 | 0/4 | 0/6 |
| Iodination | gpt-6-luna | completed | 0/2 | 0/4 | 0/6 |
| Iodination | gpt-6.1-terra | failed | 0/2 | 0/4 | 0/6 |
| Iodination | gpt-6.1-sol | failed | 0/2 | 0/4 | 0/6 |
| Furan | gpt-6-luna | completed | 0/1 | 0/2 | 0/3 |
| Furan | gpt-6.1-terra | completed | 0/1 | 0/2 | 0/3 |
| Furan | gpt-6.1-sol | failed | 0/1 | 0/2 | 0/3 |
| Quaternary | gpt-6-luna | completed | 2/2 | 4/4 | 6/6 |
| Quaternary | gpt-6.1-terra | completed | 2/2 | 4/4 | 6/6 |
| Quaternary | gpt-6.1-sol | completed | 2/2 | 4/4 | 6/6 |
| Tetrahydropyran | gpt-6-luna | completed | 2/2 | 4/4 | 6/6 |
| Tetrahydropyran | gpt-6.1-terra | completed | 2/2 | 4/4 | 6/6 |
| Tetrahydropyran | gpt-6.1-sol | failed | 0/2 | 0/4 | 0/6 |
| Cycloaddition | gpt-6-luna | failed | 0/2 | 0/4 | 0/6 |
| Cycloaddition | gpt-6.1-terra | completed | 2/2 | 4/4 | 6/6 |
| Cycloaddition | gpt-6.1-sol | completed | 2/2 | 4/4 | 6/6 |

## Failed exports

| PDF | Model | Exception | Observed cause |
|---|---|---|---|
| Iodination | gpt-6.1-terra | JSONDecodeError | Prose before fenced JSON rejected by original final parser |
| Iodination | gpt-6.1-sol | JSONDecodeError | Prose before fenced JSON rejected by original final parser |
| Furan | gpt-6.1-sol | JSONDecodeError | Prose before fenced JSON rejected by original final parser |
| Tetrahydropyran | gpt-6.1-sol | JSONDecodeError | Prose before fenced JSON rejected by original final parser |
| Cycloaddition | gpt-6-luna | RuntimeError | Malformed table-specialist JSON; all planned specialists failed |

## New usage for this original-Yufan expansion

123 new provider calls across 18 selected-page runs: **1,158,194 input + 136,934 output = 1,295,128 reported tokens**. Cached input: 86,400, already included in input. Calls without usage: 0. The focused half is entirely reused and adds zero new tokens here. This total is separate from the prior expansion's 419,084 reported tokens and its one unknown-usage call.

| Model | Pipeline wall seconds, summed | Input + output tokens |
|---|---:|---:|
| gpt-6-luna | 1,372.5 | 395,004 |
| gpt-6.1-terra | 1,545.2 | 409,776 |
| gpt-6.1-sol | 2,075.9 | 490,348 |

The 18 jobs consumed 5,068 allocated A30 seconds (84.5 GPU-minutes), including initialization and waits. Slurm process completion does not imply successful extraction; export status comes from each pipeline receipt. **Dollar cost remains unknown.**

All 18 cells match the 97 pinned Python/text source files. Raw final responses, specialist artifacts and usage receipts are retained. All task jobs are terminal and the session-owned relay is stopped; the shared proxy, VPN and SSH login remain available.
