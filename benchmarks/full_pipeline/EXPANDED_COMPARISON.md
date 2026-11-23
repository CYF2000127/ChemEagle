# Expanded PDF and model comparison

Run started 17 November 2026 (Hong Kong), continuing 18 November.
Earlier focused/fork comparison pinned at `9872c6f07b2f6bdde3f335410ae9150341725dce`.
CLI Proxy aliases; underlying model snapshots and dollar tariff unknown.

## Expanded model matrix: focused workflow and original Yufan pipeline

Expanded on 18 November 2026 using the same six real article PDFs and 11 specified experiments. Each model now has a focused column from the existing matrix and a new original-Yufan column. Luna = `gpt-6-luna`; Terra = `gpt-6.1-terra`; Sol = `gpt-6.1-sol` (CLI Proxy aliases, underlying snapshots unknown).

**Focused** is our target-assisted selected-row workflow: page image/text, target labels and one reference page. Its 18 completed calls are reused here without new inference. **Yufan** is the unmodified upstream `main.ChemEagle` hybrid at `479e81232ded6bd11dbcbc934fec27a7b9e19e1b`, run once per selected page/model on HPC, including the original planner, molecule recognizer, specialists and synthesis. It receives the identical current-page image, without supplied target labels or reference-page context. It is not our modified hybrid fork or the original whole-document `main_pdf` wrapper. These different tasks do not establish a controlled architectural accuracy/speed advantage. No SI, scanned PDFs, whole-document recall or blind held-out evaluation is included.

| Measure | Luna · focused | Luna · Yufan | Terra · focused | Terra · Yufan | Sol · focused | Sol · Yufan |
|---|---:|---:|---:|---:|---:|---:|
| Pages with final export | 6/6 | 5/6 | 6/6 | 5/6 | 6/6 | 3/6 |
| Failed page exports | 0 | 1 | 0 | 1 | 0 | 3 |
| Target entries returned | 11/11 | 4/11 | 11/11 | 6/11 | 11/11 | 4/11 |
| Outcome values matched | 22/22 | 8/22 | 22/22 | 12/22 | 22/22 | 8/22 |
| Condition fields matched | 30/33 | 12/33 | 33/33 | 18/33 | 33/33 | 12/33 |
| Invalid graph candidates | 0 | 2 | 2 | 2 | 0 | 0 |
| Parseable graph candidates | 13 | 8 | 36 | 16 | 40 | 14 |
| Model requests | 6 | 39 | 6 | 40 | 6 | 44 |
| Input tokens | 60,773 | 363,156 | 60,773 | 366,822 | 60,773 | 428,216 |
| Output tokens | 6,205 | 31,848 | 8,637 | 42,954 | 13,181 | 62,132 |
| Cached input tokens | 0 | 0 | 0 | 9,216 | 0 | 77,184 |
| Summed request seconds | 143.2 | 724.1 | 183.6 | 934.7 | 310.4 | 1,472.0 |
| Dollar cost | Unknown | Unknown | Unknown | Unknown | Unknown | Unknown |

Missing targets and failed exports receive no matched-field credit; denominators remain 11/22/33. Graph counts cover only uniquely linked target records, so fewer graphs can reflect fewer delivered rows. They are syntax counts, not molecular accuracy. Failed exports are counted even when the raw model response contains useful JSON that the original parser cannot read. Cached input is already part of input tokens. Original usage includes every specialist and final-synthesis call, including failed exports. No new focused calls were made for this expansion.

See [YUFAN_MODEL_COMPARISON.md](YUFAN_MODEL_COMPARISON.md) for page-level results, failures, source verification and new usage. The remaining focused discussion and two fork-hybrid runs below are the earlier comparison, not additional original-Yufan runs.

No experiment has a fully source-verified exact set of substrate, catalyst and product graphs. This is an unqualified outcome, not a measured 0% chemistry accuracy. Graph counts include repeated molecules across rows. RDKit syntax checks cannot certify chemical identity or stereochemistry. In the focused cohort, Terra's two invalid graphs were quarantined without rerunning a model. Its reused furan response needed the previously implemented control-character JSON parser. All 12 new focused calls in that earlier run parsed successfully. None of the focused raw responses added ee alongside reported er on these cases.

The focused Sol and Terra runs matched all checked footnote overrides. Luna retained three known epoxidation condition errors. Sol provided more parseable graph candidates, took longer, and used more output tokens. No universal model winner is established by this selected development sample.

The earlier expansion added these pages: quaternary stereocenters, Table 1 entries 7/10 (10.1038/s41467-021-27028-7); tricyclic tetrahydropyrans, Table 1 entries 8/9 (10.1038/s41467-021-27521-z); vinylindole cycloaddition, Table 5 entries 14/15 (10.3390/molecules26216751). Existing pages: epoxidation 3ai/3aj, iodination 3i/3j, furan Table 2 entry 2. Ground truth was frozen before the new calls. The quaternary paper reports 89% yield in its table and 95% in prose for entry 10; the table-specific target uses 89 and retains the conflict. The cycloaddition PDF has duplicated embedded text, while the rendered page is clean.

The focused half of the matrix totals 210,342 tokens including 65,477 reused historical tokens. That earlier focused expansion used 144,865 new tokens: Luna 34,480; Terra 36,431; Sol 73,954. See cost-accounting.json for exact new/reused accounting. Tokens are API-reported; no dollar estimate is fabricated.

## Earlier modified-fork hybrid comparison

Two real hybrid runs completed on an NVIDIA A30 at HKUST, using `main.ChemEagle` from the pinned fork, automatic inventory, actual molecule recognition, specialist tools, synthesis and bounded recovery. Both used `gpt-6.1-sol`, the same prepared page/reference packets as the focused tests, network name lookups off, at most 16 model requests and 900 seconds per page. No target inventory or expected values were supplied.

| Measure | Iodination page 4 | Tetrahydropyran page 3 |
|---|---:|---:|
| Pipeline elapsed time | 494.2 s | 584.9 s |
| Logical model requests | 12 | 13 |
| Input tokens reported | 100,671 | 130,565 |
| Output tokens reported | 10,320 | 21,188 |
| Cached input tokens (subset of input) | 8,960 | 8,960 |
| Calls without token usage | 1 | 0 |
| Exported ledger entries | 27 | 41 |
| Ledger entries with assembled records | 5 | 4 |
| Selected target labels retained | 2/2 | 2/2 |
| Selected targets with an assembled record | 0/2 | 0/2 |

**Iodination:** final synthesis exceeded the benchmark relay's 180-second upstream request deadline. The pipeline preserved entries and ran its four-call recovery budget, but 3i/3j remained empty placeholders. This is a bounded-test timeout, not evidence that normal native streaming would fail. The timed-out call has unknown token usage.

**Tetrahydropyran:** final synthesis succeeded and returned 14 reaction records, including correct outcomes and overrides for entries 8/9. Assembly nevertheless lost those records: the image inventory split each yield and ee into separate experiments, and synthesis then added another row identity. The exported 41 entries were not 41 real experiments. With the normalization and identity fix, deterministic replay of the same saved stages produces 14 inventoried entries and 14 linked records, with zero unmatched records and no new inference. For entries 8/9, all four checked outcome values and all six checked condition fields match the source. Other rows' numerical fields have not all been manually scored.

The replay preserves the original nested outcome identifiers as provenance; its new top-level IDs represent normalized rows. It is an assembly diagnostic, not a publishable repaired chemistry dataset. Applying the same graph audit quarantines all 14 repeated invalid product candidates (one invalid graph repeated across rows). Molecular identity and stereochemistry remain unresolved.

The focused Sol path returned all four selected targets on these two pages, but it receives target labels and reference context directly, while the hybrid must discover every experiment. Prompts, tools, decoding defaults and output scope differ. These numbers do not establish a fair whole-document speed/accuracy advantage or make the focused benchmark a replacement pipeline.

Transport and environment: Python 3.10, Torch 2.2.0+cu121, OpenAI SDK 3.46.0, RDKit 2022.09.5, existing pinned model assets, and isolated `pypdf==6.20.0`. Requests were relayed through authenticated SSH to local CLI Proxy; provider credentials stayed local. The temporary relay uses buffered upstream replies, so elapsed time is not a native-streaming latency benchmark. Setup initially encountered missing package/assets, then streaming/serialization adapter errors; those attempts are excluded from extraction-quality scores but their inference and allocated compute are included in costs. Exact duplicate requests reused saved responses by hash. No configuration or services were installed globally.

## Earlier expansion usage (excludes the new original-Yufan runs)

- New focused comparison: 144,865 reported tokens across 12 calls.
- New hybrid inference including setup attempts: 274,219 reported tokens across 26 calls; one call has no usage receipt.
- Total new reported usage: **364,870 input + 54,214 output = 419,084 tokens**, plus unknown usage for the timed-out call. Cached input: 17,920, already included in input tokens.
- Six old focused responses (65,477 tokens) were reused and are excluded from that new total. Local replay and regression tests made no provider calls.
- Total allocated A30 time across final and setup/cancelled jobs: 1,434 seconds (23.9 GPU-minutes). There are no remaining jobs or relay processes from this comparison.
- **Dollar cost unknown**, including HPC billing. Tokens and GPU time are not converted into prices without a tariff.


## Fixes from real hybrid failures

The live baseline remains pinned to `9872c6f`; the following local changes are verified by offline replay of its saved responses, not by a second full hybrid model run:

- **Text JSON:** the original specialist parser rejected a prose prefix before a valid complete JSON object. The new parser retains the complete raw response and reads the single object, exposing one structured reaction object. It rejects incomplete JSON, multiple objects, or trailing prose and does not repair chemical values.
- **Pending status:** the old quality audit overwrote `awaiting_review` even for empty row placeholders. It now preserves that status and adds an explicit missing-record issue. Replay keeps 22 iodination rows pending instead of relabeling them as extracted.
- **Experimental-row inventory:** the old prompt allowed one row's yield and ee to become separate experiments. The corrected prompt makes them fields of one experiment. A conservative normalizer combines only distinct metric names sharing the same row label, explicit table reference and exactly identical located evidence. The real tetrahydropyran inventory normalizes 27 metric entries to 14 rows. Repeated metrics, different source regions and explicitly different experimental conditions remain separate. Raw inventories and merge labels are retained. Synthesis records without outcome metadata reuse one unambiguous existing row rather than creating another row identity. Explicitly different runs remain separate.

Eleven focused tests pass, covering these failures and existing evidence/ledger behavior. These changes improve parsing and accounting; they do not establish molecular identity or a universal PDF accuracy claim.

A narrow additional check found Sol's encoded stereocenter in quaternary product 3a matches the paper's S assignment; Luna and Terra leave it unspecified. This checks the encoded CIP label only, not complete drawing-to-graph identity. See stereo-spot-check.json.

## Reproduction and artifacts

`manifest.json`: PDF/image hashes, source text, reference pages, target labels, exact fork commit. `ground-truth.json`: source-reviewed field targets and conflict. `run.py`: same focused prompt for all models; at most two concurrent calls; 180-second request deadline; no provider retries. `score.py`: deterministic field normalization and RDKit syntax checks. Each case/model folder retains the raw response, audited result and graph-validation result. `reused.json` identifies historical cells. `hybrid_hpc.py` and `hybrid-retry.sbatch` pin the bounded real pipeline. The SSH file relay changes transport only and keeps the CLI Proxy key local.

Raw artifacts: `/Users/mathus/Documents/ChemEagle-runs/2026-11-17-expanded-comparison`.
