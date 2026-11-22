# Full hybrid pipeline: experimental entry preservation

This change belongs to our ChemEAGLE fork. It extends `main.ChemEagle`, the hybrid
specialist-agent route inherited from Yufan. The earlier cloud Worker changes are
separate. The deprecated `ChemEagle_azure` route does not yet use the ledger.

The selected-page upstream comparison and later fork replay results are in
[EXPANDED_COMPARISON.md](EXPANDED_COMPARISON.md). Those historical runs predate
this upstream PR branch. Porting to upstream's confidence wrapper additionally
requires forwarding preservation options to `_run_chemeagle`; the PR includes
that wiring and an offline wrapper check. It has not had another full GPU run.

## Implemented first priority

- Independent image inventory before specialists, optionally reconciled with a
  supplied parser inventory (`entry_inventory`). Each entry receives a stable ID
  from paper hash, region, label and outcome. Pass article SHA-256, region ID and
  supplied SI definitions through `source_context` for paper-level provenance.
- Specialist outputs are saved before final synthesis, including raw graph arrays.
  Final-synthesis tool messages omit atom coordinates, bond matrices and molfiles.
- Deterministic reconciliation accounts for each inventoried item. Missing entries
  survive as pending records. Recognized labelled molecules remain candidates,
  without being falsely promoted to measured products. Ambiguous matches remain
  in `unassigned_candidates` for review. Duplicate/conflicting candidates remain
  in the ledger, rather than overwriting the first prediction.
- Recovery permits two attempts per item and four total per image. The second
  attempt retrieves recognizer, specialist and supplied definition evidence; when
  no changed evidence is available, its model call is skipped. Original records,
  proposals and dispositions are retained. Recovery never declares chemical
  correctness. All recovered rows remain `extracted_unresolved`.
- Each invocation has a distinct checkpoint directory under its source/config
  digest, so reruns cannot overwrite the original prediction. Completed receipts
  contain the final JSON hash. `manage.py reconcile` verifies hashes and retains
  distinct configurations as separate runs.
- Inventory/recovery model usage is recorded. Specialist, synthesis, GPU and human
  review costs are explicitly **not yet covered**; no total-cost saving is claimed.

Example with the already configured OpenAI-compatible client:

```python
result = ChemEagle(
    image_path,
    entry_inventory=parser_entries,
    source_context={"paper_sha256": pdf_sha256, "region_id": "page5-table1",
                    "definitions": supplied_si_definitions},
    checkpoint_dir="outputs/entry-checkpoints",
    recovery_budget=4,
)
```

The configured endpoint/key continues through `llm_client`; no new credential
store is used. Inventory adds one model request per image. Set `recovery_budget=0`
for preservation-only evaluation. Set `preserve_entries=False` for the existing
fork behavior. The budget counts semantic recovery calls, not transport-level SDK
retries. Existing action-observer retries remain independently configurable.

## Frozen comparisons and current evidence

Use `python benchmarks/full_pipeline/manage.py freeze families.json manifest.json`
to pin upstream `479e812`, current fork, and every article/SI hash. A family entry
contains `family_id`, `split` (`development` or `test`) and `sources` objects with
`path` and `role` (`article` or `SI`). Duplicate source hashes across families are
rejected. Keep all already inspected sources in development. This utility prepares
provenance; it does not reproduce the upstream environment or run a benchmark.

Use `python benchmarks/full_pipeline/manage.py reconcile outputs/entry-checkpoints export.json`
to rebuild an export from verified new checkpoint receipts. Legacy checkpoints
need a separate migration/audit; they must not be labelled verified merely because
their filenames exist.

Three small offline regression tests cover the named disappearance patterns
(`3ai/3aj`, `3i/3j`), unresolved Table 1 catalyst, ambiguous outcomes, recovery
bounds and raw-artifact preservation. These are regression fixtures, **not** a
reproduction of those papers with the GPU pipeline. No new API or HPC jobs were started for the initial ledger change; the later
selected-page runs are documented in the linked comparison.

## Remaining plan work — not claimed complete

1. Reproduce the three failures using original paper crops and the pinned upstream
   environment. Inventory independence, source interpretation and graph correctness
   still need image-level chemistry review. The ledger guarantees accounting only
   for discovered entries, not that the inventory found every experiment.
2. Migrate the legacy completed vision runs into a reconciled export with explicit
   paper-family scope. A local audit is saved outside the repository; it found
   more scope mismatches than the plan's earlier count of three, so that count
   must not be reused without identifying the same cohort/export.
3. Compare the existing parser and candidate parsers on a few identical development
   pages before installing/replacing models. No candidate was selected or benchmarked.
4. Add component-specific crop rendering/alternate recognizers, graph and
   phosphorus-group checks, full compound caching, and per-stage token/GPU accounting.
   Current recovery handles missing entries; it does not yet resolve all incomplete
   structures or contradictory conditions in already-present rows.
5. Annotate development families and reserve untouched article/SI families. Report
   row precision/recall, complete-experiment accuracy, role/stereo errors, unresolved
   fraction, latency and cost with denominators and uncertainty. No 95% recall,
   98% accepted precision or 30–50% token reduction has been established.

Human chemistry review is required before an entry may become `complete_checked`.
The current implementation deliberately reports zero automatically checked records.
