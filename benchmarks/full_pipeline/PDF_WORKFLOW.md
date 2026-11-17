# PDF evidence workflow

The hybrid fork accepts native PDF input through `pdf_evidence.py`. Poppler is
required (`pdfinfo`, `pdftotext`, `pdftoppm`). Heavy chemistry models remain in the
existing hybrid environment; preparation and quality checking do not load them.

```sh
python pdf_evidence.py article.pdf --output run --pages 4 --references 3 --prepare-only
python pdf_evidence.py article.pdf --output run --pages 4 --references 3 --model MODEL_ALIAS --recovery-budget 4
```

Supply `--si supplement.pdf` for supporting text. Up to two explicit reference
pages are supported. Pages are physical 1-based PDF indices. Multiple PDFs should
use separate output directories. Prepared pages are keyed by PDF hash, parser
version, page selection and rendering settings. Image hashes are recorded.
SI is supplied explicitly, hashed and retained in full in the manifest; the model
receives bounded text with truncation disclosed. SI image-page retrieval and
automatic reference selection remain future work.

For each selected page the adapter calls the actual `main.ChemEagle` hybrid
backend, passing current text, reference images and SI through `source_context`.
Failures are checkpointed per page; successful pages survive. Pages with little
embedded text are flagged as needing OCR while their images remain available to
the vision path. This does not benchmark scanned-PDF OCR quality.

Quality checks retain original model outputs. Unreported er-to-ee conversions
are withheld pending evidence, graph strings with unresolved stereochemistry are
kept as partial, and RDKit-invalid graphs are quarantined when RDKit is available.
Valid graph syntax never certifies identity or stereochemistry. Molecules with
unknown roles remain in `unassigned_structures` rather than becoming experiments.
Literal control characters copied from PDF text can be parsed without changing
values; the parser does not repair missing JSON brackets or chemical content.

`extraction_quality.usage_report` separates input, cached input and output tokens
and latency. Dollar costs remain null without supplied rates. User-supplied rates
may produce an estimate, never an actual billing claim. Image tokens are not
separately inferred from a provider that reports only aggregate input tokens.

## Small model comparison, 17 November 2026

Three development PDFs, five selected entries, two proxy models, six requests.
The matrix exercised PDF preparation, focused extraction and quality checks;
it did not rerun the entire hybrid GPU pipeline or benchmark whole-document
inventory. No SI was supplied. Inputs were identical across models and contained
no expected numerical answers or saved molecular predictions.

| Measure | gpt-6-luna | gpt-6.1-terra |
| --- | ---: | ---: |
| Selected entries returned | 5/5 | 5/5 |
| Reported outcome values matched | 10/10 | 10/10 |
| Condition fields matched | 12/15 | 15/15 |
| Invalid graph candidates | 0 | 2 |
| Verified complete experiments | 0 | 0 |
| Input tokens | 29,190 | 29,190 |
| Output tokens | 3,308 | 3,789 |
| Cached input tokens | 0 | 0 |
| Summed request time | 74.8 s | 82.1 s |
| Dollar cost | Unknown | Unknown |

Terra's furan response initially failed strict JSON parsing because it copied
literal control characters from PDF text. The deterministic parser fix recovered
that saved response without another model request. Both malformed SMILES were
then quarantined. Raw and postprocessed results are retained separately.

Terra handled all tested footnote overrides, while Luna missed the epoxidation
3ai temperature/loading overrides and 3aj solvent override. Neither regenerated
the earlier unreported ee conversions. Terra supplied more partial graphs, but
two were invalid and remaining graph identities/stereochemistry were not fully
reviewed. This tiny selected sample supports trying Terra for difficult condition
review, not selecting it as a validated universal chemistry extractor.

Raw artifacts and the full report:
`/Users/mathus/Documents/ChemEagle-runs/2026-11-17-pdf-model-comparison/`.
The current repaired CPA dataset hash was verified; older reference graphs were
not treated as current truth. Product graphs were not scored against that dataset.

## Remaining qualification

Exact catalyst/product stereochemistry, incomplete compound reconstruction,
scanned and multi-column PDF inventory recall, cross-page/SI association, automatic
reference retrieval, and full hybrid runtime integration need further measured
work. No 95% recall, 98% accepted precision or cost-reduction target is established.
The attached plan's untouched-paper benchmark and independent chemistry review
remain necessary before claiming a broadly reliable final version.
