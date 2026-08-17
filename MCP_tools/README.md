# MCP tools

`chemeagle_mcp_server.py` exposes ChemEAGLE's chemical skills as an MCP server
(`FastMCP("chemeagle-tools")`), so any MCP-capable agent can call the same tools
ChemEAGLE's own agents rely on. Clients usually register it under the key
`chemeagle`, which is where the tool ids `mcp__chemeagle__*` come from.

The 12 tools are **primitives**, not the composite agents. A composite such as
`process_reaction_image_with_product_variant_R_group` runs a whole
layout-specific pipeline internally; that pipeline *is* ChemEAGLE's
orchestration. What an MCP client gets here is the underlying skill set, which
it has to route itself.

## Tool correspondence

| MCP tool | Underlying component | Counterpart in the ChemEAGLE pipeline |
|---|---|---|
| `detect_molecules` | `ChemIEToolkit.extract_molecule_bboxes_from_figures` (MolDetector) | not called on its own; the pipeline goes straight to the coref variant below |
| `image_to_graph` | `ChemIEToolkit.molscribe.predict_image_file` (MolNexTR) | runs inside `RxnIM.predict_image_file(..., molnextr=True)` |
| `graph_to_smiles` | `molnextr.chemistry._convert_graph_to_smiles` | the same routine `update_input_with_symbols` calls after an agent has corrected the symbols |
| `parse_reaction_image` | `RxnIM.predict_image_file(..., ocr=True)` (RxnImgParser) | `get_full_reaction_template` / `get_reaction`, the reaction template parsing agent's data source |
| `extract_molecules_with_labels` | `ChemIEToolkit.extract_molecule_corefs_from_figures` | `get_multi_molecular_text_to_correct_withatoms`, the molecular recognition agent's tool |
| `ocr_image` | `pytesseract` (TesseractOCR) | `_tesseract_ocr_image`, registered as `TesseractOCR` in the condition interpretation agent |
| `reconstruct_smiles` | RDKit substructure transfer (SMILESReconstructor) | same goal as the R-group substitution in the two R-group agents, different mechanism: `replace_symbols_and_generate_smiles` edits the symbol list and rebuilds the SMILES through `_convert_graph_to_smiles` |
| `resolve_name_to_smiles` | OPSIN, then PubChem, then CIR | `_resolve_name_to_smiles`, which tries the curated alias map, then PubChem, then OPSIN, and has no CIR step |
| `validate_smiles` | RDKit parse plus canonicalisation | `_validate_and_fix_smiles`, used by the SMILES repair pass on the final output |
| `interpret_conditions` | `RxnIM.predict_image_file` condition slice (RxnConInterpreter) | `get_reaction_c`, the condition interpretation agent's tool |
| `extract_reactions_from_text` | `ChemRxnExtractor` | `extract_reactions_from_text_in_image`, a text extraction agent tool |
| `chemical_ner` | `ChemIEToolkit.chemner` (BioBERT-Large, MolNER) | `NER_from_text_in_image`, the other text extraction agent tool |

Three of the twelve have no direct counterpart in the pipeline's agent code:
`detect_molecules` is the detection half of a step the pipeline only ever calls
whole, while `resolve_name_to_smiles` and `validate_smiles` correspond to helper
routines that run after the agents finish rather than to any agent tool.

## Running the server

```bash
# stdio (default), for a local MCP client
python chemeagle_mcp_server.py --image-root <image_dir>

# streamable HTTP, for clients that attach over a port (default 8931)
python chemeagle_mcp_server.py --image-root <image_dir> \
    --transport streamable-http --host 127.0.0.1 --port 8931

# on GPU, with the models warmed at startup and a call trace written as JSONL
python chemeagle_mcp_server.py --image-root <image_dir> \
    --device cuda --warm --trace calls.jsonl

# smoke-test the tools on one image, then exit
python chemeagle_mcp_server.py --selftest <image_path>
```

`--selftest` exercises 11 of the 12 tools; `graph_to_smiles` is left out because
it takes a graph rather than an image. `--transport` accepts `stdio`, `sse`, or
`streamable-http`.

`--image-root` is a sandbox boundary, not a convenience: the server refuses any
path outside it and any path that is not an image. Weights load lazily on the
first call that needs them, and resolve exactly as they do for the main
pipeline, so `rxn.ckpt` belongs in the repository root alongside the other
checkpoints.

## Using it from a coding agent

Any MCP-capable agent can drive these tools, which is the point of shipping them
this way: the agent brings its own planning and calls the chemistry skills as it
goes.

Claude Code, over stdio:

```bash
claude mcp add chemeagle -- python MCP_tools/chemeagle_mcp_server.py \
    --image-root <image_dir>
```

Codex and other clients take the same command in their own MCP configuration:

```json
{
  "mcpServers": {
    "chemeagle": {
      "command": "python",
      "args": ["MCP_tools/chemeagle_mcp_server.py", "--image-root", "<image_dir>"]
    }
  }
}
```

Once registered, the tools appear under the `chemeagle` namespace (Claude Code
shows them as `mcp__chemeagle__detect_molecules` and so on). For a long-lived
server that several sessions attach to, start it with
`--transport streamable-http` and point the client at the port instead.

Two things worth setting deliberately: `--device cuda` if a GPU is available,
since every image tool runs a vision model, and `--image-root` narrowed to the
directory the agent is allowed to read, since a coding agent has its own file
access and the flag is what keeps tool calls inside the intended corpus.
