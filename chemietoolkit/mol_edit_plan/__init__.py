"""Edit-plan variant of the molecular recognition agent's LLM stage.

The vision tool output is turned into a catalog with immutable ids; the LLM
returns a small edit plan (OCR corrections, text corrections, single-valued
definitions, expansion groups, decisions); ``postprocess.process`` applies it
deterministically. See postprocess.py and prompt/prompt_Mol_Plan.txt.
"""
from .postprocess import SCHEMA, PlanError, catalog, process  # noqa: F401
