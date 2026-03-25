import re
from typing import List, Optional


AGENT_NAME_TO_TOOL = {
    "structure-based r-group substitution agent": "process_reaction_image_with_product_variant_R_group",
    "text-based r-group substitution agent": "process_reaction_image_with_table_R_group",
    "reaction template parsing agent": "get_full_reaction_template",
    "molecular recognition agent": "get_multi_molecular_full",
    "text extraction agent": "text_extraction_agent",
}


def _clean_agent_name(raw_name: str) -> str:
    """Strip leading numbering (e.g. '1.', '2)', '- ') from an agent name."""
    cleaned = re.sub(r'^[\d]+[.):\-\s]+', '', raw_name.strip())
    cleaned = re.sub(r'^[-•*]\s*', '', cleaned)
    return cleaned.strip()


def _parse_planner_output(raw_output: str) -> List[str]:
    """Parse planner text output into a clean list of agent names."""
    cleaned = re.sub(r'[{}]', '', raw_output).strip()
    agents = [_clean_agent_name(a) for a in cleaned.split(',') if a.strip()]
    return [a for a in agents if a]


def _select_main_area(agent_names_lower: List[str]) -> str:
    """Select the main area tool name from a list of agent names (substring match).
    Priority: structure R-group > text R-group > reaction template > molecular recognition."""
    priority = [
        ("structure-based r-group substitution agent", "process_reaction_image_with_product_variant_R_group"),
        ("text-based r-group substitution agent", "process_reaction_image_with_table_R_group"),
        ("reaction template parsing agent", "get_full_reaction_template"),
        ("molecular recognition agent", "get_multi_molecular_full"),
    ]
    for keyword, tool_name in priority:
        if any(keyword in agent for agent in agent_names_lower):
            return tool_name
    return "get_full_reaction_template"


def _has_text_extraction(agent_names_lower: List[str]) -> bool:
    """Check if text extraction agent is in the agent list (substring match)."""
    return any("text extraction agent" in a or "text_extraction_agent" in a
               for a in agent_names_lower)
