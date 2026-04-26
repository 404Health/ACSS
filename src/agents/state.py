from typing import TypedDict, List, Dict, Any

class ClinicalState(TypedDict):
    """
    This represents the global state of the ACSS pipeline.
    As data moves from agent to agent, this dictionary gets updated.
    """
    raw_audio_path: str
    transcript: str
    extracted_facts: Dict[str, Any]  # The structured JSON data
    soap_note: str
    validation_flags: List[str]      # [RED FLAG] warnings for the UI
    final_approved: bool             # Human-in-the-Loop authorization