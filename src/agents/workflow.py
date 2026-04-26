from langgraph.graph import StateGraph, END
from src.agents.state import ClinicalState
from src.agents.extractor import real_extractor_node
from src.agents.scribe import real_scribe_node
from src.agents.validator import real_validator_node

# --- 1. Define the Nodes (The Agents) ---
# Right now, these are just placeholder functions. 
# Later, we will inject the local Qwen LLM into these!

def extractor_node(state: ClinicalState):
    print("-> 🕵️‍♂️ Extractor Agent: Parsing Arabizi/English transcript into JSON...")
    return {"extracted_facts": {"status": "success", "vital_signs": "extracted"}}

def scribe_node(state: ClinicalState):
    print("-> ✍️ Scribe Agent: Drafting professional SOAP note...")
    return {"soap_note": "Subjective: Patient reports headache...\nObjective: Vitals stable..."}

def validator_node(state: ClinicalState):
    print("-> 🛡️ Validator Agent: Checking for hallucinations & mapping ICD-10 codes...")
    return {"validation_flags": ["[FLAG: Verify dosage severity]"]}

# --- 2. Build the State Machine ---
def build_workflow():
    print("Building LangGraph Architecture...")
    workflow = StateGraph(ClinicalState)

    # Add the agent nodes to the graph
    workflow.add_node("extractor", real_extractor_node)
    workflow.add_node("scribe", real_scribe_node)
    workflow.add_node("validator", real_validator_node)

    # Define the strict routing path
    workflow.set_entry_point("extractor")
    workflow.add_edge("extractor", "scribe")
    workflow.add_edge("scribe", "validator")
    workflow.add_edge("validator", END)

    return workflow.compile()

# --- 3. Local Test Execution ---
if __name__ == "__main__":
    app = build_workflow()
    
    # We simulate a messy transcript entering the system
    initial_state = {
        "transcript": "Al mareed came in complaining of a severe headache. BP is 120/80.",
        "extracted_facts": {},
        "soap_note": "",
        "validation_flags": [],
        "final_approved": False
    }
    
    print("\n🚀 Starting Multi-Agent Pipeline...")
    final_state = app.invoke(initial_state)
    
    print("\n--- 🏁 Pipeline Finished ---")
    print("Final SOAP Note Generated:\n", final_state["soap_note"])
    print("Validation Flags Raised:\n", final_state["validation_flags"])