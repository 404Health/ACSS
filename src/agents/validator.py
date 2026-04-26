import json
from langchain_core.messages import SystemMessage, HumanMessage
from src.llm.local_model import get_llm
from src.agents.state import ClinicalState

# System prompt forcing strict auditing and ICD-10-AM coding
VALIDATOR_PROMPT = """
You are a strict Clinical Auditor and Medical Coder for a Saudi hospital.
You will be provided with the original Transcript and a drafted SOAP Note.

Your job is twofold:
1. HALLUCINATION CHECK: Does the SOAP note contain any medical facts NOT present in the transcript? 
2. CODING: Attempt to map any clear diagnoses or symptoms to standard ICD-10-AM codes.

You must output STRICTLY a JSON object in this exact format, with no conversational text:
{
    "validation_flags": ["list of strings detailing any hallucinations or warnings, or 'None'"],
    "icd_10_codes": ["list of strings with predicted codes, e.g., 'R51 - Headache'"]
}
"""

def real_validator_node(state: ClinicalState):
    print("-> 🛡️ Validator Agent: Waking up Qwen to audit the note and assign codes...")
    
    # 1. Connect to local model (Temperature 0.0 for strict factual auditing)
    llm = get_llm(temperature=0.0) 
    
    # 2. Package the context (Transcript + SOAP Note)
    context_payload = f"""
    --- ORIGINAL TRANSCRIPT ---
    {state['transcript']}
    
    --- DRAFTED SOAP NOTE ---
    {state['soap_note']}
    """
    
    messages = [
        SystemMessage(content=VALIDATOR_PROMPT),
        HumanMessage(content=context_payload)
    ]
    
    # 3. Ask the model to audit and code
    try:
        print("   (Auditing... Qwen is cross-referencing data)")
        response = llm.invoke(messages)
        
        # 4. Clean up and parse JSON
        raw_output = response.content.replace("```json", "").replace("```", "").strip()
        audit_data = json.loads(raw_output)
        
        # Combine flags and codes into the state's validation flags list
        combined_flags = audit_data.get("validation_flags", [])
        combined_flags.append(f"Suggested Codes: {audit_data.get('icd_10_codes', [])}")
        
        print("   ✅ Audit Complete!")
        # We also set final_approved to False, enforcing the Human-in-the-Loop rule
        return {"validation_flags": combined_flags, "final_approved": False}
        
    except json.JSONDecodeError:
        print("   ❌ Error: The Validator did not return valid JSON.")
        return {"validation_flags": ["System Error: Failed to parse audit data."], "final_approved": False}
    except Exception as e:
        print(f"   ❌ System Error: {e}")
        return {"validation_flags": [f"Error: {e}"], "final_approved": False}