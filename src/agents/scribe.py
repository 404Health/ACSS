import json
from langchain_core.messages import SystemMessage, HumanMessage
from src.llm.local_model import get_llm
from src.agents.state import ClinicalState

# System prompt forcing a structured SOAP format and clinical safety checks
SCRIBE_PROMPT = """
You are an expert Clinical Scribe for a Saudi hospital.
Your job is to take structured JSON medical data and write a professional, highly concise SOAP note.

Format exactly like this:
SUBJECTIVE: [Patient complaints]
OBJECTIVE: [Vitals and measurable data]
ASSESSMENT: [Possible clinical impression based on data]
PLAN: [Recommended monitoring or next steps]

CRITICAL CLINICAL RULE (CDS): 
If you notice any abnormal vital signs (e.g., Blood Pressure higher than 120/80) or severe symptoms, you MUST add a [RED FLAG] tag immediately next to that specific item in the note.

Output ONLY the text of the SOAP note. Do not include any conversational filler.
"""

def real_scribe_node(state: ClinicalState):
    print("-> ✍️ Scribe Agent: Waking up Qwen to draft SOAP note...")
    
    # 1. Connect to local model (Temperature 0.1 for natural writing, but strict formatting)
    llm = get_llm(temperature=0.1) 
    
    # 2. Convert the Extractor's JSON dictionary back into a string for the LLM
    json_data = json.dumps(state["extracted_facts"], indent=2)
    
    # 3. Package the prompt and the JSON data
    messages = [
        SystemMessage(content=SCRIBE_PROMPT),
        HumanMessage(content=f"Structured Clinical Data:\n{json_data}")
    ]
    
    # 4. Ask the model to write the note
    try:
        print("   (Drafting... Qwen is writing the clinical note)")
        response = llm.invoke(messages)
        
        print("   ✅ SOAP Note Drafted!")
        return {"soap_note": response.content.strip()}
        
    except Exception as e:
        print(f"   ❌ System Error: {e}")
        return {"soap_note": "Error generating note."}