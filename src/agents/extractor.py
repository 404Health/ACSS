import json
from langchain_core.messages import SystemMessage, HumanMessage
from src.llm.local_model import get_llm
from src.agents.state import ClinicalState

# System prompt forcing strict JSON output and Arabizi translation
EXTRACTOR_PROMPT = """
You are a high-precision clinical data extractor for a Saudi hospital. 
Your job is to read raw clinical audio transcripts (which may contain a mix of English and Arabizi).
You must extract the clinical facts and output them strictly as a JSON object.
Do NOT output any conversational text. ONLY output the JSON.

Expected JSON Structure:
{
    "chief_complaint": "string",
    "vitals": {"blood_pressure": "string", "temperature": "string"},
    "medications": ["list of strings"],
    "allergies": ["list of strings"]
}
"""

def real_extractor_node(state: ClinicalState):
    print("-> 🕵️‍♂️ Extractor Agent: Waking up Qwen to parse transcript...")
    
    # 1. Connect to our local model
    llm = get_llm(temperature=0.0) # Absolute zero temperature for strict data extraction
    
    # 2. Package the prompt and the transcript
    messages = [
        SystemMessage(content=EXTRACTOR_PROMPT),
        HumanMessage(content=f"Transcript to process:\n{state['transcript']}")
    ]
    
    # 3. Ask the model to extract the data
    try:
        print("   (Thinking... this may take a few moments for the 9B model)")
        response = llm.invoke(messages)
        
        # 4. Clean up the response to ensure it's pure JSON
        raw_output = response.content.replace("```json", "").replace("```", "").strip()
        extracted_data = json.loads(raw_output)
        
        print("   ✅ Extraction Successful!")
        return {"extracted_facts": extracted_data}
        
    except json.JSONDecodeError:
        print("   ❌ Error: The model did not return valid JSON.")
        return {"extracted_facts": {"error": "Failed to parse JSON", "raw_output": response.content}}
    except Exception as e:
        print(f"   ❌ System Error: {e}")
        return {"extracted_facts": {}}