from langchain_ollama import ChatOllama

def get_llm(temperature=0.1):
    """
    Initializes and returns the local Qwen model via Ollama.
    This is the core engine for the Extractor, Scribe, and Validator agents.
    """
    print("Initializing connection to local Qwen model...")
    
    # ChatOllama connects seamlessly to the background Ollama service
    llm = ChatOllama(
        model="qwen2.5:3b", # Ensure this matches the tag you pulled in the terminal
        base_url="http://localhost:11434",
        temperature=temperature 
    )
    return llm

# Quick local test to verify the connection
if __name__ == "__main__":
    print("Testing LLM connection...")
    test_llm = get_llm()
    
    # We send a simple clinical prompt to test the model's response
    test_prompt = "You are an AI clinical scribe. Briefly explain the 4 sections of a SOAP note."
    
    try:
        response = test_llm.invoke(test_prompt)
        print("\n--- Model Response ---")
        print(response.content)
        print("\n----------------------")
        print("Success! The local LLM is connected and responding.")
    except Exception as e:
        print(f"Connection failed. Make sure Ollama is running. Error: {e}")