import whisperx
import torch
import os

def process_clinical_audio(audio_file_path: str) -> str:
    """
    Takes a raw audio file (wav/mp3), transcribes it using WhisperX, 
    and returns the raw text transcript.
    """
    print(f"🎙️ Waking up WhisperX to process: {audio_file_path}")
    
    # 1. Hardware Check: Use GPU if available, otherwise CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    
    print(f"   (Using device: {device.upper()})")

    try:
        # 2. Load the base Whisper model (small and fast for testing)
        # In a real hospital, you would use the "large-v3" model for max accuracy
        model = whisperx.load_model("base", device, compute_type=compute_type)
        
        # 3. Load and process the audio file
        print("   (Listening and transcribing...)")
        audio = whisperx.load_audio(audio_file_path)
        result = model.transcribe(audio, batch_size=8)
        
        # 4. Combine all the timestamped chunks into one paragraph
        full_transcript = " ".join([segment["text"] for segment in result["segments"]])
        
        print("   ✅ Transcription Complete!")
        return full_transcript.strip()
        
    except Exception as e:
        print(f"   ❌ Audio Processing Error: {e}")
        return "Error processing audio file."

# --- Local Test ---
if __name__ == "__main__":
    # Create a dummy test file path
    test_audio = "data/raw_audio/test_dictation.wav"
    
    if os.path.exists(test_audio):
        transcript = process_clinical_audio(test_audio)
        print("\n--- Whisper Output ---")
        print(transcript)
    else:
        print(f"⚠️ Test bypassed: Please place a dummy audio file at {test_audio} to test.")