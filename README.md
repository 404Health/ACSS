🛡️ Autonomous Clinical Scribe System (ACSS)

🚀 Overview
The Autonomous Clinical Scribe System (ACSS) is an offline, multi-agent AI workflow that transforms messy, bilingual (Arabic/English) clinical audio into structured SOAP notes and validated hospital-ready data. Engineered specifically for on-premise hospital servers, this system guarantees absolute patient data privacy through a strict "Zero Data Egress" policy.

By aligning with Vision 2030 digital health goals and NPHIES interoperability, the ACSS is designed as a Human-in-the-Loop (HiTL) assistant—not an autonomous decision-maker—shifting clinician workflows from manual authoring to structured validation.

✨ Key Features

Zero Data Egress: A completely offline-first architecture that eliminates the use of cloud APIs, ensuring absolute compliance with national digital health data governance and data localization requirements.


Bilingual Arabic/English Processing: Natively handles linguistic complexity, accurately interpreting consultations that frequently mix English with Arabic medical slang.


Clinical Safety & HiTL Enforcement: Operates on a "Safety Over Automation" principle. No output is authorized for the Electronic Health Record (EHR) without explicit, mandatory physician approval via a multi-layered validation UI.


Medical Standard Mapping: Automatically cross-references and maps terms to rigorous Saudi and international standards, including ICD-10-AM, ACHI, ACS, and SFDA-GTIN.

🏗️ System Architecture (The Multi-Agent Pipeline)
The system relies on a local Qwen 9B LLM and is orchestrated via LangGraph.

Phase 1: Pre-Processing (Secure Audio Capture)

Leverages a local instance of WhisperX to transcribe bilingual clinical audio.

Utilizes speaker diarization to separate the physician's voice from the patient's voice.

Phase 2: Multi-Agent Core Pipeline


🕵️‍♂️ The Extractor Agent: Translates Arabic/English to English and structures raw clinical facts (Vitals, Medications, History) into standardized JSON.


✍️ The Scribe Agent: Converts the extracted JSON into a professional SOAP Note format, utilizing Clinical Decision Support (CDS) to flag anomalies (e.g., drug allergies).


🛡️ The Validator & Coder Agent: Cross-references the generated SOAP note against the exact transcript to prevent AI hallucinations, and rigorously assigns standard medical codes.

Phase 3: The Shield (Streamlit Dashboard)

Pauses the automated process to display a Human-in-the-Loop interface where physicians conduct Transcript, Semantic, and Clinical Validations.

Phase 4: Final Distribution

Pushes clinician-approved SOAP notes to the hospital EHR and automatically drafts a simplified, bilingual summary of care for the patient (e.g., for the Sehhaty app).

🛠️ Tech Stack

AI Orchestration: LangGraph / Python 


Large Language Model (LLM): Qwen 9B (running locally via Ollama) 


Audio Processing Engine: WhisperX & FFmpeg (PyTorch CUDA) 


Frontend Web Dashboard: Streamlit 

⚙️ Setup & Installation
To maintain low processing latency, dedicated on-premise GPU servers (e.g., NVIDIA A100s or equivalent consumer GPUs) are highly recommended.

1. Install Audio Dependencies
You must install ffmpeg for background audio processing to work. Download it and add it to your Windows System PATH (e.g., C:\ffmpeg\bin).

2. Ensure GPU PyTorch (CUDA)
By default, Windows PyTorch downloads standard CPU models. To prevent CPU-bottlenecking during WhisperX transcription, install the specific CUDA (NVIDIA) versions of PyTorch.

3. Optimize Local Model Memory
To prevent the 9B Qwen model from overflowing your GPU VRAM, you can constrain its context window (e.g., setting it to 8,192 tokens) using an Ollama Modelfile to ensure it runs entirely on your graphics card.

🚀 Running the Application
Launch the Human-in-the-Loop interface by activating your virtual environment and running the local web server:
