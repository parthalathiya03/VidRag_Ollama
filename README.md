# 🎙️ VidRag_Ollama: Voice RAG AI Agent

A **voice-enabled Retrieval-Augmented Generation (RAG) system** built using **Ollama**. Users can ask questions through speech, and the agent responds using local knowledge or falls back to Ollama.

---

## Features

- Voice-enabled AI agent using **STT → RAG → TTS**
- Real-time **speech-to-text (DeepgramSTT)** and **text-to-speech (ElevenLabsTTS)**
- Local knowledge retrieval from `.txt` documents using **FAISS**
- Fallback to **OpenAI GPT-4o** if documents are not relevant
- Real-time logging of queries, RAG retrieval, and responses
- Seamless interaction flow with agent entry, user message handling, and exit

---

## Getting Started

## 🎬 VidRag_Ollama — Live Demonstrations

Here are walkthrough videos demonstrating **VidRag_Ollama**, a Voice-Enabled RAG AI Agent built using **Ollama**:

### 🔹 Voice Query & Response Flow
[![Watch Demo 1](https://img.shields.io/badge/Watch-Voice%20Interaction%20Demo-red?logo=google-drive)](https://drive.google.com/file/d/1zOeQxBnv7vyzbQZZvfn74adXN-NYZB-4/view?usp=sharing)

### 🔹 Local RAG Retrieval in Action
[![Watch Demo 2](https://img.shields.io/badge/Watch-RAG%20Pipeline%20Demo-blue?logo=google-drive)](https://drive.google.com/file/d/1s7yLRnpxfnZ_g3drANB6oXC7IGobisFR/view?usp=sharing)


### 1. Clone the repository

```bash
git clone https://github.com/parthalathiya03/VidRag_Ollama.git
cd VidRag_Ollama
```


## Project Structure

```
VidRag_Ollama/
├── app.py                # Entry point for the voice AI agent
├── rag_pipeline.py       # Local RAG pipeline for document retrieval
├── docs/                 # Folder containing .txt documents for RAG
├── scripts/                
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
```


### 2. Create and activate a virtual environment

```bash
# macOS/Linux
python3.12 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up your Local Ollama model 

Ollama enables you to run open-source large language models (LLMs) directly on your machine. Follow these steps to get started:

### 1. Install Ollama

- **macOS**: Use Homebrew:
```bash
brew install ollama

After installation, start the Ollama server:
ollama serve
ollama pull <model-name>
ollama run <model-name>


### 5. Running the Voice Agent

```bash
streamlit app.py
```
Open the browser URL shown in the console to interact with the agent.

---

## How it Works

### 1. User Speech Input & Text as well (STT)

- Users speak into their microphone.
- speech_recognition converts speech to text.
- Text is sent to the RAG pipeline or fallback LLM.

### 2. RAG Query Processing

- Local `.txt` documents in `docs/` are split into chunks with **LangChain RecursiveCharacterTextSplitter**.
- Chunks are embedded using **HuggingFace Sentence Transformers**.
- FAISS retrieves the most relevant chunks.
- If no relevant docs, **Ollama** generates the response.

### 3. Response Generation

- A coherent text response is produced based on RAG or LLM output.
- Responses are optimized for **spoken-word delivery**.

### 4. Voice Output (TTS)

- Text is converted to speech via **ElevenLabsTTS**.
- Users hear the response in real-time.

### 5. Features
🎤 Multi-Modal Input

Voice Input: Speak naturally using Deepgram's speech-to-text API
Text Input: Type your questions as an alternative option
Adjustable Recording: Configure recording duration (3-10 seconds)
Real-time Transcription: Instant conversion of speech to text

🧠 Intelligent RAG (Retrieval-Augmented Generation)

Local Document Indexing: Automatically indexes all .txt files in your docs/ folder
Semantic Search: Uses sentence-transformers for contextual understanding
FAISS Vector Store: Lightning-fast similarity search through documents
Smart Context Retrieval: Finds the most relevant information from your knowledge base
Similarity Threshold: Configurable threshold to determine document relevance

🔄 Intelligent Fallback System

Context-Aware Responses: When relevant documents are found, answers are grounded in your knowledge base
Automatic Fallback: Seamlessly switches to Ollama's base knowledge when no relevant documents exist
Transparent Source Attribution: Always shows whether answer came from documents or base LLM
Best of Both Worlds: Combines local knowledge with general AI capabilities


---

## License

For educational and demonstration purposes. Respect licenses of **VideoSDK, OpenAI, Deepgram, ElevenLabs, and HuggingFace**.
