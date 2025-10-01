# 🎙️ VidRag_Ollama: Voice RAG AI Agent

A **voice-enabled Retrieval-Augmented Generation (RAG) system** built using **VideoSDK Agents SDK**. Users can ask questions through speech, and the agent responds using local knowledge or falls back to GPT-4o.

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

### 1. Clone the repository

```bash
git clone https://github.com/parthalathiya03/VidRag_Ollama.git
cd VidRag_Ollama
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
pip install "videosdk-agents[deepgram,openai,elevenlabs,silero,turn_detector]"
```

Or, if you have a `requirements.txt`:

```bash
pip install -r requirements.txt
```

### 4. Set up your API keys

Create a `.env` file in the project root:

```env
DEEPGRAM_API_KEY="Your Deepgram API Key"
OPENAI_API_KEY="Your OpenAI API Key"
ELEVENLABS_API_KEY="Your ElevenLabs API Key"
VIDEOSDK_AUTH_TOKEN="Your VideoSDK Auth token"
```

---

## Running the Voice Agent

```bash
python main.py
```

Optionally, run in console mode:

```bash
python main.py console
```

Open the browser URL shown in the console to interact with the agent.

---

## How it Works

### 1. User Speech Input (STT)

- Users speak into their microphone.
- **DeepgramSTT** converts speech to text.
- Text is sent to the RAG pipeline or fallback LLM.

### 2. RAG Query Processing

- Local `.txt` documents in `docs/` are split into chunks with **LangChain RecursiveCharacterTextSplitter**.
- Chunks are embedded using **HuggingFace Sentence Transformers**.
- FAISS retrieves the most relevant chunks.
- If no relevant docs, **GPT-4o** generates the response.

### 3. Response Generation

- A coherent text response is produced based on RAG or LLM output.
- Responses are optimized for **spoken-word delivery**.

### 4. Voice Output (TTS)

- Text is converted to speech via **ElevenLabsTTS**.
- Users hear the response in real-time.

### 5. Features

- End-to-end voice interaction: **STT → RAG → TTS**
- Real-time logging of queries and responses
- Local document knowledge retrieval
- Fallback to GPT-4o for unknown queries
- Smooth conversation flow with entry, message, and exit handling

---

## Project Structure

```
VidRag_Ollama/
├── main.py               # Entry point for the voice AI agent
├── rag_pipeline.py       # Local RAG pipeline for document retrieval
├── docs/                 # Folder containing .txt documents for RAG
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
```

---

## License

For educational and demonstration purposes. Respect licenses of **VideoSDK, OpenAI, Deepgram, ElevenLabs, and HuggingFace**.