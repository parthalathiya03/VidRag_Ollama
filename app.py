import streamlit as st
import speech_recognition as sr
import pyttsx3
import logging
from rag_pipeline import LocalRAGPipeline


logging.basicConfig(
    filename="voice_rag.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


logging.info("App started")


st.title("🎙️ Local Voice RAG Assistant with Fallback")
st.caption("Ask questions from local docs or general knowledge (via Ollama LLaMA3)")

rag = LocalRAGPipeline()
logging.info("Initialized LocalRAGPipeline")

recognizer = sr.Recognizer()
engine = pyttsx3.init()
logging.info("Speech recognizer and TTS engine initialized")


if st.button("🎤 Ask Question by Voice"):
    with sr.Microphone() as source:
        st.write("Listening...")
        logging.info("Listening to user voice input")
        audio = recognizer.listen(source)
        try:
            query = recognizer.recognize_google(audio)
            st.write(f"🗣️ You said: {query}")
            logging.info(f"User said: {query}")

            answer = rag.query(query)
            st.success(answer)
            logging.info(f"RAG answer: {answer}")

            engine.say(answer)
            engine.runAndWait()
        except Exception as e:
            st.error(f"Speech recognition failed: {e}")
            logging.error(f"Speech recognition error: {e}")


text_query = st.text_input("Or type your question:")
if st.button("Ask"):
    if text_query:
        logging.info(f"User typed: {text_query}")
        answer = rag.query(text_query)
        st.success(answer)
        logging.info(f"RAG answer: {answer}")
        engine.say(answer)
        engine.runAndWait()

# ==================HYBRID RERANKING RAG APPROCH=================================================================================

# import streamlit as st
# import speech_recognition as sr
# import pyttsx3
# import logging
# from rag_pipeline import HybridRAGRerankPipeline

# # ------------------------------
# # Configure Logging
# # ------------------------------
# logging.basicConfig(
#     filename="voice_rag.log",
#     level=logging.INFO,
#     format="%(asctime)s - %(levelname)s - %(message)s"
# )
# logging.info("App started")

# # ------------------------------
# # Streamlit UI
# # ------------------------------
# st.title("🎙️ Local Voice Hybrid RAG Assistant (with Reranking)")
# st.caption("Ask questions from local docs (hybrid retrieval + reranking) or general knowledge via Ollama LLaMA3")

# # Initialize RAG pipeline
# rag = HybridRAGRerankPipeline()
# logging.info("Initialized HybridRAGRerankPipeline")

# recognizer = sr.Recognizer()
# engine = pyttsx3.init()
# logging.info("Speech recognizer and TTS engine initialized")

# # ------------------------------
# # Voice Query
# # ------------------------------
# if st.button("🎤 Ask Question by Voice"):
#     with sr.Microphone() as source:
#         st.write("Listening...")
#         logging.info("Listening to user voice input")
#         audio = recognizer.listen(source)
#         try:
#             query = recognizer.recognize_google(audio)
#             st.write(f"🗣️ You said: {query}")
#             logging.info(f"User said: {query}")

#             answer = rag.query(query)
#             st.success(answer)
#             logging.info(f"RAG answer: {answer}")

#             engine.say(answer)
#             engine.runAndWait()
#         except Exception as e:
#             st.error(f"Speech recognition failed: {e}")
#             logging.error(f"Speech recognition error: {e}")

# # ------------------------------
# # Text Query
# # ------------------------------
# text_query = st.text_input("Or type your question:")
# if st.button("Ask"):
#     if text_query:
#         logging.info(f"User typed: {text_query}")
#         answer = rag.query(text_query)
#         st.success(answer)
#         logging.info(f"RAG answer: {answer}")
#         engine.say(answer)
#         engine.runAndWait()
