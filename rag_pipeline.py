import os
import subprocess
import glob
import logging
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# ------------------------------
# Configure Logging
# ------------------------------
logging.basicConfig(
    filename="rag_pipeline.log",  # Log file
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logging.info("RAG pipeline module loaded")


class LocalRAGPipeline:
    def __init__(self, docs_path="docs"):
        self.docs_path = docs_path
        self.vectorstore = None
        self.embedder = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        logging.info(
            f"Initialized HuggingFaceEmbeddings with model: {self.embedder.model_name}")
        self._build_vectorstore()

    def _build_vectorstore(self):
        logging.info(f"Building vectorstore from docs in: {self.docs_path}")
        doc_files = glob.glob(os.path.join(self.docs_path, "*.txt"))
        docs = []

        for file in doc_files:
            try:
                with open(file, "r", encoding="utf-8") as f:
                    content = f.read()
                    docs.append(Document(page_content=content,
                                metadata={"source": file}))
                    logging.info(f"Loaded document: {file}")
            except Exception as e:
                logging.error(f"Failed to read document {file}: {e}")

        if not docs:
            logging.warning(
                "No documents found in ./docs, RAG will always fallback.")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500, chunk_overlap=100)
        split_docs = splitter.split_documents(docs)
        logging.info(
            f"Split {len(docs)} documents into {len(split_docs)} chunks")

        if split_docs:
            self.vectorstore = FAISS.from_documents(split_docs, self.embedder)
            logging.info("RAG vectorstore built successfully")
        else:
            self.vectorstore = None
            logging.warning("Vectorstore not created due to empty split_docs")

    def retrieve_context(self, query, k=3):
        logging.info(f"Retrieving context for query: {query}")
        if not self.vectorstore:
            logging.warning("No vectorstore found, returning empty context")
            return []

        results = self.vectorstore.similarity_search(query, k=k)
        logging.info(f"Retrieved {len(results)} documents from vectorstore")
        return results

    def run_ollama(self, prompt):
        """Runs local LLM via Ollama CLI"""
        logging.info("Running Ollama LLM...")
        try:
            result = subprocess.run(
                ["ollama", "run", "llama3"],
                input=prompt.encode(),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True
            )
            output = result.stdout.decode().strip()
            logging.info("Ollama LLM returned output successfully")
            return output
        except subprocess.CalledProcessError as e:
            logging.error(f"Ollama CLI error: {e.stderr.decode().strip()}")
            return "⚠️ Error running Ollama LLM"

    def query(self, question):
        """Main pipeline: retrieve → augment → fallback"""
        logging.info(f"Processing query: {question}")
        docs = self.retrieve_context(question)

        if docs:
            context = "\n\n".join([d.page_content for d in docs])
            prompt = f"""You are a helpful AI assistant. Use the following context to answer the question.

Context:
{context}

Question: {question}

If the context is irrelevant, answer from your general knowledge.
Answer:"""
            return self.run_ollama(prompt)
        else:
            # Fallback if no docs found
            logging.warning(
                "No relevant docs found. Falling back to general LLM.")
            fallback_prompt = f"Answer this question using your general knowledge:\n{question}"
            return self.run_ollama(fallback_prompt)


# ==================HYBRID RERANKING RAG APPROCH=================================================================================

# import os
# import subprocess
# import glob
# import logging
# from langchain_community.vectorstores import FAISS
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.schema import Document
# from rank_bm25 import BM25Okapi
# from sentence_transformers import SentenceTransformer
# import numpy as np

# # ------------------------------
# # Configure Logging
# # ------------------------------
# logging.basicConfig(
#     filename="rag_pipeline.log",
#     level=logging.INFO,
#     format="%(asctime)s - %(levelname)s - %(message)s"
# )
# logging.info("Reranking Hybrid RAG pipeline loaded")


# class HybridRAGRerankPipeline:
#     def __init__(self, docs_path="docs"):
#         self.docs_path = docs_path
#         self.vectorstore = None
#         self.embedder = HuggingFaceEmbeddings(
#             model_name="sentence-transformers/all-mpnet-base-v2")
#         self.bm25 = None
#         self.documents = []
#         self.reranker = SentenceTransformer("all-MiniLM-L6-v2")
#         self._build_index()

#     def _build_index(self):
#         logging.info(
#             f"Building hybrid RAG index from docs in {self.docs_path}")
#         doc_files = glob.glob(f"{self.docs_path}/*.txt")

#         for file in doc_files:
#             try:
#                 with open(file, "r", encoding="utf-8") as f:
#                     content = f.read()
#                     doc = Document(page_content=content,
#                                    metadata={"source": file})
#                     self.documents.append(doc)
#                     logging.info(f"Loaded document: {file}")
#             except Exception as e:
#                 logging.error(f"Failed to read document {file}: {e}")

#         # Split documents for vectorstore
#         splitter = RecursiveCharacterTextSplitter(
#             chunk_size=400, chunk_overlap=100)
#         split_docs = splitter.split_documents(self.documents)
#         if split_docs:
#             self.vectorstore = FAISS.from_documents(split_docs, self.embedder)
#             logging.info("Vectorstore created")
#         else:
#             logging.warning("No documents to build vectorstore")

#         # BM25 index
#         tokenized_corpus = [doc.page_content.split() for doc in self.documents]
#         if tokenized_corpus:
#             self.bm25 = BM25Okapi(tokenized_corpus)
#             logging.info("BM25 index created")
#         else:
#             logging.warning("BM25 index not created")

#     def retrieve_context(self, query, k_semantic=5, k_keyword=5, top_k_rerank=5):
#         logging.info(f"Hybrid retrieval for query: {query}")

#         # Semantic retrieval
#         semantic_docs = self.vectorstore.similarity_search(
#             query, k=k_semantic) if self.vectorstore else []
#         logging.info(f"Semantic retrieval returned {len(semantic_docs)} docs")

#         # Keyword retrieval
#         keyword_docs = []
#         if self.bm25:
#             tokenized_query = query.split()
#             keyword_docs = self.bm25.get_top_n(
#                 tokenized_query, self.documents, n=k_keyword)
#             logging.info(
#                 f"Keyword retrieval returned {len(keyword_docs)} docs")

#         # Merge results (deduplicate)
#         merged_docs = {doc.metadata["source"]
#             : doc for doc in semantic_docs + keyword_docs}
#         merged_list = list(merged_docs.values())
#         logging.info(f"Merged docs count: {len(merged_list)}")

#         # --- Rerank merged docs ---
#         if merged_list:
#             query_emb = self.reranker.encode([query])
#             doc_embs = self.reranker.encode(
#                 [doc.page_content for doc in merged_list])
#             # cosine similarity approx
#             scores = np.dot(doc_embs, query_emb.T).flatten()
#             ranked_docs = [doc for _, doc in sorted(
#                 zip(scores, merged_list), key=lambda x: x[0], reverse=True)]
#             logging.info(
#                 f"Reranked top {min(top_k_rerank, len(ranked_docs))} docs")
#             return ranked_docs[:top_k_rerank]
#         else:
#             return []

#     def run_ollama(self, prompt):
#         logging.info("Running Ollama LLM...")
#         try:
#             result = subprocess.run(
#                 ["ollama", "run", "llama3"],
#                 input=prompt.encode(),
#                 stdout=subprocess.PIPE,
#                 stderr=subprocess.PIPE,
#                 check=True
#             )
#             output = result.stdout.decode().strip()
#             logging.info("Ollama LLM returned output successfully")
#             return output
#         except subprocess.CalledProcessError as e:
#             logging.error(f"Ollama CLI error: {e.stderr.decode().strip()}")
#             return "⚠️ Error running Ollama LLM"

#     def query(self, question):
#         logging.info(f"Processing query: {question}")
#         docs = self.retrieve_context(question)

#         if docs:
#             context = "\n\n".join([d.page_content for d in docs])
#             prompt = f"""You are a helpful AI assistant. Use the following context to answer the question.

# Context:
# {context}

# Question: {question}

# If the context is irrelevant, answer from your general knowledge.
# Answer:"""
#         else:
#             logging.warning("No relevant docs found, using fallback")
#             prompt = f"Answer this question using your general knowledge:\n{question}"

#         return self.run_ollama(prompt)
