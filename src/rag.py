"""Main RAG Implementation"""

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_classic.chains import RetrievalQA

import sys
sys.path.append('..')
from config.settings import (
    HF_MODEL_NAME,
    VECTOR_DB_PATH,
    EMBEDDING_MODEL
)


class RAGSystem:
    """Retrieval-Augmented Generation System"""

    def __init__(self):
        """Initialize RAG system with embeddings and LLM"""
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        # Using HuggingFaceHub for free inference
        endpoint = HuggingFaceEndpoint(
            repo_id=HF_MODEL_NAME,
            task="conversational",
            temperature=0.7,
            max_new_tokens=512,
        )
        self.llm = ChatHuggingFace(llm=endpoint)
        self.vector_store = None
        self.qa_chain = None

    def load_documents(self, file_path: str):
        """Load documents from file"""
        loader = TextLoader(file_path)
        documents = loader.load()

        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        docs = text_splitter.split_documents(documents)
        return docs

    def create_vector_store(self, documents):
        """Create vector store from documents"""
        self.vector_store = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=VECTOR_DB_PATH
        )

    def initialize_qa_chain(self):
        """Initialize QA chain"""
        if self.vector_store is None:
            raise ValueError("Vector store not initialized. Load documents first.")

        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(search_kwargs={"k": 3})
        )

    def query(self, question: str) -> str:
        """Query the RAG system"""
        if self.qa_chain is None:
            raise ValueError("QA chain not initialized.")

        result = self.qa_chain.invoke({"query": question})
        return result["result"]
