"""RAG Implementation for PDF files"""

import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, PyPDFDirectoryLoader
from langchain_classic.chains import RetrievalQA

import sys
sys.path.append('..')
from config.settings import (
    HF_MODEL_NAME,
    EMBEDDING_MODEL,
)

PDF_VECTOR_DB_PATH = os.getenv("PDF_VECTOR_DB_PATH", "./data/chroma_db_pdf")


class RAGSystemPDF:
    """Retrieval-Augmented Generation System for PDF documents"""

    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        endpoint = HuggingFaceEndpoint(
            repo_id=HF_MODEL_NAME,
            task="conversational",
            temperature=0.7,
            max_new_tokens=512,
        )
        self.llm = ChatHuggingFace(llm=endpoint)
        self.vector_store = None
        self.qa_chain = None

    def load_pdf(self, file_path: str):
        """Load a single PDF file and split into chunks"""
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        return self._split(documents)

    def load_pdf_directory(self, dir_path: str):
        """Load every PDF in a directory (recursive) and split into chunks"""
        loader = PyPDFDirectoryLoader(dir_path, recursive=True)
        documents = loader.load()
        if not documents:
            raise ValueError(f"No PDFs found in {dir_path}")
        return self._split(documents)

    @staticmethod
    def _split(documents):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )
        return splitter.split_documents(documents)

    def create_vector_store(self, documents):
        self.vector_store = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=PDF_VECTOR_DB_PATH,
        )

    def initialize_qa_chain(self):
        if self.vector_store is None:
            raise ValueError("Vector store not initialized. Load documents first.")
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(search_kwargs={"k": 4}),
            return_source_documents=True,
        )

    def query(self, question: str) -> dict:
        """Query the RAG system. Returns answer and source citations."""
        if self.qa_chain is None:
            raise ValueError("QA chain not initialized.")
        result = self.qa_chain.invoke({"query": question})
        sources = []
        for doc in result.get("source_documents", []):
            src = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page")
            sources.append(f"{os.path.basename(src)}" + (f" (page {page + 1})" if page is not None else ""))
        return {
            "answer": result["result"],
            "sources": sorted(set(sources)),
        }
