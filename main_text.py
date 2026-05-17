"""Main entry point for RAG application"""

from src.rag import RAGSystem
import os


def main():
    """Main function"""
    print("Initializing RAG System...")

    # Initialize RAG system
    rag = RAGSystem()

    # Example: Load documents (replace with your document path)
    document_path = "./data/sample.txt"

    if not os.path.exists(document_path):
        print(f"Please add documents to {document_path}")
        return

    print(f"Loading documents from {document_path}...")
    documents = rag.load_documents(document_path)

    print(f"Creating vector store with {len(documents)} documents...")
    rag.create_vector_store(documents)

    print("Initializing QA chain...")
    rag.initialize_qa_chain()

    # Interactive query loop
    print("\nRAG System ready! Type 'quit' to exit.\n")
    while True:
        question = input("Ask a question: ").strip()
        if question.lower() == "quit":
            break

        try:
            response = rag.query(question)
            print(f"\nAnswer: {response}\n")
        except Exception as e:
            print(f"Error: {e}\n")


if __name__ == "__main__":
    main()
