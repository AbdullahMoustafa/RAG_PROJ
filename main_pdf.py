"""Entry point for the PDF-based RAG application"""

import os
from src.rag_pdf import RAGSystemPDF


def main():
    print("Initializing PDF RAG System...")
    rag = RAGSystemPDF()

    pdf_dir = "./data/pdf"
    if not os.path.isdir(pdf_dir):
        print(f"PDF directory not found: {pdf_dir}")
        return

    print(f"Loading PDFs from {pdf_dir}...")
    documents = rag.load_pdf_directory(pdf_dir)

    print(f"Creating vector store with {len(documents)} chunks...")
    rag.create_vector_store(documents)

    print("Initializing QA chain...")
    rag.initialize_qa_chain()

    print("\nPDF RAG System ready! Type 'quit' to exit.\n")
    while True:
        question = input("Ask a question: ").strip()
        if question.lower() == "quit":
            break
        if not question:
            continue

        try:
            response = rag.query(question)
            print(f"\nAnswer: {response['answer']}")
            if response["sources"]:
                print("\nSources:")
                for s in response["sources"]:
                    print(f"  - {s}")
            print()
        except Exception as e:
            print(f"Error: {e}\n")


if __name__ == "__main__":
    main()
