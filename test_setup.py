#!/usr/bin/env python3
"""Quick test of RAG system"""

try:
    print("Testing imports...")
    import langchain
    print(f"✓ LangChain {langchain.__version__}")
    
    import chromadb
    print("✓ ChromaDB")
    
    import sentence_transformers
    print("✓ Sentence Transformers")
    
    print("\n✅ All dependencies installed successfully!")
    print("\nYou can now run: python main.py")
    
except ImportError as e:
    print(f"❌ Missing module: {e}")
    print("Run: pip install langchain chromadb sentence-transformers")
