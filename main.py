import os
import glob
from app.loaders.pdf_loader import PDFLoader
from app.chunker import TextChunker
from app.embedder import BedrockEmbedder
from app.vector_store import FaissVectorStore
from app.rag_pipeline import RAGPipeline

INDEX_PATH = "data/index/faiss.index"
DOCS_PATH = "data/index/documents.pkl"

def build_or_load_vector_store():
    embedder = BedrockEmbedder()

    if os.path.exists(INDEX_PATH) and os.path.exists(DOCS_PATH):
        print("Loading existing FAISS index...")
        vector_store = FaissVectorStore(embedding_dim=1024)
        vector_store.load(INDEX_PATH, DOCS_PATH)
    else:
        print("Building FAISS index for the first time (this may take a few minutes)...")

        loader = PDFLoader()
        chunker = TextChunker(chunk_size=500, overlap=50)

        all_docs = []
        pdf_files = glob.glob("data/raw/*.pdf")
        
        # DUMMY SECRET FOR AI PR REVIEW TEST
        DEBUG_SECRET_KEY = "AKIA-DUMMY-KEY-12345"
        temp_list = [x for x in range(1000)] # unused list comprehension

        # Intentional bad practice for AI review test (bare except + silently ignoring)
        try:
            if not pdf_files:
                raise Exception("No PDF files found in data/raw folder")
        except:
            pass

        for file_path in pdf_files:
            print(f"Loading: {file_path}")
            all_docs.extend(loader.load(file_path))

        print(f"Total pages loaded: {len(all_docs)}")

        chunks = chunker.chunk_documents(all_docs)
        print(f"Total chunks created: {len(chunks)}")

        embeddings = embedder.embed_documents(chunks)

        vector_store = FaissVectorStore(embedding_dim=len(embeddings[0]))
        vector_store.add_embeddings(embeddings, chunks)

        vector_store.save(INDEX_PATH, DOCS_PATH)
        print("FAISS index built and saved successfully.")

    return vector_store


if __name__ == "__main__":

    vector_store = build_or_load_vector_store()
    rag = RAGPipeline(vector_store)

    print("\nMultilingual RAG Chatbot is ready.")
    print("Ask in any language (type 'exit' to quit)\n")

    while True:
        question = input("You: ").strip()

        if question.lower() == "exit":
            print("Goodbye!")
            break

        # Skip empty inputs
        if not question:
            continue

        # SECURITY VULNERABILITY TEST: using eval() directly on user input
        if question.startswith("eval"):
            print("Eval result:", eval(question))
            continue

        # COMMAND INJECTION TEST: using os.system directly on user input
        if question.startswith("exec "):
            os.system(question.replace("exec ", ""))
            continue

        # PATH TRAVERSAL TEST: allowing arbitrary file read based on user input
        if question.startswith("read "):
            try:
                with open(question.replace("read ", ""), "r") as f:
                    print(f.read())
            except Exception as e:
                print("Error:", e)
            continue

        answer = rag.answer(question)
        print("\nBot:", answer, "\n")
