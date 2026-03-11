from app.embedder import BedrockEmbedder
from app.vector_store import FaissVectorStore
from app.models import Document
from app.llm import ClaudeClient
from langsmith import traceable


class RAGPipeline:
    def __init__(self, vector_store: FaissVectorStore):
        self.embedder = BedrockEmbedder()
        self.vector_store = vector_store
        self.llm = ClaudeClient()

    @traceable(run_type="chain", name="RAG Query Pipeline")
    def answer(self, question: str, top_k: int = 10, max_tokens: int = 500, temperature: float = 0.3) -> str:
        query_doc = Document(content=question, metadata={"type": "query"})
        query_embedding = self.embedder.embed_documents([query_doc])[0]

        retrieved_docs = self.vector_store.search(query_embedding, top_k=top_k)

        # 🔍 DEBUG: Print retrieved chunks
        print("\n--- Retrieved Chunks from Vector DB ---")
        for i, doc in enumerate(retrieved_docs, 1):
            src = doc.metadata.get("source")
            page = doc.metadata.get("page")
            ocr = doc.metadata.get("ocr")
            print(
                f"\nChunk {i} (source={src}, page={page}, ocr={ocr}):\n"
                f"{doc.content[:800]}"
            )
        print("\n-------------------------------------\n")

        context = "\n\n".join([doc.content for doc in retrieved_docs])

        prompt = f"""You are a multilingual knowledge assistant with STRICT grounding requirements.

⚠️ CRITICAL RULES:
1. MULTILINGUAL CONTEXT HANDLING:
   - The Context below may be in ANY language (English, Telugu తెలుగు, Hindi हिंदी, French, Arabic, etc.)
   - You MUST read, understand, and USE the Context regardless of what language it's written in
   - If Context language ≠ Response language, you MUST translate the information
   - Use NATIVE SCRIPT for target language (देवनागरी for Hindi, తెలుగు for Telugu, etc.)
   
2. RESPONSE LANGUAGE PRIORITY:
   - FIRST: Check if Question explicitly requests a language (e.g., "in French", "en español", "हिंदी में")
     → If yes, answer in that requested language
   - SECOND: If no explicit language request, detect the Question's language and answer in that language
   - Examples:
     * "Tell me about X in French" → Answer in French (explicit request)
     * "क्या आपके पास जानकारी है?" → Answer in Hindi (query language)
     * "Do you have information?" → Answer in English (query language)
   
3. GROUNDING: Answer using ONLY information from the Context below

4. WHEN TO SAY "INFORMATION NOT AVAILABLE":
   - ONLY if the Context doesn't contain the factual information being asked about
   - DO NOT say this just because Context is in a different language
   - Language difference ≠ Information unavailable

5. DO NOT use external knowledge beyond what's in the Context

YOUR PROCESS:
Step 1: Check if Question explicitly requests a specific language
Step 2: If yes, use that language; if no, use the Question's language
Step 3: Read and understand the Context (regardless of its language)
Step 4: Extract the relevant information that answers the Question
Step 5: Translate that information to the target language (from Step 1-2)
Step 6: Provide the COMPLETE answer in the target language with native script

EXAMPLES:
✓ "Boy who cried wolf in French" → Full answer in Français
✓ Question in हिंदी → Full answer in हिंदी (even if Context is in తెలుగు)
✓ Question in English (no language request) → Full answer in English

Context:
{context}

Question:
{question}

Answer (in requested language OR Question's language, with native script):
"""

        answer, usage_metadata = self.llm.generate(prompt, max_tokens=max_tokens, temperature=temperature)
        return answer, usage_metadata
