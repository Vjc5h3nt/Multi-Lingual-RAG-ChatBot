from app.embedder import BedrockEmbedder
from app.vector_store import FaissVectorStore
from app.models import Document
from app.llm import ClaudeClient
from app.language import get_language_label, resolve_target_language
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
        target_language_code, target_language_reason = resolve_target_language(question)
        target_language_label = get_language_label(target_language_code)

        prompt = f"""You are a multilingual knowledge assistant with STRICT grounding requirements.

TARGET RESPONSE LANGUAGE: {target_language_label} ({target_language_code})
LANGUAGE SELECTION REASON: {target_language_reason}

FINAL LANGUAGE REQUIREMENT:
- Your entire answer MUST be written in {target_language_label}
- Do not answer in the Context language unless {target_language_label} is also the target language
- If the Context is in another language, translate the relevant information into {target_language_label}

⚠️ CRITICAL RULES:
1. MULTILINGUAL CONTEXT HANDLING:
   - The Context below may be in ANY language, including major European languages plus Indian and other widely used languages
   - You MUST read, understand, and USE the Context regardless of what language it's written in
   - If Context language ≠ Response language, you MUST translate the information
   - Use NATIVE SCRIPT for target language (for example: Devanagari for Hindi, తెలుగు for Telugu, தமிழ் for Tamil, Cyrillic for Russian/Ukrainian, etc.)
   
2. RESPONSE LANGUAGE PRIORITY:
   - FIRST: Check if the Question explicitly requests a language (e.g., "in French", "en español", "हिंदी में")
     → If yes, answer in that requested language
   - SECOND: If no explicit language is requested, answer in the dominant language used by the user in the full Question
   - For mixed-language Questions, choose the language that represents most of the sentence
   - A short greeting or a few words in another language should NOT override the dominant language of the full Question
   - Determine the response language from the user's Question ONLY, never from the Context language
   - If the Question is in English, the answer MUST be fully in English unless the user explicitly requests another language
   - NEVER switch to Hindi, Telugu, or any other Context language unless the user explicitly asks for that language
   - DO NOT maintain or fall back to any default language
   - Examples:
     * "Tell me about X in French" → Answer in French (explicit request)
     * "क्या आपके पास जानकारी है?" → Answer in Hindi (same language as the user)
     * "Do you have information?" → Answer in English (same language as the user)
     * "Привіт, how is it going?" → Answer in English because most of the sentence is English
     * "Привіт, how is it going? Reply in Ukrainian." → Answer in Ukrainian because the user explicitly requested it
     * English Question + Hindi Context → Answer in English by translating the retrieved Hindi information
   
3. GROUNDING: Answer using ONLY information from the Context below

4. WHEN TO SAY "INFORMATION NOT AVAILABLE":
   - ONLY if the Context doesn't contain the factual information being asked about
   - DO NOT say this just because Context is in a different language
   - Language difference ≠ Information unavailable

5. DO NOT use external knowledge beyond what's in the Context

YOUR PROCESS:
Step 1: Check if the Question explicitly requests a specific language
Step 2: If yes, use that language; if no, identify the dominant language of the user's full Question and respond in that language
Step 3: Ignore the Context language when deciding the answer language
Step 4: Read and understand the Context (regardless of its language)
Step 5: Extract the relevant information that answers the Question
Step 6: Translate that information to the target language (from Step 1-2)
Step 7: Provide the COMPLETE answer in the target language with native script

EXAMPLES:
✓ "Boy who cried wolf in French" → Full answer in Français
✓ Question in हिंदी → Full answer in हिंदी (even if Context is in తెలుగు)
✓ Question in English (no language request) → Full answer in English
✓ User asks in Telugu → Full answer in Telugu unless another language is explicitly requested
✓ Mixed question with mostly English text → Full answer in English unless another language is explicitly requested
✓ Portuguese Question + Hindi Context → Full answer in Portuguese
✓ Russian Question + English Context → Full answer in Russian
✓ German Question + Hindi Context → Full answer in German

Context:
{context}

Question:
{question}

Answer (in requested language OR Question's language, with native script):
"""

        answer, usage_metadata = self.llm.generate(prompt, max_tokens=max_tokens, temperature=temperature)
        return answer, usage_metadata
