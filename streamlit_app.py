import streamlit as st
import os
import glob
import random
from typing import List, Tuple
from app.loaders.pdf_loader import PDFLoader
from app.chunker import TextChunker
from app.embedder import BedrockEmbedder
from app.vector_store import FaissVectorStore
from app.rag_pipeline import RAGPipeline
from app.models import Document
from app.language import get_language_label, resolve_target_language

# Initialize LangSmith for backend analytics
from dotenv import load_dotenv
load_dotenv()

# Set LangSmith environment variables for tracing
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2", "true")
os.environ["LANGCHAIN_ENDPOINT"] = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY", "")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGSMITH_PROJECT", "multilingual-rag-chatbot")

# Page configuration
st.set_page_config(
    page_title="Multilingual RAG Chatbot",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Fun loading messages (Slack-style)
LOADING_MESSAGES = [
    "🧠 Reading your mind...",
    "🔍 Searching the knowledge base...",
    "✨ Crafting the perfect answer...",
    "🚀 Launching brain cells...",
    "💡 Connecting the dots...",
    "🎯 Hunting for answers...",
    "🤖 Consulting the AI overlords...",
    "📚 Flipping through pages...",
    "🔮 Gazing into the crystal ball...",
    "⚡ Charging up neurons...",
    "🎨 Painting your answer...",
    "🌟 Making magic happen...",
    "🧩 Piecing it together...",
    "🎪 Performing AI acrobatics...",
    "🎭 Rehearsing the response...",
]

# Custom CSS for modern, professional look
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;600;700&display=swap');
    
    /* Main container styling */
    .main {
        background-color: #f8f9fa;
    }
    
    /* Title styling */
    .title-container {
        text-align: center;
        padding: 0.8rem 0 0.6rem 0;
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #d946ef 100%);
        color: white;
        border-radius: 8px;
        margin-bottom: 1.5rem;
        box-shadow: 0 2px 8px rgba(139, 92, 246, 0.3);
    }
    
    .title-container h1 {
        margin: 0;
        font-size: 1.5rem;
        font-weight: 600;
        font-family: 'Montserrat', sans-serif;
    }
    
    .title-container p {
        margin: 0.3rem 0 0 0;
        font-size: 0.85rem;
        opacity: 0.95;
        font-family: 'Montserrat', sans-serif;
    }
    
    /* Chat message styling */
    .stChatMessage {
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    
    /* Input box styling */
    .stChatInputContainer {
        border-top: 2px solid #e0e0e0;
        padding-top: 1rem;
    }
    
    /* Context card styling */
    .context-card {
        background-color: #f8f9fa;
        border-left: 4px solid #667eea;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 5px;
        font-size: 0.9rem;
    }
    
    .context-meta {
        color: #666;
        font-size: 0.85rem;
        margin-bottom: 0.5rem;
        font-style: italic;
    }
    
    /* Loading spinner */
    .stSpinner > div {
        border-top-color: #667eea !important;
    }
    
    /* Info boxes */
    .info-box {
        background-color: white;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    
    .info-label {
        font-weight: 600;
        color: #667eea;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .info-value {
        font-size: 1rem;
        color: #333;
        margin-top: 0.25rem;
    }
</style>
""", unsafe_allow_html=True)

INDEX_PATH = "data/index/faiss.index"
DOCS_PATH = "data/index/documents.pkl"
RAW_DATA_DIR = "data/raw"
INDEX_DIR = "data/index"
DEFAULT_OCR_LANGUAGES = (
    "eng",
    "hin",
    "tel",
    "tam",
    "rus",
    "ukr",
    "por",
    "spa",
    "fra",
    "deu",
    "ita",
)
OCR_LANGUAGE_LABELS = {
    "eng": "English",
    "hin": "Hindi",
    "tel": "Telugu",
    "tam": "Tamil",
    "rus": "Russian",
    "ukr": "Ukrainian",
    "por": "Portuguese",
    "spa": "Spanish",
    "fra": "French",
    "deu": "German",
    "ita": "Italian",
}


def ensure_data_directories():
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    os.makedirs(INDEX_DIR, exist_ok=True)


def clear_cached_rag():
    st.cache_resource.clear()
    st.session_state.pop("rag", None)


def delete_vector_store_files():
    removed_files = []
    for path in (INDEX_PATH, DOCS_PATH):
        if os.path.exists(path):
            os.remove(path)
            removed_files.append(path)
    return removed_files


def save_uploaded_pdfs(uploaded_files) -> List[str]:
    ensure_data_directories()
    saved_files = []

    for uploaded_file in uploaded_files:
        destination = os.path.join(RAW_DATA_DIR, uploaded_file.name)
        with open(destination, "wb") as output_file:
            output_file.write(uploaded_file.getbuffer())
        saved_files.append(destination)

    return saved_files


def get_available_ocr_languages() -> List[str]:
    try:
        import pytesseract

        installed = pytesseract.get_languages(config="")
        filtered = [code for code in installed if code in OCR_LANGUAGE_LABELS]
        if filtered:
            return sorted(filtered)
    except Exception:
        pass

    return list(DEFAULT_OCR_LANGUAGES)


def format_ocr_language_option(code: str) -> str:
    return f"{OCR_LANGUAGE_LABELS.get(code, code)} ({code})"


def build_vector_store(
    show_progress: bool = False,
    force_ocr: bool = False,
    ocr_languages: Tuple[str, ...] = DEFAULT_OCR_LANGUAGES,
):
    ensure_data_directories()
    embedder = BedrockEmbedder()
    loader = PDFLoader(force_ocr=force_ocr, ocr_languages=list(ocr_languages))
    chunker = TextChunker(chunk_size=200, overlap=20)

    all_docs = []
    pdf_files = glob.glob(os.path.join(RAW_DATA_DIR, "*.pdf"))

    if not pdf_files:
        raise FileNotFoundError("No PDF files found in data/raw folder")

    progress_bar = None
    status_text = None
    if show_progress:
        progress_bar = st.progress(0)
        status_text = st.empty()

    for idx, file_path in enumerate(pdf_files):
        if status_text is not None:
            status_text.text(f"Loading: {os.path.basename(file_path)}")
        all_docs.extend(loader.load(file_path))
        if progress_bar is not None:
            progress_bar.progress((idx + 1) / len(pdf_files))

    if status_text is not None:
        status_text.text("Creating chunks...")
    chunks = chunker.chunk_documents(all_docs)

    if status_text is not None:
        status_text.text("Generating embeddings...")
    embeddings = embedder.embed_documents(chunks)

    vector_store = FaissVectorStore(embedding_dim=len(embeddings[0]))
    vector_store.add_embeddings(embeddings, chunks)
    vector_store.save(INDEX_PATH, DOCS_PATH)

    if progress_bar is not None:
        progress_bar.empty()
    if status_text is not None:
        status_text.empty()

    return vector_store

@st.cache_resource
def load_vector_store(
    force_ocr: bool = False,
    ocr_languages: Tuple[str, ...] = DEFAULT_OCR_LANGUAGES,
):
    """Load or build the FAISS vector store"""
    ensure_data_directories()

    if os.path.exists(INDEX_PATH) and os.path.exists(DOCS_PATH):
        vector_store = FaissVectorStore(embedding_dim=1024)
        vector_store.load(INDEX_PATH, DOCS_PATH)
        return vector_store

    return build_vector_store(
        show_progress=True,
        force_ocr=force_ocr,
        ocr_languages=ocr_languages,
    )

def get_answer_with_context(rag: RAGPipeline, question: str, top_k: int = 10, max_tokens: int = 500, temperature: float = 0.3) -> Tuple[str, List[Document]]:
    """Get answer and retrieved context"""
    from app.models import Document
    
    query_doc = Document(content=question, metadata={"type": "query"})
    query_embedding = rag.embedder.embed_documents([query_doc])[0]
    
    retrieved_docs = rag.vector_store.search(query_embedding, top_k=top_k)
    
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
    
    answer, usage_metadata = rag.llm.generate(prompt, max_tokens=max_tokens, temperature=temperature)
    return answer, retrieved_docs, usage_metadata

def main():
    # Header
    st.markdown("""
    <div class="title-container">
        <h1> Multilingual RAG Chatbot 🤖</h1>
        <p> Crafted by Perplex Squad with<svg xmlns="http://www.w3.org/2000/svg" x="0px" y="0px" width="24" height="24" viewBox="0 0 48 48" style="vertical-align: middle; margin-left: 6px;">
<path fill="#252f3e" d="M13.527,21.529c0,0.597,0.064,1.08,0.176,1.435c0.128,0.355,0.287,0.742,0.511,1.161 c0.08,0.129,0.112,0.258,0.112,0.371c0,0.161-0.096,0.322-0.303,0.484l-1.006,0.677c-0.144,0.097-0.287,0.145-0.415,0.145 c-0.16,0-0.319-0.081-0.479-0.226c-0.224-0.242-0.415-0.5-0.575-0.758c-0.16-0.274-0.319-0.58-0.495-0.951 c-1.245,1.483-2.81,2.225-4.694,2.225c-1.341,0-2.411-0.387-3.193-1.161s-1.181-1.806-1.181-3.096c0-1.37,0.479-2.483,1.453-3.321 s2.267-1.258,3.911-1.258c0.543,0,1.102,0.048,1.692,0.129s1.197,0.21,1.836,0.355v-1.177c0-1.225-0.255-2.08-0.75-2.58 c-0.511-0.5-1.373-0.742-2.602-0.742c-0.559,0-1.133,0.064-1.724,0.21c-0.591,0.145-1.165,0.322-1.724,0.548 c-0.255,0.113-0.447,0.177-0.559,0.21c-0.112,0.032-0.192,0.048-0.255,0.048c-0.224,0-0.335-0.161-0.335-0.5v-0.79 c0-0.258,0.032-0.451,0.112-0.564c0.08-0.113,0.224-0.226,0.447-0.339c0.559-0.29,1.229-0.532,2.012-0.726 c0.782-0.21,1.612-0.306,2.49-0.306c1.9,0,3.289,0.435,4.183,1.306c0.878,0.871,1.325,2.193,1.325,3.966v5.224H13.527z M7.045,23.979c0.527,0,1.07-0.097,1.644-0.29c0.575-0.193,1.086-0.548,1.517-1.032c0.255-0.306,0.447-0.645,0.543-1.032 c0.096-0.387,0.16-0.855,0.16-1.403v-0.677c-0.463-0.113-0.958-0.21-1.469-0.274c-0.511-0.064-1.006-0.097-1.501-0.097 c-1.07,0-1.852,0.21-2.379,0.645s-0.782,1.048-0.782,1.854c0,0.758,0.192,1.322,0.591,1.709 C5.752,23.786,6.311,23.979,7.045,23.979z M19.865,25.721c-0.287,0-0.479-0.048-0.607-0.161c-0.128-0.097-0.239-0.322-0.335-0.629 l-3.752-12.463c-0.096-0.322-0.144-0.532-0.144-0.645c0-0.258,0.128-0.403,0.383-0.403h1.565c0.303,0,0.511,0.048,0.623,0.161 c0.128,0.097,0.223,0.322,0.319,0.629l2.682,10.674l2.49-10.674c0.08-0.322,0.176-0.532,0.303-0.629 c0.128-0.097,0.351-0.161,0.639-0.161h1.277c0.303,0,0.511,0.048,0.639,0.161c0.128,0.097,0.239,0.322,0.303,0.629l2.522,10.803 l2.762-10.803c0.096-0.322,0.208-0.532,0.319-0.629c0.128-0.097,0.335-0.161,0.623-0.161h1.485c0.255,0,0.399,0.129,0.399,0.403 c0,0.081-0.016,0.161-0.032,0.258s-0.048,0.226-0.112,0.403l-3.847,12.463c-0.096,0.322-0.208,0.532-0.335,0.629 s-0.335,0.161-0.607,0.161h-1.373c-0.303,0-0.511-0.048-0.639-0.161c-0.128-0.113-0.239-0.322-0.303-0.645l-2.474-10.4 L22.18,24.915c-0.08,0.322-0.176,0.532-0.303,0.645c-0.128,0.113-0.351,0.161-0.639,0.161H19.865z M40.379,26.156 c-0.83,0-1.66-0.097-2.458-0.29c-0.798-0.193-1.421-0.403-1.836-0.645c-0.255-0.145-0.431-0.306-0.495-0.451 c-0.064-0.145-0.096-0.306-0.096-0.451v-0.822c0-0.339,0.128-0.5,0.367-0.5c0.096,0,0.192,0.016,0.287,0.048 c0.096,0.032,0.239,0.097,0.399,0.161c0.543,0.242,1.133,0.435,1.756,0.564c0.639,0.129,1.261,0.193,1.9,0.193 c1.006,0,1.788-0.177,2.331-0.532c0.543-0.355,0.83-0.871,0.83-1.532c0-0.451-0.144-0.822-0.431-1.129 c-0.287-0.306-0.83-0.58-1.612-0.838l-2.315-0.726c-1.165-0.371-2.027-0.919-2.554-1.645c-0.527-0.709-0.798-1.499-0.798-2.338 c0-0.677,0.144-1.274,0.431-1.79s0.671-0.967,1.149-1.322c0.479-0.371,1.022-0.645,1.66-0.838C39.533,11.081,40.203,11,40.906,11 c0.351,0,0.718,0.016,1.07,0.064c0.367,0.048,0.702,0.113,1.038,0.177c0.319,0.081,0.623,0.161,0.91,0.258s0.511,0.193,0.671,0.29 c0.224,0.129,0.383,0.258,0.479,0.403c0.096,0.129,0.144,0.306,0.144,0.532v0.758c0,0.339-0.128,0.516-0.367,0.516 c-0.128,0-0.335-0.064-0.607-0.193c-0.91-0.419-1.932-0.629-3.065-0.629c-0.91,0-1.628,0.145-2.123,0.451 c-0.495,0.306-0.75,0.774-0.75,1.435c0,0.451,0.16,0.838,0.479,1.145c0.319,0.306,0.91,0.613,1.756,0.887l2.267,0.726 c1.149,0.371,1.98,0.887,2.474,1.548s0.734,1.419,0.734,2.257c0,0.693-0.144,1.322-0.415,1.87 c-0.287,0.548-0.671,1.032-1.165,1.419c-0.495,0.403-1.086,0.693-1.772,0.903C41.943,26.043,41.193,26.156,40.379,26.156z"></path><path fill="#f90" d="M43.396,33.992c-5.252,3.918-12.883,5.998-19.445,5.998c-9.195,0-17.481-3.434-23.739-9.142 c-0.495-0.451-0.048-1.064,0.543-0.709c6.769,3.966,15.118,6.369,23.755,6.369c5.827,0,12.229-1.225,18.119-3.741 C43.508,32.364,44.258,33.347,43.396,33.992z M45.583,31.477c-0.671-0.871-4.438-0.419-6.146-0.21 c-0.511,0.064-0.591-0.387-0.128-0.726c3.001-2.128,7.934-1.516,8.509-0.806c0.575,0.726-0.16,5.708-2.969,8.094 c-0.431,0.371-0.846,0.177-0.655-0.306C44.833,35.927,46.254,32.331,45.583,31.477z"></path>
</svg></p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 🔧 System Information")
        
        st.markdown("""
        <div class="info-box">
            <div class="info-label">LLM Model</div>
            <div class="info-value">Claude 3.5 Sonnet</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### ⚙️ LLM Parameters & Recommended Settings 💡")
        
        # Temperature slider
        temperature = st.slider(
            "🌡️ Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.3,
            step=0.1,
            help="Controls randomness: 0.0 = deterministic, 1.0 = creative"
        )
        
        # Max tokens slider
        max_tokens = st.slider(
            "📏 Max Tokens",
            min_value=100,
            max_value=4000,
            value=500,
            step=100,
            help="Maximum length of the response"
        )
        
        st.markdown(f"""
        <div style="font-size: 0.85em; color: #888; margin-top: 10px;">
            <b>Current Settings:</b><br/>
            Temperature: {temperature}, Max Tokens: {max_tokens}<br/>
        </div>
        """, unsafe_allow_html=True)
        
        # Recommended settings guide
        # st.markdown("#### 💡 Recommended Settings")
        st.markdown("""
        <div style="font-size: 0.8em; background-color: #f8f9fa; padding: 10px; border-radius: 5px; margin-top: 10px;">
            <table style="width: 100%; border-collapse: collapse;">
                <thead>
                    <tr style="border-bottom: 2px solid #dee2e6;">
                        <th style="text-align: left; padding: 5px;">Use Case</th>
                        <th style="text-align: center; padding: 5px;">Temp</th>
                        <th style="text-align: center; padding: 5px;">Tokens</th>
                    </tr>
                </thead>
                <tbody>
                    <tr style="border-bottom: 1px solid #dee2e6;">
                        <td style="padding: 5px;">📚 Factual Q&A</td>
                        <td style="text-align: center; padding: 5px;">0.0-0.3</td>
                        <td style="text-align: center; padding: 5px;">500-1000</td>
                    </tr>
                    <tr style="border-bottom: 1px solid #dee2e6;">
                        <td style="padding: 5px;">📝 Summaries</td>
                        <td style="text-align: center; padding: 5px;">0.3-0.5</td>
                        <td style="text-align: center; padding: 5px;">1000-2000</td>
                    </tr>
                    <tr>
                        <td style="padding: 5px;">🎨 Creative</td>
                        <td style="text-align: center; padding: 5px;">0.7-1.0</td>
                        <td style="text-align: center; padding: 5px;">1000-4000</td>
                    </tr>
                </tbody>
            </table>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.markdown("""
        <div class="info-box">
            <div class="info-label">Embeddings</div>
            <div class="info-value">Amazon Titan v2</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box">
            <div class="info-label">Vector Database</div>
            <div class="info-value">FAISS</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box">
            <div class="info-label">Mode</div>
            <div class="info-value">Multilingual RAG</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")

        st.markdown("### 📚 Document Management")
        force_ocr = st.checkbox(
            "Force OCR during ingestion",
            value=False,
            help="Use OCR even when the PDF already contains an extracted text layer."
        )
        available_ocr_languages = get_available_ocr_languages()
        default_ocr_languages = [
            code for code in DEFAULT_OCR_LANGUAGES if code in available_ocr_languages
        ] or available_ocr_languages[:3]
        selected_ocr_languages = st.multiselect(
            "OCR languages",
            options=available_ocr_languages,
            default=default_ocr_languages,
            format_func=format_ocr_language_option,
            help="These Tesseract languages are used only when OCR runs during ingestion."
        )
        uploaded_files = st.file_uploader(
            "Upload PDF documents",
            type=["pdf"],
            accept_multiple_files=True,
            help="Uploaded PDFs are saved to data/raw and re-indexed into the vector database."
        )
        selected_ocr_languages_tuple = tuple(selected_ocr_languages or default_ocr_languages)

        if st.button("📥 Upload & Ingest Documents", use_container_width=True):
            if not uploaded_files:
                st.warning("Select at least one PDF file to upload.")
            elif not selected_ocr_languages_tuple:
                st.warning("Select at least one OCR language.")
            else:
                try:
                    saved_files = save_uploaded_pdfs(uploaded_files)
                    delete_vector_store_files()
                    clear_cached_rag()
                    with st.spinner("Uploading files and rebuilding the vector database..."):
                        st.session_state.rag = RAGPipeline(
                            build_vector_store(
                                show_progress=True,
                                force_ocr=force_ocr,
                                ocr_languages=selected_ocr_languages_tuple,
                            )
                        )
                    st.success(
                        f"Uploaded {len(saved_files)} file(s) and rebuilt the vector database."
                    )
                    st.rerun()
                except Exception as exc:
                    st.error(f"Failed to upload and ingest documents: {exc}")

        if st.button("🗑️ Delete Vector DB Data", use_container_width=True):
            try:
                removed_files = delete_vector_store_files()
                clear_cached_rag()
                if removed_files:
                    st.success("Deleted the existing vector database files.")
                else:
                    st.info("No vector database files were present to delete.")
                st.rerun()
            except Exception as exc:
                st.error(f"Failed to delete vector database files: {exc}")

        st.caption("Deleting the vector DB removes only indexed data in data/index. Source PDFs in data/raw are kept.")
        st.caption(
            "OCR uses the selected Tesseract language packs when OCR is triggered or forced."
        )

        st.markdown("---")
        
        # Toggle for showing context
        show_context = st.toggle("📄 Show Retrieved Context", value=False)
        
        st.markdown("---")
        
        st.markdown("### 🌍 Supported Languages")
        st.markdown("""
        - European languages such as English, French, German, Spanish, Portuguese, Italian, Russian, Ukrainian, and more
        - Indian languages such as Hindi, Telugu, Tamil, and more
        - Arabic and other widely used languages
        """)
        
        st.markdown("---")
        
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.rerun()
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "rag" not in st.session_state:
        try:
            with st.spinner("🔄 Loading vector store and initializing RAG pipeline..."):
                vector_store = load_vector_store(
                    force_ocr=force_ocr,
                    ocr_languages=selected_ocr_languages_tuple,
                )
                st.session_state.rag = RAGPipeline(vector_store)
        except FileNotFoundError:
            st.info("Upload PDFs from the sidebar to build the knowledge base.")
            st.stop()
    
    # Display chat history
    for message in st.session_state.messages:
        avatar = "👤" if message["role"] == "user" else "🤖"
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])
            
            # Show usage metadata if available for assistant messages
            if message["role"] == "assistant" and "usage" in message:
                usage = message["usage"]
                st.markdown(f"""
                <div style="font-size: 0.75em; color: #888; margin-top: 15px; padding: 8px; background-color: #f0f0f0; border-radius: 5px;">
                    <b>📊 Generation Stats:</b> 
                    Temperature: {usage['temperature']} | 
                    Tokens: {usage['output_tokens']} / {usage['max_tokens']} | 
                    Total: {usage['total_tokens']} (Input: {usage['input_tokens']})
                </div>
                """, unsafe_allow_html=True)
            
            # Show context if available and toggle is on
            if show_context and message["role"] == "assistant" and "context" in message:
                with st.expander("📚 Retrieved Context Chunks"):
                    for i, doc in enumerate(message["context"], 1):
                        source = doc.metadata.get("source", "Unknown")
                        page = doc.metadata.get("page", "?")
                        ocr = doc.metadata.get("ocr", False)
                        
                        st.markdown(f"""
                        <div class="context-card">
                            <div class="context-meta">
                                📄 Chunk {i} | Source: {os.path.basename(source)} | Page: {page} | OCR: {ocr}
                            </div>
                            <div>{doc.content[:500]}{'...' if len(doc.content) > 500 else ''}</div>
                        </div>
                        """, unsafe_allow_html=True)
    
    # Chat input
    if prompt := st.chat_input("🤗 Ask your question in any language. Let's go!"):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant", avatar="🤖"):
            # Pick a random loading message
            loading_message = random.choice(LOADING_MESSAGES)
            with st.spinner(loading_message):
                try:
                    answer, context_docs, usage_metadata = get_answer_with_context(
                        st.session_state.rag, 
                        prompt, 
                        top_k=10,
                        max_tokens=max_tokens,
                        temperature=temperature
                    )
                    
                    st.markdown(answer)
                    
                    # Display token usage metadata
                    st.markdown(f"""
                    <div style="font-size: 0.75em; color: #888; margin-top: 15px; padding: 8px; background-color: #f0f0f0; border-radius: 5px;">
                        <b>📊 Generation Stats:</b> 
                        Temperature: {usage_metadata['temperature']} | 
                        Tokens: {usage_metadata['output_tokens']} / {usage_metadata['max_tokens']} | 
                        Total: {usage_metadata['total_tokens']} (Input: {usage_metadata['input_tokens']})
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Store message with context and metadata
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": answer,
                        "context": context_docs,
                        "usage": usage_metadata
                    })
                    
                    # Show context if toggle is on
                    if show_context:
                        with st.expander("📚 Retrieved Context Chunks"):
                            for i, doc in enumerate(context_docs, 1):
                                source = doc.metadata.get("source", "Unknown")
                                page = doc.metadata.get("page", "?")
                                ocr = doc.metadata.get("ocr", False)
                                
                                st.markdown(f"""
                                <div class="context-card">
                                    <div class="context-meta">
                                        📄 Chunk {i} | Source: {os.path.basename(source)} | Page: {page} | OCR: {ocr}
                                    </div>
                                    <div>{doc.content[:500]}{'...' if len(doc.content) > 500 else ''}</div>
                                </div>
                                """, unsafe_allow_html=True)
                    
                except Exception as e:
                    error_msg = f"❌ Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": error_msg
                    })

if __name__ == "__main__":
    main()
