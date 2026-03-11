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

    .stButton button {
        display: flex;
        align-items: center;
        justify-content: flex-start;
        gap: 0.55rem;
    }

    div.element-container:has(.danger-btn) {
        display: none;
    }

    div.element-container:has(.danger-btn) + div button:hover {
        background-color: #fee2e2;
        border-color: #fca5a5;
        color: #b91c1c;
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


@st.dialog("Delete Vector Database")
def confirm_delete_vector_db():
    st.warning("This will delete the existing FAISS index and document metadata from data/index.")
    confirmation_text = st.text_input(
        "Type `delete` to confirm",
        key="delete_vector_db_confirmation",
        placeholder="delete",
    )

    confirm_col, cancel_col = st.columns(2)
    with confirm_col:
        confirm_clicked = st.button("Confirm Delete", use_container_width=True)
    with cancel_col:
        cancel_clicked = st.button("Cancel", use_container_width=True)

    if confirm_clicked:
        if confirmation_text.strip().lower() != "delete":
            st.error("Type `delete` exactly to confirm.")
        else:
            removed_files = delete_vector_store_files()
            clear_cached_rag()
            st.session_state["vector_db_delete_result"] = removed_files
            st.session_state["delete_vector_db_confirmation"] = ""
            st.rerun()

    if cancel_clicked:
        st.session_state["delete_vector_db_confirmation"] = ""
        st.rerun()


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
        <h1> Multilingual RAG Chatbot <svg xmlns="http://www.w3.org/2000/svg" width="26" height="26" viewBox="0 0 64 64" fill="none" style="vertical-align: middle; margin-left: 6px;"><circle cx="32" cy="32" r="30" fill="#38bdf8"/><rect x="16" y="18" width="32" height="24" rx="8" fill="#f8fafc"/><rect x="20" y="22" width="24" height="16" rx="5" fill="#111827"/><circle cx="26" cy="29" r="3" fill="#67e8f9"/><circle cx="38" cy="29" r="3" fill="#67e8f9"/><path d="M26 35c2 2 4 3 6 3s4-1 6-3" stroke="#5eead4" stroke-width="3" stroke-linecap="round"/><rect x="29" y="12" width="6" height="8" rx="3" fill="#f8fafc"/><circle cx="32" cy="10" r="3" fill="#f8fafc"/></svg></h1>
        <p>Crafted by Perplex Squad</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        available_ocr_languages = get_available_ocr_languages()
        default_ocr_languages = [
            code for code in DEFAULT_OCR_LANGUAGES if code in available_ocr_languages
        ] or available_ocr_languages[:3]

        with st.expander("Model Settings", expanded=False):
            st.markdown("""
            <div class="info-box">
                <div class="info-label">LLM Model</div>
                <div class="info-value">Claude 3.5 Sonnet</div>
            </div>
            """, unsafe_allow_html=True)

            temperature = st.slider(
                "🌡️ Temperature",
                min_value=0.0,
                max_value=1.0,
                value=0.3,
                step=0.1,
                help="Controls randomness: 0.0 = deterministic, 1.0 = creative"
            )

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

        with st.expander("Document Management", expanded=False):
            uploaded_files = st.file_uploader(
                "Upload PDFs",
                type=["pdf"],
                accept_multiple_files=True,
                help="Uploaded PDFs are saved to data/raw and re-indexed into the vector database.",
            )

            upload_clicked = st.button("Upload & Index Documents", use_container_width=True, icon=":material/upload_file:")
            reindex_clicked = st.button("Reindex Existing Documents", use_container_width=True, icon=":material/sync:")
            st.markdown('<div class="danger-btn"></div>', unsafe_allow_html=True)
            delete_clicked = st.button("Delete Vector Database", use_container_width=True, icon=":material/delete:")

        with st.expander("Languages & OCR", expanded=False):
            force_ocr = st.toggle(
                "Enable OCR for scanned PDFs",
                value=False,
                help="Use OCR even when the PDF already contains an extracted text layer.",
            )
            selected_ocr_languages = st.multiselect(
                "OCR languages",
                options=available_ocr_languages,
                default=default_ocr_languages,
                format_func=format_ocr_language_option,
                help="These Tesseract language packs are used only when OCR runs during ingestion.",
            )

        with st.expander("Debug Tools", expanded=False):
            show_context = st.toggle("Show Retrieved Context", value=False)

        with st.expander("System Info", expanded=False):
            st.markdown(
                """
                <div class="info-box">
                    <div class="info-label">Embeddings</div>
                    <div class="info-value">• Amazon Titan v2</div>
                </div>
                <div class="info-box">
                    <div class="info-label">Vector Database</div>
                    <div class="info-value">• FAISS</div>
                </div>
                <div class="info-box">
                    <div class="info-label">Mode</div>
                    <div class="info-value">• Multilingual RAG</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        if st.button("Clear Chat History"):
            st.session_state.messages = []
            st.rerun()

        selected_ocr_languages_tuple = tuple(selected_ocr_languages or default_ocr_languages)

        if upload_clicked:
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

        if reindex_clicked:
            try:
                delete_vector_store_files()
                clear_cached_rag()
                with st.spinner("Reindexing documents from data/raw..."):
                    st.session_state.rag = RAGPipeline(
                        build_vector_store(
                            show_progress=True,
                            force_ocr=force_ocr,
                            ocr_languages=selected_ocr_languages_tuple,
                        )
                    )
                st.success("Reindexed the existing documents successfully.")
                st.rerun()
            except Exception as exc:
                st.error(f"Failed to reindex documents: {exc}")

        if delete_clicked:
            confirm_delete_vector_db()

        if "vector_db_delete_result" in st.session_state:
            removed_files = st.session_state.pop("vector_db_delete_result")
            if removed_files:
                st.success("Deleted the existing vector database files.")
            else:
                st.info("No vector database files were present to delete.")
    
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
