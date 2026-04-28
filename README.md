# Multi-Lingual RAG ChatBot

The Multi-Lingual RAG ChatBot is a document-grounded question-answering application for multilingual PDF collections. It ingests PDF files, extracts text directly or through OCR when needed, builds a FAISS vector index using Amazon Bedrock embeddings, and answers user questions through a retrieval-augmented generation pipeline backed by Anthropic Claude on AWS Bedrock.

The project includes both a Streamlit web interface and a command-line entry point. It is designed for teams that need a practical way to query multilingual document sets while preserving strict grounding to retrieved content.

## Key Capabilities

- Multilingual question answering with automatic response-language selection
- PDF ingestion with fallback OCR for scanned or low-quality text layers
- Retrieval over chunked document content using FAISS
- Amazon Titan text embeddings through AWS Bedrock
- Claude-based answer generation through AWS Bedrock
- Optional LangSmith tracing for token usage and cost visibility
- Streamlit interface for document upload, reindexing, and interactive chat

## Architecture

The application follows a straightforward RAG pipeline:

1. PDF files are read from `data/raw/`.
2. `PDFLoader` extracts text with `pypdf` and falls back to OCR when extraction quality is weak.
3. `TextChunker` splits page content into overlapping chunks.
4. `BedrockEmbedder` creates vector embeddings with Amazon Titan.
5. `FaissVectorStore` stores embeddings and document metadata locally.
6. `RAGPipeline` retrieves relevant chunks and builds a grounded prompt.
7. `ClaudeClient` generates the final answer using the configured Bedrock model.

## Repository Structure

```text
.
|-- app/
|   |-- chunker.py
|   |-- config.py
|   |-- embedder.py
|   |-- language.py
|   |-- llm.py
|   |-- models.py
|   |-- pricing.py
|   |-- rag_pipeline.py
|   |-- vector_store.py
|   `-- loaders/
|       |-- base_loader.py
|       `-- pdf_loader.py
|-- data/
|   |-- index/
|   `-- raw/
|-- .streamlit/
|   `-- config.toml
|-- main.py
|-- streamlit_app.py
|-- requirements.txt
`-- catalog-info.yaml
```

## Technology Stack

- Python
- Streamlit
- AWS Bedrock
- Anthropic Claude
- Amazon Titan Embeddings
- FAISS
- pypdf
- Tesseract OCR
- LangSmith

## Prerequisites

Before running the project, make sure the following dependencies are available:

- Python 3.10 or later
- AWS account access with Bedrock model permissions
- Tesseract OCR installed locally
- Poppler installed locally for PDF-to-image conversion during OCR

On macOS, Tesseract and Poppler can typically be installed with Homebrew. On Windows, the current OCR loader expects Tesseract and Poppler to be available from fixed local paths configured in [`app/loaders/pdf_loader.py`](/Users/manoharmallipudi/Projects/Multi-Lingual-RAG-ChatBot/app/loaders/pdf_loader.py).

## Configuration

Create a local `.env` file from `.env.example` and provide the required values:

```bash
cp .env.example .env
```

### Required environment variables

- `AWS_REGION`: AWS region for Bedrock access
- `AWS_PROFILE`: local AWS CLI profile to use

### Optional environment variables

- `BEDROCK_MODEL_ID`: overrides the default Claude model
- `LANGSMITH_API_KEY`: enables LangSmith tracing
- `LANGSMITH_PROJECT`: LangSmith project name
- `LANGCHAIN_TRACING_V2`: tracing toggle
- `LANGCHAIN_ENDPOINT`: tracing endpoint

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running the Application

### Streamlit interface

```bash
streamlit run streamlit_app.py
```

The Streamlit application supports:

- Uploading one or more PDF files
- Rebuilding the FAISS index from the sidebar
- Toggling OCR for scanned PDFs
- Inspecting retrieved context chunks during chat

### Command-line interface

```bash
python main.py
```

The CLI builds or loads the local vector index and starts an interactive chat session in the terminal.

## Data Flow

- Source PDFs are stored in `data/raw/`
- FAISS index data is stored in `data/index/faiss.index`
- Serialized document metadata is stored in `data/index/documents.pkl`

If indexed data already exists, the application loads it from disk. If not, it builds the index from the available PDF files.

## OCR Behavior

The loader first attempts direct text extraction. OCR is triggered when:

- OCR is explicitly forced from the UI, or
- a significant portion of pages are empty or appear to contain poor-quality extracted text

Supported OCR language defaults include:

- English
- Hindi
- Telugu
- Tamil
- Russian
- Ukrainian
- Portuguese
- Spanish
- French
- German
- Italian

## Language Handling

The response language is resolved from the user query in code before the final prompt is sent to the model. The application:

- respects explicit requests such as "answer in French" or "reply in Hindi"
- otherwise detects the dominant language of the user input
- prevents retrieved document language from overriding the chosen response language
- uses retrieved context as the only factual basis for the answer
- translates grounded information into the selected response language when needed

This separation is important in multilingual RAG because source documents and user questions may be in different languages. The system therefore treats:

- query language selection as an application-level decision
- context language as source material to retrieve and translate from when necessary

## Observability

When LangSmith is configured, the project records:

- LLM traces
- token usage
- estimated invocation cost metadata

Pricing defaults are maintained in [`app/pricing.py`](/Users/manoharmallipudi/Projects/Multi-Lingual-RAG-ChatBot/app/pricing.py).

## Limitations

- The current vector store is local-only and not intended for distributed production workloads
- OCR path configuration on Windows is hard-coded in the loader
- There are currently no automated tests in the repository
- Pricing values in `app/pricing.py` should be reviewed when models or Bedrock pricing change

## Recommended Next Improvements

- Add automated tests for ingestion, retrieval, and language selection
- Introduce structured logging instead of console debugging
- Externalize OCR path configuration for Windows environments
- Add deployment instructions for a hosted Streamlit environment
- Add sample documents and screenshots for easier onboarding

## Backstage Catalog

The repository includes a Backstage descriptor at [`catalog-info.yaml`](/Users/manoharmallipudi/Projects/Multi-Lingual-RAG-ChatBot/catalog-info.yaml) so the service can be registered directly from the source repository.

## License

No license file is currently included in the repository. Add one if the project is intended for broader distribution.
