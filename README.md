
# DPDP-Compliance-System

General-purpose multi-agent system for automated DPDP compliance assessment.
## Project Setup and Running the Assessment

### Requirements

- Python 3.10 or later
- Git
- Groq API key for real LLM-based assessment
- Official DPDP Act and DPDP Rules PDFs for legal RAG

### 1. Clone the repository

```powershell
git clone https://github.com/krithiika-s/DPDP-Compliance-System.git
cd DPDP-Compliance-System
```

### 2. Create and activate a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

### 4. Configure environment variables

Create a local `.env` file in the repository root.

Configure the following variables using your own Groq credentials and selected model:

```text
GROQ_API_KEY=your_groq_api_key
LLM_PROVIDER=groq
LLM_MODEL=your_supported_model
```

Never commit `.env` or expose API keys.

### 5. Run the assessment

**Demo mode (default):**

```powershell
python -m scripts.run_assessment
```

Uses the demo pipeline without real LLM API calls.

**Real Agents 1–3:**

```powershell
python -m scripts.run_assessment --real
```

Uses real LLM calls for policy extraction, implementation analysis and alignment.

**Full real pipeline:**

```powershell
python -m scripts.run_assessment --full-real
```

Runs the complete pipeline, including legal reasoning, evidence gates, deterministic scoring and report generation. This mode makes real API calls and may incur usage charges.

Use `--project` and `--policy` to provide custom input paths.

**Interactive Streamlit assessment app:**

Create `.streamlit/secrets.toml` locally with:

```toml
APP_PASSWORD = "your_private_password"
GROQ_API_KEY = "your_groq_api_key"
LLM_MODEL = "your_supported_model"
```

Launch the interactive app from the repository root:

```powershell
streamlit run streamlit_app.py
```

### 6. Run tests

```powershell
python -m pytest -q
```

The automated tests use mocks and local test fixtures where appropriate. They do not require live Groq API calls.

### Important limitations

This project is a research prototype, not a legal certification tool. Assessment findings require human review. A complete successful live end-to-end assessment has not yet been established.
## Legal RAG Setup (Person 3)

The Legal RAG component retrieves relevant provisions from the official
Digital Personal Data Protection Act, 2023 and DPDP Rules, 2025.

### 1. Install dependencies

From the repository root, run:

```powershell
python -m pip install -r requirements.txt
```

### 2. Check legal documents

Place the following official PDF documents inside:

`knowledge_base/legal_docs/`

- `2bf1f0e9f04e6fb4f8fef35e82c42aa5.pdf` - DPDP Act, 2023
- `DPDP_Rules_2025_English_only (1).pdf` - DPDP Rules, 2025

### 3. Build the legal vector index

From the repository root, run:

```powershell
python -m agents.legal_rag.ingest
```

This command:
- Extracts text from the official legal PDFs.
- Splits the text into smaller chunks.
- Generates embeddings using `sentence-transformers/all-MiniLM-L6-v2`.
- Stores the embeddings in a ChromaDB collection named `dpdp_legal_sources`.

The generated database is stored at:

`knowledge_base/chroma_db/`

The first run may download the Hugging Face embedding model if it is not already cached.

### 4. Git tracking

The generated ChromaDB directory is excluded from Git through `.gitignore`:

`knowledge_base/chroma_db/`

Each developer must build the local vector index before using the real legal retriever. The generated database is not committed to the repository.

### 5. Legal retrieval behavior and limitations

The retriever:
- Uses similarity search against primary-authority legal sources.
- Retrieves up to five nearest passages by default.
- Applies a provisional maximum relevance-distance threshold of `1.0`.
- Discards results whose distance exceeds the configured threshold.

The threshold has not yet been fully calibrated. Retrieval may miss exact legal provisions or rank semantically related but incorrect provisions above the correct provision.

Extraction and structural validation of the two source documents were confirmed:
- DPDP Act, 2023: 21 pages and 101 chunks.
- DPDP Rules, 2025: 18 pages and 93 chunks.
- Total: 39 pages and 194 chunks.

These indexing results were confirmed in the development environment. Each developer must build their own local index.

The system is a research prototype. Legal retrieval and provision-level accuracy require further evaluation before relying on the results for legal decisions.
