# COMP5423 Group 14 — Skill-Augmented Search Agent

End-to-end skill-augmented multi-step search agent for BrowseComp-Plus (100,195 documents) using **DeepSeek-V4-Flash**.

## 1. Code Structure
```text
code/
├── config.py                  # Global configuration (reads env vars / optional local .env)
├── .env.example               # Template for local API configuration (contains no credentials)
├── requirements.txt           # Python 3.12.7 pinned dependencies
├── run_predict.py             # Batch runner producing test_prediction.jsonl
├── validate_submission.py     # Schema and completeness validator for test_prediction.jsonl
├── agent/
│   ├── types.py               # SearchHit, ToolResult, AgentState
│   ├── llm_client.py          # OpenAI-compatible API client with retry and token accounting
│   ├── context.py             # Structured state, deduplication, pruning, and compression
│   └── loop.py                # Iterative ReAct loop and agent assembly factory
├── retrieval/
│   ├── base.py                # Retriever and DocStore interfaces + factory
│   ├── doc_store.py           # Thread-safe SQLiteDocStore + query-aligned snippet builder
│   ├── bm25_retriever.py      # Category 1: Sparse retrieval (BM25S)
│   ├── dense_base.py          # Shared GPU FP16 sharded encoding + FAISS IndexFlatIP
│   ├── dense_retriever.py     # Category 2: Encoder dense retrieval (BAAI/bge-base-en-v1.5)
│   ├── llm_dense_retriever.py # Category 3: Instruction-aware LLM dense retrieval (Qwen3-Embedding-0.6B)
│   ├── fusion.py              # Weighted Reciprocal Rank Fusion (RRF) behind one Retriever
│   ├── build_indexes.py       # Index builder CLI for docstore, bm25, dense, and llm_dense
│   └── smoke_test.py          # Quick latency and hit-quality check over validation questions
├── tools/
│   ├── base.py                # Tool base class and ToolRegistry
│   ├── search.py              # The single unified retrieval tool (`search`)
│   ├── doc_tools.py           # Document inspection tools (`open_doc`, `find_in_doc`)
│   └── skill_tool.py          # Runtime Agent Skills loader (`load_skill`)
├── data/
│   └── loader.py              # Parquet corpus streaming and JSONL question loader
├── eval/
│   ├── metrics.py             # Normalized EM and efficiency aggregation
│   ├── run.py                 # Validation and ablation experiment runner
│   └── demo_skills.py         # Automatic skill-selection demonstration runner
├── skills/
│   ├── multi-clue-decomposition/  # Original skill + references/clue-priority.md
│   ├── entity-disambiguation/     # Original skill
│   ├── search-recovery/           # Original skill
│   └── validate_skills.py         # Agent Skills specification and runtime-loading validator
├── tests/                     # Unit tests for context control, normalized EM, and skills
└── ui/
    ├── web_app.py             # Streamlit Web UI
    └── app.py                 # Terminal CLI
```

## 2. Environment Setup
- Python 3.12+ (tested on Python 3.12.7 with CUDA 12.8 on NVIDIA RTX 3070).
- Create an isolated virtual environment and install dependencies:
  ```powershell
  uv venv ../.venv --python 3.12
  ..\.venv\Scripts\activate
  uv pip install --default-index https://download.pytorch.org/whl/cu128 torch==2.11.0+cu128
  uv pip install -r requirements.txt
  ```
- Place the BrowseComp-Plus corpus Parquet files in `data_files/corpus/` and question files (`validation.jsonl`, `test.jsonl`) in `data_files/`.
- Build the SQLite document store and all three retrieval category indexes over all 100,195 documents:
  ```powershell
  python retrieval/build_indexes.py --step all
  ```
  (Alternatively, download pre-built `index_files/` from our public read-only cloud link: `<ADD_PUBLIC_READONLY_CLOUD_LINK>`.)

## 3. Supplying the LLM API Key
Never commit or submit credentials. Provide your API key via environment variables (or copy `.env.example` to `../.env` locally):

**PowerShell (Windows):**
```powershell
$env:DEEPSEEK_API_KEY="your-api-key-here"
$env:LLM_BASE_URL="https://api.deepseek.com"
$env:LLM_MODEL="deepseek-v4-flash"
```

**Bash (Linux/macOS):**
```bash
export DEEPSEEK_API_KEY="your-api-key-here"
export LLM_BASE_URL="https://api.deepseek.com"
export LLM_MODEL="deepseek-v4-flash"
```

## 4. Reproducing `test_prediction.jsonl`
```powershell
python run_predict.py `
  --input data_files/test.jsonl `
  --output ../test_prediction.jsonl `
  --retriever fusion `
  --workers 4 `
  --resume

python validate_submission.py `
  --pred ../test_prediction.jsonl `
  --input data_files/test.jsonl `
  --expected 100
```

## 5. Validation & Ablation Experiments
```powershell
# Sparse BM25S only
python eval/run.py --input data_files/validation.jsonl --retriever bm25 --tag E1_bm25 --workers 4 --resume
# Encoder dense (BGE-base-en-v1.5) only
python eval/run.py --input data_files/validation.jsonl --retriever dense --tag E2_dense --workers 4 --resume
# LLM instruction-aware dense (Qwen3-Embedding-0.6B) only
python eval/run.py --input data_files/validation.jsonl --retriever llm_dense --tag E2b_llm_dense --workers 4 --resume
# 3-category RRF fusion without structured context management
python eval/run.py --input data_files/validation.jsonl --retriever fusion --no-context --tag E4_no_ctx --workers 4 --resume
# 3-category RRF fusion with context management, without skills
python eval/run.py --input data_files/validation.jsonl --retriever fusion --no-skills --tag E5_ctx_no_skills --workers 4 --resume
# 3-category RRF fusion with context management + forced single skill
python eval/run.py --input data_files/validation.jsonl --retriever fusion --force-skill multi-clue-decomposition --tag E6_force_skill --workers 4 --resume
# Full system: 3-category RRF fusion + structured context management + automatic skill selection
python eval/run.py --input data_files/validation.jsonl --retriever fusion --tag E7_full --workers 4 --resume
```

## 6. Interactive UI, Skill Validation & Unit Tests
```powershell
# Launch Streamlit Web UI
streamlit run ui/web_app.py

# Or launch terminal CLI
python ui/app.py --retriever fusion

# Validate Agent Skills specification and runtime-only loading contract
python skills/validate_skills.py

# Run unit tests
python -m unittest discover -s tests -p "test_*.py" -v

# Run automatic skill-selection showcase
python eval/demo_skills.py --retriever fusion --out eval_outputs/skill_demo.json
```
