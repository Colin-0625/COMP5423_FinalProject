"""Global configuration loaded from environment variables (with optional local .env fallback)."""
import os


def _load_local_dotenv() -> None:
    """Load key=value pairs from ../.env or ./.env without overriding existing environment variables."""
    here = os.path.dirname(os.path.abspath(__file__))
    for candidate in (os.path.join(os.path.dirname(here), ".env"), os.path.join(here, ".env")):
        if not os.path.isfile(candidate):
            continue
        with open(candidate, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith("#") or "=" not in s:
                    continue
                k, v = s.split("=", 1)
                k = k.strip()
                v = v.strip().strip('"').strip("'")
                if k and k not in os.environ:
                    os.environ[k] = v


_load_local_dotenv()

# LLM backbone configuration (OpenAI-compatible endpoint serving DeepSeek-V4-Flash)
LLM_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "https://api.deepseek.com")
LLM_MODEL = os.environ.get("LLM_MODEL", "deepseek-v4-flash")
LLM_TEMPERATURE = float(os.environ.get("LLM_TEMPERATURE", "0.0"))
LLM_MAX_TOKENS = int(os.environ.get("LLM_MAX_TOKENS", "6144"))
LLM_TIMEOUT = int(os.environ.get("LLM_TIMEOUT", "120"))
LLM_RETRIES = int(os.environ.get("LLM_RETRIES", "5"))

# Agent and tool defaults
MAX_STEPS = int(os.environ.get("MAX_STEPS", "12"))
SEARCH_TOP_K = int(os.environ.get("SEARCH_TOP_K", "5"))
SNIPPET_CHARS = 300
OPEN_DOC_CHARS = 3000

# Context management parameters
CTX_RECENT_TURNS = int(os.environ.get("CTX_RECENT_TURNS", "3"))
CTX_MAX_EVIDENCE = int(os.environ.get("CTX_MAX_EVIDENCE", "8"))
CTX_MAX_CHARS = int(os.environ.get("CTX_MAX_CHARS", "12000"))
CTX_ENABLE_LLM_SUMMARY = os.environ.get("CTX_ENABLE_LLM_SUMMARY", "0").lower() in ("1", "true", "yes")

# Data, index, and skill paths
DATA_DIR = os.environ.get("DATA_DIR", os.path.join(os.path.dirname(__file__), "data_files"))
INDEX_DIR = os.environ.get("INDEX_DIR", os.path.join(os.path.dirname(__file__), "index_files"))
SKILLS_DIR = os.path.join(os.path.dirname(__file__), "skills")
