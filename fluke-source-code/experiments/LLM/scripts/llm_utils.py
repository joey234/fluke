import os
import re
import hashlib
from pathlib import Path
from typing import List, Optional

_CACHE: dict[str, str] = {}

def _hash_texts(texts: List[str]) -> str:
    h = hashlib.sha256()
    for t in texts:
        if t:
            h.update(t.encode('utf-8', errors='ignore'))
            h.update(b'\x1f')
    return h.hexdigest()

def _postprocess_number(s: str) -> Optional[str]:
    if not s:
        return None
    s = s.strip()
    # Keep only the first line in case the model added notes
    s = s.splitlines()[0].strip()
    # Remove currency symbols and commas
    s = re.sub(r'^[$€£¥₹₽]\s*', '', s)
    s = s.replace(',', '')
    # Extract a number or simple fraction
    m = re.search(r'[+-]?(?:\d+(?:\.\d+)?(?:[eE][+-]?\d+)?|\d+\s*/\s*\d+)', s)
    return m.group(0) if m else None

def llm_extract_number_from_texts(texts: List[str], model: Optional[str] = None, temperature: float = 0.0, timeout: int = 30) -> Optional[str]:
    """Use an OpenAI LLM to extract a final numeric answer from provided texts.

    - Only runs if env var USE_LLM_GSM_PARSE is truthy and OPENAI_API_KEY is set.
    - Defaults to model 'gpt-4.1'.
    - Returns a numeric string (e.g., '1596' or '3/4') or None.
    """
    if not os.environ.get('USE_LLM_GSM_PARSE'):
        return None
    if 'OPENAI_API_KEY' not in os.environ:
        return None
    model = model or os.environ.get('LLM_PREDICT_MODEL', 'gpt-4.1')
    payload = '\n\n'.join([t for t in texts if isinstance(t, str) and t.strip()])
    if not payload.strip():
        return None
    key = _hash_texts([model, payload])
    if key in _CACHE:
        return _CACHE[key]

    prompt = (
        "You will be given an assistant's reasoning and/or output for a GSM-style math question. "
        "Extract the assistant's final numeric answer only. Output NOTHING except the number. "
        "If multiple numbers appear, choose the one the assistant explicitly declares as the final answer (often after ####). "
        "Allowed formats: integer, decimal, scientific notation, or a simple fraction like 3/4."
    )

    # Try Responses API first, then ChatCompletions as a fallback.
    try:
        try:
            from openai import OpenAI  # type: ignore
            client = OpenAI()
            resp = client.responses.create(
                model=model,
                input=[
                    {"role":"system","content":prompt},
                    {"role":"user","content":payload},
                ],
                temperature=temperature,
                max_output_tokens=10,
            )
            content = resp.output_text if hasattr(resp, 'output_text') else None
        except Exception:
            import openai  # type: ignore
            openai.api_key = os.environ['OPENAI_API_KEY']
            chat = openai.ChatCompletion.create(
                model=model,
                messages=[
                    {"role":"system","content":prompt},
                    {"role":"user","content":payload},
                ],
                temperature=temperature,
                max_tokens=10,
            )
            content = chat.choices[0].message.get('content') if chat.choices else None
        val = _postprocess_number(content or '')
        if val:
            _CACHE[key] = val
            return val
    except Exception:
        return None
    return None

# --- .env loading helpers ---

def _load_dotenv_from(start: Path) -> None:
    """Load .env from start or its parents into os.environ if present."""
    cur = start
    for _ in range(10):  # climb up to 10 levels to avoid runaway
        envp = cur / '.env'
        if envp.exists():
            try:
                for line in envp.read_text(encoding='utf-8').splitlines():
                    s = line.strip()
                    if not s or s.startswith('#'):
                        continue
                    if '=' in s:
                        k, v = s.split('=', 1)
                        k = k.strip()
                        v = v.strip().strip('"\'')
                        if k and (k not in os.environ):
                            os.environ[k] = v
            except Exception:
                pass
            break
        if cur.parent == cur:
            break
        cur = cur.parent

def load_dotenv_if_present() -> None:
    try:
        _load_dotenv_from(Path(__file__).resolve().parent)
    except Exception:
        pass
    # Default to enabling LLM GSM parse unless explicitly disabled
    os.environ.setdefault('USE_LLM_GSM_PARSE', '1')

# Load .env on import by default
load_dotenv_if_present()
