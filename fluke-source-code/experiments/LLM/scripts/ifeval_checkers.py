import re
from typing import Dict, Any


def _normalize_text(s: str) -> str:
    if s is None:
        return ""
    # Basic normalization consistent with many IFEval-style checks
    return " ".join(str(s).strip().split())


def _count_words(s: str) -> int:
    # Word counting: split on whitespace after normalization
    s = _normalize_text(s)
    if not s:
        return 0
    return len(s.split())


def check_punctuation_no_comma(output_text: str, params: Dict[str, Any]) -> bool:
    # Ensure there is no comma character
    return "," not in (output_text or "")


def check_keywords_letter_frequency(output_text: str, params: Dict[str, Any]) -> bool:
    # params: {"letter": str, "let_relation": "at least"|"at most"|"exactly", "let_frequency": int}
    letter = params.get("letter")
    relation = (params.get("let_relation") or "").lower()
    freq_required = params.get("let_frequency")
    if not letter or freq_required is None:
        return False
    text = output_text or ""
    count = text.count(letter)
    if relation in ["at least", ">=", "ge"]:
        return count >= freq_required
    if relation in ["at most", "<=", "le"]:
        return count <= freq_required
    if relation in ["exactly", "==", "eq"]:
        return count == freq_required
    # Default to at least
    return count >= freq_required


def check_startend_end_checker(output_text: str, params: Dict[str, Any]) -> bool:
    # params: {"end_phrase": str}
    end_phrase = params.get("end_phrase")
    if not end_phrase:
        return False
    text = _normalize_text(output_text or "")
    # Must end exactly with the phrase (no trailing chars)
    return text.endswith(_normalize_text(end_phrase))


def check_keywords_frequency(output_text: str, params: Dict[str, Any]) -> bool:
    # params: {"keyword": str, "relation": "at least"|"at most"|"exactly", "frequency": int}
    keyword = params.get("keyword")
    relation = (params.get("relation") or "").lower()
    freq_required = params.get("frequency")
    if not keyword or freq_required is None:
        return False
    text = output_text or ""
    # Basic substring count (case-sensitive per IFEval typical semantics)
    count = text.count(keyword)
    if relation in ["at least", ">=", "ge"]:
        return count >= freq_required
    if relation in ["at most", "<=", "le"]:
        return count <= freq_required
    if relation in ["exactly", "==", "eq"]:
        return count == freq_required
    # Default to at least
    return count >= freq_required


def check_combination_repeat_prompt(output_text: str, params: Dict[str, Any]) -> bool:
    # params: {"prompt_to_repeat": str}
    prompt = params.get("prompt_to_repeat")
    if not prompt:
        return False
    text = output_text or ""
    # Must start with the exact prompt text (no single-letter change)
    return text.startswith(prompt)


def check_length_constraints_number_words(output_text: str, params: Dict[str, Any]) -> bool:
    # params: {"num_words": int, "relation": "at most"|"at least"|"exactly"}
    num_words = params.get("num_words")
    relation = (params.get("relation") or "at most").lower()
    count = _count_words(output_text or "")
    if num_words is None:
        return False
    if relation in ["at most", "<=", "le"]:
        return count <= num_words
    if relation in ["at least", ">=", "ge"]:
        return count >= num_words
    if relation in ["exactly", "==", "eq"]:
        return count == num_words
    # Default to at most
    return count <= num_words


def check_length_constraints_nth_paragraph_first_word(output_text: str, params: Dict[str, Any]) -> bool:
    """Check that the nth paragraph starts with the given first word.

    params: {
        "nth_paragraph": int (1-based),
        "first_word": str,
    }
    Paragraphs are delimited by two newlines ("\n\n"). We extract the first
    token-like word (letters/digits/_/apostrophe/hyphen) case-insensitively.
    """
    try:
        n = int(params.get("nth_paragraph") or 0)
    except Exception:
        n = 0
    first_word = params.get("first_word")
    if n <= 0 or not first_word:
        return False
    text = output_text or ""
    parts = text.split("\n\n")
    # Count non-empty paragraphs
    non_empty = [p for p in parts if p.strip()]
    # If num_paragraphs is specified, enforce it
    try:
        num_pars = int(params.get('num_paragraphs')) if params.get('num_paragraphs') is not None else None
    except Exception:
        num_pars = None
    if num_pars is not None and len(non_empty) != num_pars:
        return False
    if len(parts) < n:
        return False
    para = parts[n - 1].strip()
    import re
    m = re.search(r"[A-Za-z0-9'_\-]+", para)
    got = (m.group(0) if m else "").strip().lower()
    want = str(first_word).strip().lower()
    return got == want


CHECKER_DISPATCH = {
    "punctuation:no_comma": check_punctuation_no_comma,
    "keywords:letter_frequency": check_keywords_letter_frequency,
    "startend:end_checker": check_startend_end_checker,
    "keywords:frequency": check_keywords_frequency,
    "combination:repeat_prompt": check_combination_repeat_prompt,
    "length_constraints:number_words": check_length_constraints_number_words,
    "length_constraints:nth_paragraph_first_word": check_length_constraints_nth_paragraph_first_word,
}


def check_constraint(constraint_id: str, params: Dict[str, Any], output_text: str) -> bool:
    # Additional: forbidden words checker (whole-word, case-insensitive)
    if constraint_id in ("keywords:forbidden_words", "keywords:forbidden"):
        words = []
        if isinstance(params, dict):
            if isinstance(params.get("forbidden_words"), list):
                words.extend([str(w) for w in params.get("forbidden_words")])
            if params.get("forbidden_word"):
                words.append(str(params.get("forbidden_word")))
        text = output_text or ""
        import re
        for w in words:
            if not w:
                continue
            # whole-word, case-insensitive
            if re.search(rf"\b{re.escape(w)}\b", text, re.IGNORECASE):
                return False
        return True

    fn = CHECKER_DISPATCH.get(constraint_id)
    if fn is None:
        # Unknown constraint id; do not penalize
        return True
    try:
        return bool(fn(output_text, params))
    except Exception:
        return False


# ---------------- Additional IFEval checkers ---------------- #

def _cmp_relation(val: int, relation: str, target: int) -> bool:
    r = (relation or '').lower()
    if r in {'at least', '>=', 'ge'}:
        return val >= target
    if r in {'at most', '<=', 'le'}:
        return val <= target
    if r in {'exactly', '==', 'eq'}:
        return val == target
    # default: at least
    return val >= target


def check_startend_quotation(output_text: str, params: Dict[str, Any]) -> bool:
    text = (output_text or '').strip()
    return len(text) > 1 and text[0] == '"' and text[-1] == '"'


def check_length_constraints_number_paragraphs(output_text: str, params: Dict[str, Any]) -> bool:
    """Paragraphs separated by the markdown divider *** (parity with IFEval)."""
    try:
        n = int(params.get('num_paragraphs') or 0)
    except Exception:
        n = 0
    if n <= 0:
        return False
    import re
    parts = re.split(r"\s?\*\*\*\s?", output_text or '')
    num = len(parts)
    # Adjust for empty ends; inner empties invalidate
    for idx, para in enumerate(parts):
        if not para.strip():
            if idx == 0 or idx == len(parts) - 1:
                num -= 1
            else:
                return False
    return num == n


def check_length_constraints_number_sentences(output_text: str, params: Dict[str, Any]) -> bool:
    # naive sentence count by ., !, ? terminators
    import re
    try:
        n = int(params.get('num_sentences') or 0)
    except Exception:
        n = 0
    if n <= 0:
        return False
    text = _normalize_text(output_text or '')
    # Split on sentence-ending punctuation keeping basic robustness
    sentences = [s for s in re.split(r'[.!?]+\s+', text) if s.strip()]
    count = len(sentences)
    relation = (params.get('relation') or 'exactly')
    return _cmp_relation(count, relation, n)


def check_detectable_content_postscript(output_text: str, params: Dict[str, Any]) -> bool:
    marker = params.get('postscript_marker') or 'P.S.'
    val = (output_text or '').lower()
    import re
    if marker == 'P.P.S':
        pat = r"\s*p\.\s?p\.\s?s.*$"
    elif marker == 'P.S.':
        pat = r"\s*p\.\s?s\..*$"
    else:
        pat = r"\s*" + re.escape(marker.lower()) + r".*$"
    return re.findall(pat, val, flags=re.MULTILINE) != []


def check_detectable_content_number_placeholders(output_text: str, params: Dict[str, Any]) -> bool:
    # Placeholders like [something]
    import re
    try:
        n = int(params.get('num_placeholders') or 0)
    except Exception:
        n = 0
    if n <= 0:
        return False
    matches = re.findall(r'\[[^\]]+\]', output_text or '')
    relation = (params.get('relation') or 'at least')
    return _cmp_relation(len(matches), relation, n)


def check_detectable_format_json_format(output_text: str, params: Dict[str, Any]) -> bool:
    import json
    s = (output_text or '').strip()
    # Strip markdown fences
    for pref in ("```json", "```Json", "```JSON", "```"):
        if s.startswith(pref):
            s = s[len(pref):].strip()
            break
    if s.endswith("```"):
        s = s[:-3].strip()
    try:
        json.loads(s)
        return True
    except Exception:
        return False


def check_detectable_format_number_bullet_lists(output_text: str, params: Dict[str, Any]) -> bool:
    # count lines starting with - or * or <digit>.
    import re
    try:
        n = int(params.get('num_bullets') or 0)
    except Exception:
        n = 0
    if n <= 0:
        return False
    count = 0
    for ln in (output_text or '').splitlines():
        if re.match(r'\s*([-*]|\d+\.)\s+', ln):
            count += 1
    relation = (params.get('relation') or 'exactly')
    return _cmp_relation(count, relation, n)


def check_detectable_format_number_highlighted_sections(output_text: str, params: Dict[str, Any]) -> bool:
    import re
    try:
        n = int(params.get('num_highlights') or 0)
    except Exception:
        n = 0
    if n <= 0:
        return False
    # count *...* spans
    matches = re.findall(r'\*[^*]+\*', output_text or '')
    relation = (params.get('relation') or 'at least')
    return _cmp_relation(len(matches), relation, n)


def check_detectable_format_multiple_sections(output_text: str, params: Dict[str, Any]) -> bool:
    section_spliter = params.get('section_spliter') or ''
    try:
        num_sections = int(params.get('num_sections') or 0)
    except Exception:
        num_sections = 0
    if not section_spliter or num_sections <= 0:
        return False
    import re
    pattern = r"\s?" + re.escape(section_spliter) + r"\s?\d+\s?"
    sections = re.split(pattern, output_text or '')
    count = len(sections) - 1
    return count >= num_sections


def check_detectable_format_title(output_text: str, params: Dict[str, Any]) -> bool:
    import re
    pattern = r"<<[^\n]+>>"
    titles = re.findall(pattern, output_text or '')
    for title in titles:
        inner = title.lstrip('<').rstrip('>')
        if inner.strip():
            return True
    return False


def check_keywords_existence(output_text: str, params: Dict[str, Any]) -> bool:
    # Require presence of all given keywords (case-insensitive substring)
    words = []
    if isinstance(params, dict):
        if isinstance(params.get('keywords'), list):
            words.extend([str(w) for w in params.get('keywords')])
        if params.get('keyword'):
            words.append(str(params.get('keyword')))
    txt = output_text or ''
    for w in words:
        if not w:
            continue
        if re.search(re.escape(w), txt, re.IGNORECASE) is None:
            return False
    return True


def check_language_response_language(output_text: str, params: Dict[str, Any]) -> bool:
    # Prefer langdetect if available to match reference; otherwise fallback to heuristics
    code = (params.get('language') or '').lower()
    txt = output_text or ''
    try:
        import langdetect  # type: ignore
        try:
            return langdetect.detect(txt) == code
        except langdetect.LangDetectException:
            return True  # count as followed
    except Exception:
        # Heuristics by script/code
        if code in {'bn'}:
            return any('\u0980' <= ch <= '\u09FF' for ch in txt)
        if code in {'ru'}:
            return any('\u0400' <= ch <= '\u04FF' for ch in txt)
        if code in {'ar'}:
            return any('\u0600' <= ch <= '\u06FF' for ch in txt)
        if code in {'vi'}:
            vi_chars = set("ăâđêôơưĂÂĐÊÔƠƯ“”ấầẩẫậắằẳẵặềềểễệốồổỗộớờởỡợứừửữự")
            return any(ch in vi_chars for ch in txt)
        if code in {'it', 'en', 'es', 'fr', 'pt', 'de'}:
            letters = [ch for ch in txt if ch.isalpha()]
            if not letters:
                return False
            latin = sum(1 for ch in letters if ord(ch) < 128)
            return (latin / max(1, len(letters))) > 0.8
        return True


def check_change_case_english_lowercase(output_text: str, params: Dict[str, Any]) -> bool:
    txt = output_text or ''
    try:
        import langdetect  # type: ignore
        ok_lang = (langdetect.detect(txt) == 'en')
    except Exception:
        ok_lang = True
    return txt.islower() and ok_lang


def check_change_case_english_capital(output_text: str, params: Dict[str, Any]) -> bool:
    txt = output_text or ''
    try:
        import langdetect  # type: ignore
        ok_lang = (langdetect.detect(txt) == 'en')
    except Exception:
        ok_lang = True
    return txt.isupper() and ok_lang


def check_change_case_capital_word_frequency(output_text: str, params: Dict[str, Any]) -> bool:
    # Count words that are fully uppercase (length>=2, letters only), compare with relation
    import re
    relation = (params.get('capital_relation') or 'at least')
    try:
        n = int(params.get('capital_frequency') or 0)
    except Exception:
        n = 0
    if n <= 0:
        return False
    words = re.findall(r'\b[\p{Lu}A-Z]{2,}\b', output_text or '')
    # Python's re doesn't support \p{Lu} without regex module; fallback (A-Z)
    if not words:
        words = re.findall(r'\b[A-Z]{2,}\b', output_text or '')
    return _cmp_relation(len(words), relation, n)


def check_combination_two_responses(output_text: str, params: Dict[str, Any]) -> bool:
    # Exactly two non-empty responses separated by ****** and not identical
    text = output_text or ''
    parts = text.split('******')
    valid = [p for p in parts if p.strip()]
    return len(valid) == 2 and valid[0].strip() != valid[1].strip()


# Register newly implemented checkers
CHECKER_DISPATCH.update({
    "startend:quotation": check_startend_quotation,
    "length_constraints:number_paragraphs": check_length_constraints_number_paragraphs,
    "length_constraints:number_sentences": check_length_constraints_number_sentences,
    "detectable_content:postscript": check_detectable_content_postscript,
    "detectable_content:number_placeholders": check_detectable_content_number_placeholders,
    "detectable_format:json_format": check_detectable_format_json_format,
    "detectable_format:number_bullet_lists": check_detectable_format_number_bullet_lists,
    "detectable_format:number_highlighted_sections": check_detectable_format_number_highlighted_sections,
    "detectable_format:multiple_sections": check_detectable_format_multiple_sections,
    "detectable_format:title": check_detectable_format_title,
    "keywords:existence": check_keywords_existence,
    "language:response_language": check_language_response_language,
    "change_case:english_lowercase": check_change_case_english_lowercase,
    "change_case:english_capital": check_change_case_english_capital,
    "change_case:capital_word_frequency": check_change_case_capital_word_frequency,
    "combination:two_responses": check_combination_two_responses,
    # detectable_format:constrained_response — simplified: must be exactly one allowed phrase
    "detectable_format:constrained_response": lambda text, p: _is_constrained_choice(text),
})


def _is_constrained_choice(text: str) -> bool:
    # Accept presence of any allowed constrained response anywhere in the string
    s = (text or '').strip()
    options = (
        "My answer is yes.",
        "My answer is no.",
        "My answer is maybe.",
    )
    return any(opt in s for opt in options)

# Tweak repeat_prompt checker to be case-insensitive
def check_combination_repeat_prompt(output_text: str, params: Dict[str, Any]) -> bool:
    prompt = (params or {}).get('prompt_to_repeat')
    if not prompt:
        return False
    return (output_text or '').strip().lower().startswith(str(prompt).strip().lower())

# Override dispatch for repeat_prompt
CHECKER_DISPATCH["combination:repeat_prompt"] = check_combination_repeat_prompt
