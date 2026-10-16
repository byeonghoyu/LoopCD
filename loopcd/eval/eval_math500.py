from __future__ import annotations
from typing import Any


STOPS: list[str] = ["Problem:"]


_FEWSHOT_PREFIX: str | None = None


def _doc_to_text(problem: str) -> str:
    return "Problem:" + "\n" + problem + "\n\n" + "Solution:"


def _load_minerva_utils():
    import importlib.util as _il
    from pathlib import Path as _Path
    import lm_eval as _le
    utils_path = _Path(_le.__file__).parent / "tasks" / "minerva_math" / "utils.py"
    spec = _il.spec_from_file_location("_minerva_math_utils", utils_path)
    m = _il.module_from_spec(spec); spec.loader.exec_module(m)
    return m


_MU = None
def _mu():
    global _MU
    if _MU is None:
        _MU = _load_minerva_utils()
    return _MU


def _ensure_fewshot_prefix() -> str:
    global _FEWSHOT_PREFIX
    if _FEWSHOT_PREFIX is None:
        parts = []
        for fs in _mu().list_fewshot_samples():
            parts.append(_doc_to_text(fs["problem"]) + "\n" + fs["solution"] + "\n\n")
        _FEWSHOT_PREFIX = "".join(parts)
    return _FEWSHOT_PREFIX


def fmt_prompt(question: str, tokenizer=None) -> str:
    return _ensure_fewshot_prefix() + _doc_to_text(question)


def setup(tokenizer) -> list[str]:
    return STOPS


def truncate_at_stops(comp: str, stops=None) -> str:
    out = comp
    for s in (stops or STOPS):
        if s in out:
            out = out.split(s)[0]
    return out


def score(text: str, doc_or_gold) -> dict[str, Any]:
    mu = _mu()

    if isinstance(doc_or_gold, dict):
        solution = doc_or_gold.get("_solution") or doc_or_gold.get("gold")
    else:
        solution = doc_or_gold

    boxed = mu.last_boxed_only_string(solution)
    if boxed is None:
        answer = ""
    else:
        try:
            answer = mu.normalize_final_answer(mu.remove_boxed(boxed))
        except Exception:
            answer = ""
    doc = {"problem": "", "solution": solution, "answer": answer}
    res = mu.process_results(doc, [text])
    return {
        "strict_ok": bool(res.get("exact_match", 0)),
        "flex_ok":   bool(res.get("math_verify", 0)),
        "strict_pred": "",
        "flex_pred":   "",
        "gold_extracted": answer,
    }


def load_docs(limit: int | None = None) -> list[dict]:
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    out = []
    n = len(ds)
    for i in range(n):
        if limit is not None and i >= limit:
            break
        sol = ds[i]["solution"]
        out.append({
            "doc_id": i,
            "question": ds[i]["problem"],
            "gold": sol,
            "_solution": sol,
        })
    return out
