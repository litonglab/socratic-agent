from __future__ import annotations

import json
import sys
import threading
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from agentic_rag import rag, topo_rag

_STDOUT_LOCK = threading.Lock()
_DEFAULT_MAX_SOURCES = 6
_MAX_SOURCES_LIMIT = 12


def _run_with_redirected_stdout(func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Redirect stdout to stderr to keep stdio MCP responses clean."""
    with _STDOUT_LOCK:
        with redirect_stdout(sys.stderr):
            return func(*args, **kwargs)


def _normalize_category(category: Optional[str]) -> Optional[str]:
    normalized = (category or "").strip().upper()
    return normalized or None


def _coerce_non_negative_int(value: Any, default: int) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return default


def _coerce_max_sources(value: Any) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = _DEFAULT_MAX_SOURCES
    return min(max(1, parsed), _MAX_SOURCES_LIMIT)


def retrieve_course_docs(
    query: str,
    category: Optional[str] = None,
    hint_level: int = 0,
    max_sources: int = _DEFAULT_MAX_SOURCES,
) -> Dict[str, Any]:
    query = (query or "").strip()
    if not query:
        return {"ok": False, "error": "query is required"}

    normalized_category = _normalize_category(category)
    normalized_hint_level = _coerce_non_negative_int(hint_level, default=0)
    normalized_max_sources = _coerce_max_sources(max_sources)

    return _run_with_redirected_stdout(
        rag.retrieve_course_docs,
        query=query,
        category=normalized_category,
        hint_level=normalized_hint_level,
        max_sources=normalized_max_sources,
    )


def get_topology_context(
    query: str,
    experiment_id: Optional[str] = None,
) -> Dict[str, Any]:
    normalized_query = (query or "").strip()
    normalized_experiment_id = (experiment_id or "").strip() or None
    if not normalized_query and not normalized_experiment_id:
        return {"ok": False, "error": "query or experiment_id is required"}

    try:
        retriever = topo_rag.LoadTopoRetriever(
            experiment_id=normalized_experiment_id,
            store_root=topo_rag.TOPO_STORE_ROOT,
        )
        result = retriever(normalized_query or normalized_experiment_id or "")
        resolved_experiment_id = result.get("experiment_id")
        return {
            "ok": bool(result.get("context")),
            "query": normalized_query,
            "experiment_id": resolved_experiment_id,
            "experiment_label": (
                topo_rag.experiment_label_from_id(resolved_experiment_id)
                if resolved_experiment_id
                else None
            ),
            "topo_ids": result.get("topo_ids", []),
            "context": result.get("context", ""),
            "warning": result.get("warning", ""),
        }
    except Exception as exc:
        return {
            "ok": False,
            "query": normalized_query,
            "experiment_id": normalized_experiment_id,
            "error": f"get_topology_context failed: {exc}",
        }


def list_available_experiments() -> Dict[str, Any]:
    try:
        experiment_ids = topo_rag._list_available_experiment_ids(topo_rag.TOPO_STORE_ROOT)
        experiments = [
            {
                "experiment_id": experiment_id,
                "experiment_label": topo_rag.experiment_label_from_id(experiment_id),
            }
            for experiment_id in experiment_ids
        ]
        return {
            "ok": True,
            "count": len(experiments),
            "experiments": experiments,
        }
    except Exception as exc:
        return {
            "ok": False,
            "count": 0,
            "experiments": [],
            "error": f"list_available_experiments failed: {exc}",
        }


def _load_manifest_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def get_experiment_manifest(experiment_id: str) -> Dict[str, Any]:
    raw_experiment_id = (experiment_id or "").strip()
    if not raw_experiment_id:
        return {"ok": False, "error": "experiment_id is required"}

    try:
        normalized_experiment_id = topo_rag.normalize_experiment_id(raw_experiment_id)
    except ValueError as exc:
        return {"ok": False, "experiment_id": raw_experiment_id, "error": str(exc)}

    experiment_dir = topo_rag._resolve_experiment_dir_for_runtime(
        topo_rag.TOPO_STORE_ROOT,
        normalized_experiment_id,
    )
    manifest_path = experiment_dir / "manifest.json"
    approved_dir = experiment_dir / "approved_json"
    topo_ids = (
        sorted(path.stem for path in approved_dir.glob("*.json"))
        if approved_dir.exists()
        else []
    )

    if manifest_path.exists():
        try:
            manifest = _load_manifest_json(manifest_path)
        except Exception as exc:
            return {
                "ok": False,
                "experiment_id": normalized_experiment_id,
                "experiment_label": topo_rag.experiment_label_from_id(normalized_experiment_id),
                "store_dir": str(experiment_dir),
                "manifest_path": str(manifest_path),
                "error": f"failed to parse manifest.json: {exc}",
            }

        return {
            "ok": True,
            "experiment_id": manifest.get("experiment_id", normalized_experiment_id),
            "experiment_label": manifest.get(
                "experiment_label",
                topo_rag.experiment_label_from_id(normalized_experiment_id),
            ),
            "store_dir": str(experiment_dir),
            "manifest_path": str(manifest_path),
            "approved_topology_count": len(topo_ids),
            "topo_ids": topo_ids,
            "manifest": manifest,
        }

    if approved_dir.exists():
        return {
            "ok": True,
            "experiment_id": normalized_experiment_id,
            "experiment_label": topo_rag.experiment_label_from_id(normalized_experiment_id),
            "store_dir": str(experiment_dir),
            "manifest_path": None,
            "approved_topology_count": len(topo_ids),
            "topo_ids": topo_ids,
            "manifest": None,
            "warning": "manifest.json not found; returned a summary derived from approved_json.",
        }

    return {
        "ok": False,
        "experiment_id": normalized_experiment_id,
        "experiment_label": topo_rag.experiment_label_from_id(normalized_experiment_id),
        "store_dir": str(experiment_dir),
        "manifest_path": None,
        "error": (
            f"experiment data not found under {topo_rag.TOPO_STORE_ROOT}/{normalized_experiment_id}"
        ),
    }
