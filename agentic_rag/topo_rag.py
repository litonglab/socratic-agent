from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, TypeVar

from tqdm import tqdm

from .topo_models import (
    TopologyBuildManifest,
    TopologyDraftArtifact,
    TopologyExtraction,
    TopologyImageClassificationArtifact,
    TopologyImageClassificationDecision,
    TopologyManifestItem,
    TopologyReviewArtifact,
    TopologyReviewDecision,
)


SUPPORTED_IMAGE_SUFFIX = {".png", ".jpg", ".jpeg", ".webp"}
TOPO_STORE_ROOT = os.getenv("TOPO_STORE_ROOT", "topo_store")
TOPO_DEFAULT_EXPERIMENT_ID = os.getenv("TOPO_DEFAULT_EXPERIMENT_ID", "")
TOPO_PERSIST_DIR = os.getenv("TOPO_PERSIST_DIR", "")

_EXPERIMENT_RE = re.compile(r"(?:实验\s*|lab[\s_-]?)(\d+)", re.IGNORECASE)
_PHY_RE = re.compile(r"^(?:E|GE|G|Gi|Fa|Eth)\d+/\d+/\d+$", re.IGNORECASE)
_SVI_RE = re.compile(r"^(?:Vlanif|Vlan)\d+$", re.IGNORECASE)
_PC_RE = re.compile(r"^PC", re.IGNORECASE)
_APPROVED_STATUSES = {"approved", "approved_with_warnings"}
_GLOBAL_APPROVED_TOPO_REGISTRY: Dict[str, Dict[str, dict]] = {}

SYSTEM_TOPO = """你是计算机网络实验拓扑图解析器。
任务：从给定拓扑图中提取设备、接口、链路、IP/掩码/VLAN/子网等信息，并输出为严格匹配 schema 的 JSON。
要求：
1) 不要臆造：看不清就填 null/留空，并在 warnings 说明不确定性来源。
2) 设备命名尽量沿用图中标注（如 R1/SW1/PC1）。
3) 接口名/端口号看不清可留空；但链路“设备-设备”关系尽量保留。
4) 只输出结构化内容，不输出额外解释性文本。
5) 若图中字段包含问号'?'或任意不确定字符：必须原样写入 *_raw 字段，并将规范字段设为 null。
6) 严禁将问号解释为任何数字，严禁基于经验补全缺失数字。
7) 物理接口如 E1/0/1、GE0/0/1 填 kind=physical；Vlan2/Vlanif2 填 kind=svi；PC 端若无接口名填 kind=host_nic。
"""

USER_TOPO = "请解析这张网络拓扑图，输出设备、接口、链路、IP/掩码/VLAN/子网信息。"

SYSTEM_TOPO_CLASSIFY = """你是计算机网络实验图片分类器。
任务：判断给定图片是否属于网络拓扑图。
输出字段：
1) image_type: topology / non_topology / unclear
2) classification_confidence: 0~1
3) reason: 简短说明

判断标准：
- topology：图片主要内容是设备、链路、接口、IP、VLAN、子网等网络拓扑要素。
- non_topology：图片主要是命令参考、表格、纯文字说明、配置片段、流程图、实验截图、结果截图、空白模板等，不是网络拓扑图。
- unclear：图片可能和网络相关，但无法可靠判断是否为拓扑图，或者拓扑元素过少/过模糊。

只输出严格 JSON，不要输出额外解释。
"""

SYSTEM_TOPO_REVIEW = """你是计算机网络实验拓扑图审核器。
任务：给定一张候选图片和一份拓扑 JSON 草稿，判断其是否是可用的网络拓扑结构化结果，并在必要时修正。
必须遵守：
1) 先判断图片类型：topology / non_topology / unclear。
2) 如果图片不是网络拓扑图，应设置 review_status=rejected_non_topology，is_usable=false，corrected_topology 置为空结构。
3) 如果图片是拓扑图，但草稿有明显臆造、关键设备/链路缺失严重或无法可靠使用，应设置 review_status=needs_manual_review，is_usable=false。
4) 如果整体可用但有少量不确定字段，设置 review_status=approved_with_warnings，is_usable=true。
5) 只有在设备和主要链路关系清晰、没有明显臆造时，才可设置 review_status=approved。
6) corrected_topology 必须基于图片进行修正，不确定处保留为空或写入 *_raw / warnings，绝不臆造。
7) 输出必须是严格 JSON。
"""

T = TypeVar("T")


def _safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _clear_directory_files(path: Path) -> None:
    if not path.exists():
        return
    for child in path.iterdir():
        if child.is_file():
            child.unlink(missing_ok=True)


def _sha1_bytes(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()


def normalize_experiment_id(raw: str) -> str:
    raw = (raw or "").strip()
    if not raw:
        raise ValueError("experiment_id 不能为空")

    match = _EXPERIMENT_RE.search(raw)
    if match:
        return f"lab{int(match.group(1))}"

    lowered = raw.lower()
    lowered = re.sub(r"[^a-z0-9_-]+", "_", lowered).strip("_")
    if lowered:
        return lowered

    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:8]
    return f"exp_{digest}"


def extract_experiment_id(text: str) -> Optional[str]:
    match = _EXPERIMENT_RE.search(text or "")
    if not match:
        return None
    return f"lab{int(match.group(1))}"


def experiment_label_from_id(experiment_id: str) -> str:
    match = re.search(r"lab(\d+)$", experiment_id or "", re.IGNORECASE)
    if match:
        return f"实验{int(match.group(1))}"
    return experiment_id


def infer_experiment_id_from_docx_path(docx_path: str) -> str:
    experiment_id = extract_experiment_id(Path(docx_path).stem)
    if experiment_id:
        return experiment_id
    raise ValueError(
        f"无法从文件名 {Path(docx_path).name!r} 推断实验编号，请显式传入 experiment_id。"
    )


def _read_json_model(path: Path, model_cls: Type[T]) -> Optional[T]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return model_cls.model_validate(payload)
    except Exception:
        return None


def _write_json_model(path: Path, model: Any) -> None:
    if hasattr(model, "model_dump"):
        payload = model.model_dump()
    else:
        payload = model
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _manifest_path_str(path: Path, base_dir: Path) -> str:
    try:
        return str(path.relative_to(base_dir))
    except ValueError:
        return str(path)


def _ensure_store_layout(store_dir: Path) -> Dict[str, Path]:
    layout = {
        "root": store_dir,
        "images": store_dir / "images",
        "classifications": store_dir / "classifications",
        "raw_json": store_dir / "raw_json",
        "reviews": store_dir / "reviews",
        "approved_json": store_dir / "approved_json",
    }
    for path in layout.values():
        _safe_mkdir(path)
    return layout


def _resolve_store_dir(
    experiment_id: str,
    *,
    output_root: str = TOPO_STORE_ROOT,
    store_dir: Optional[str] = None,
) -> Path:
    if store_dir:
        return Path(store_dir)
    return Path(output_root) / experiment_id


def _resolve_experiment_dir_for_runtime(store_root: str, experiment_id: str) -> Path:
    root_path = Path(store_root)
    if (root_path / "approved_json").exists():
        return root_path

    experiment_path = root_path / experiment_id
    if experiment_path.exists():
        return experiment_path

    if TOPO_PERSIST_DIR:
        legacy_path = Path(TOPO_PERSIST_DIR)
        if legacy_path.name == experiment_id and (legacy_path / "approved_json").exists():
            return legacy_path

    return experiment_path


def _list_available_experiment_ids(store_root: str = TOPO_STORE_ROOT) -> List[str]:
    root_path = Path(store_root)
    candidates: List[str] = []

    if root_path.exists():
        if (root_path / "approved_json").exists():
            candidates.append(root_path.name)
        else:
            for child in sorted(root_path.iterdir()):
                if child.is_dir() and (child / "approved_json").exists():
                    candidates.append(child.name)

    if TOPO_PERSIST_DIR:
        legacy_path = Path(TOPO_PERSIST_DIR)
        if legacy_path.exists() and (legacy_path / "approved_json").exists() and legacy_path.name not in candidates:
            candidates.append(legacy_path.name)

    return candidates


def _docx_extract_images(docx_path: str, images_dir: Path, min_pixels: int = 120 * 120) -> List[str]:
    """
    从 docx 中提取候选图片并保存到 images_dir。
    返回去重后的图片文件路径列表。
    """
    from docx import Document as DocxDocument
    from PIL import Image

    _safe_mkdir(images_dir)

    doc = DocxDocument(str(docx_path))
    extracted: List[str] = []
    seen: set[str] = set()

    for rel in doc.part.rels.values():
        if "image" not in str(rel.reltype):
            continue
        try:
            blob = rel.target_part.blob
        except Exception:
            continue

        digest = _sha1_bytes(blob)
        if digest in seen:
            continue
        seen.add(digest)

        suffix = Path(str(rel.target_part.partname)).suffix.lower() or ".png"
        out_path = images_dir / f"{digest}{suffix}"

        if not out_path.exists():
            out_path.write_bytes(blob)

        try:
            with Image.open(out_path) as image:
                if (image.size[0] * image.size[1]) < min_pixels:
                    out_path.unlink(missing_ok=True)
                    continue
        except Exception:
            out_path.unlink(missing_ok=True)
            continue

        extracted.append(str(out_path))

    return extracted


def _image_to_data_url(image_path: str) -> str:
    path = Path(image_path)
    suffix = path.suffix.lower()

    if suffix not in SUPPORTED_IMAGE_SUFFIX:
        from PIL import Image

        with Image.open(path) as image:
            buf = io.BytesIO()
            image.convert("RGB").save(buf, format="PNG")
            raw = buf.getvalue()
        b64 = base64.b64encode(raw).decode("utf-8")
        return f"data:image/png;base64,{b64}"

    mime = "image/png"
    if suffix in {".jpg", ".jpeg"}:
        mime = "image/jpeg"
    elif suffix == ".webp":
        mime = "image/webp"

    b64 = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def _classify_interface(itf) -> None:
    name = (itf.name or "").strip()
    if _PC_RE.match(itf.device) and not name:
        itf.kind = "host_nic"
        return
    if _SVI_RE.match(name):
        itf.kind = "svi"
        return
    if _PHY_RE.match(name):
        itf.kind = "physical"
        return
    itf.kind = itf.kind or "unknown"


def _postprocess_topology_for_kind(topo: TopologyExtraction) -> TopologyExtraction:
    for itf in topo.interfaces:
        _classify_interface(itf)

        if itf.kind == "physical" and itf.vlan_raw and re.fullmatch(r"\d+(-\d+)?", itf.vlan_raw.strip()):
            itf.allowed_vlans = itf.vlan_raw.strip()

    trunk_ports = set()
    for link in topo.links:
        if (link.medium or "").lower() == "trunk":
            if link.a.interface:
                trunk_ports.add((link.a.device, link.a.interface))
            if link.b.interface:
                trunk_ports.add((link.b.device, link.b.interface))

    for itf in topo.interfaces:
        if itf.kind == "physical" and itf.name and (itf.device, itf.name) in trunk_ports:
            itf.mode = itf.mode or "trunk"

    return topo


def classify_topology_image_with_gpt4o(
    image_path: str,
    experiment_id: str,
    *,
    model: str = "gpt-4o-mini",
) -> TopologyImageClassificationArtifact:
    from openai import OpenAI

    client = OpenAI()
    data_url = _image_to_data_url(image_path)

    resp = client.responses.parse(
        model=model,
        input=[
            {"role": "system", "content": SYSTEM_TOPO_CLASSIFY},
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": (
                            f"实验ID：{experiment_id}\n"
                            "请判断这张图片是否属于网络拓扑图。"
                        ),
                    },
                    {"type": "input_image", "image_url": data_url},
                ],
            },
        ],
        text_format=TopologyImageClassificationDecision,
    )

    decision = resp.output_parsed
    return TopologyImageClassificationArtifact(
        image_id=f"topo_{Path(image_path).stem}",
        experiment_id=experiment_id,
        source_image=str(image_path),
        classifier_model=model,
        image_type=decision.image_type,
        classification_confidence=decision.classification_confidence,
        reason=decision.reason,
    )


def extract_topology_draft_with_gpt4o(
    image_path: str,
    experiment_id: str,
    *,
    model: str = "gpt-4o",
) -> TopologyDraftArtifact:
    from openai import OpenAI

    client = OpenAI()
    data_url = _image_to_data_url(image_path)

    resp = client.responses.parse(
        model=model,
        input=[
            {"role": "system", "content": SYSTEM_TOPO},
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": USER_TOPO},
                    {"type": "input_image", "image_url": data_url},
                ],
            },
        ],
        text_format=TopologyExtraction,
    )

    topology = _postprocess_topology_for_kind(resp.output_parsed)
    return TopologyDraftArtifact(
        image_id=f"topo_{Path(image_path).stem}",
        experiment_id=experiment_id,
        source_image=str(image_path),
        extractor_model=model,
        topology=topology,
    )


def _finalize_review_artifact(review: TopologyReviewArtifact) -> TopologyReviewArtifact:
    review.corrected_topology = _postprocess_topology_for_kind(review.corrected_topology)

    if review.review_status == "rejected_non_topology":
        review.is_usable = False
        review.corrected_topology = TopologyExtraction(
            warnings=["该图片被审核为非拓扑图，未生成可用拓扑结构化数据。"]
        )
        return review

    if review.is_usable and not (review.corrected_topology.devices or review.corrected_topology.links):
        review.is_usable = False
        review.review_status = "needs_manual_review"
        review.issues.append("审核后的 JSON 仍缺少有效设备或链路，不能直接使用。")

    if review.review_status == "approved" and review.issues:
        review.review_status = "approved_with_warnings"

    return review


def review_topology_draft_with_gpt4o(
    image_path: str,
    draft: TopologyDraftArtifact,
    *,
    model: str = "gpt-4o",
) -> TopologyReviewArtifact:
    from openai import OpenAI

    client = OpenAI()
    data_url = _image_to_data_url(image_path)
    draft_json = json.dumps(draft.topology.model_dump(), ensure_ascii=False, indent=2)

    resp = client.responses.parse(
        model=model,
        input=[
            {"role": "system", "content": SYSTEM_TOPO_REVIEW},
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": (
                            f"实验ID：{draft.experiment_id}\n"
                            f"候选拓扑 JSON 草稿如下：\n{draft_json}\n\n"
                            "请结合图片逐项审核，并返回可直接落盘的修正结果。"
                        ),
                    },
                    {"type": "input_image", "image_url": data_url},
                ],
            },
        ],
        text_format=TopologyReviewDecision,
    )

    decision = resp.output_parsed
    review = TopologyReviewArtifact(
        image_id=draft.image_id,
        experiment_id=draft.experiment_id,
        source_image=draft.source_image,
        review_model=model,
        image_type=decision.image_type,
        review_status=decision.review_status,
        is_usable=decision.is_usable,
        review_confidence=decision.review_confidence,
        issues=decision.issues,
        summary=decision.summary,
        corrected_topology=decision.corrected_topology,
    )
    return _finalize_review_artifact(review)


def _approved_topology_from_review(review: TopologyReviewArtifact) -> Optional[TopologyExtraction]:
    if review.review_status not in _APPROVED_STATUSES:
        return None
    if not review.is_usable:
        return None
    topo = review.corrected_topology
    if not (topo.devices or topo.links):
        return None
    return topo


def build_topology_store(
    docx_path: str,
    *,
    output_root: str = TOPO_STORE_ROOT,
    experiment_id: Optional[str] = None,
    store_dir: Optional[str] = None,
    classify_model: str = "gpt-4o-mini",
    extract_model: str = "gpt-4o",
    review_model: str = "gpt-4o",
    overwrite: bool = False,
    min_image_pixels: int = 120 * 120,
) -> TopologyBuildManifest:
    resolved_experiment_id = normalize_experiment_id(
        experiment_id or infer_experiment_id_from_docx_path(docx_path)
    )
    resolved_store_dir = _resolve_store_dir(
        resolved_experiment_id,
        output_root=output_root,
        store_dir=store_dir,
    )
    layout = _ensure_store_layout(resolved_store_dir)
    manifest_path = resolved_store_dir / "manifest.json"

    if overwrite:
        _clear_directory_files(layout["images"])
        _clear_directory_files(layout["classifications"])
        _clear_directory_files(layout["raw_json"])
        _clear_directory_files(layout["reviews"])
        _clear_directory_files(layout["approved_json"])
        manifest_path.unlink(missing_ok=True)

    images = _docx_extract_images(docx_path, layout["images"], min_pixels=min_image_pixels)
    if not images:
        raise RuntimeError("未从 docx 中抽取到可用图片（可能图片太小/格式不支持/文档无图片）。")

    items: List[TopologyManifestItem] = []
    for image_path in tqdm(sorted(images), desc="Building topology store"):
        image_name = Path(image_path).stem
        image_id = f"topo_{image_name}"
        classification_path = layout["classifications"] / f"{image_id}.classification.json"
        raw_path = layout["raw_json"] / f"{image_id}.raw.json"
        review_path = layout["reviews"] / f"{image_id}.review.json"
        approved_path = layout["approved_json"] / f"{image_id}.json"

        classification = None if overwrite else _read_json_model(
            classification_path, TopologyImageClassificationArtifact
        )
        if classification is None:
            try:
                classification = classify_topology_image_with_gpt4o(
                    image_path,
                    resolved_experiment_id,
                    model=classify_model,
                )
            except Exception as exc:
                classification = TopologyImageClassificationArtifact(
                    image_id=image_id,
                    experiment_id=resolved_experiment_id,
                    source_image=str(image_path),
                    classifier_model=classify_model,
                    image_type="unclear",
                    classification_confidence=0.0,
                    reason=f"GPT 图片分类失败: {repr(exc)}",
                )
            _write_json_model(classification_path, classification)

        if classification.image_type == "non_topology":
            raw_path.unlink(missing_ok=True)
            review_path.unlink(missing_ok=True)
            approved_path.unlink(missing_ok=True)

            items.append(
                TopologyManifestItem(
                    image_id=image_id,
                    source_image=_manifest_path_str(Path(image_path), resolved_store_dir),
                    classification_path=_manifest_path_str(classification_path, resolved_store_dir),
                    raw_json_path=None,
                    review_path=None,
                    approved_json_path=None,
                    image_type=classification.image_type,
                    classification_confidence=classification.classification_confidence,
                    review_status="rejected_non_topology",
                    is_usable=False,
                    review_confidence=0.0,
                    summary=f"前置分类判定为非拓扑图：{classification.reason}",
                )
            )
            continue

        draft = None if overwrite else _read_json_model(raw_path, TopologyDraftArtifact)
        if draft is None:
            try:
                draft = extract_topology_draft_with_gpt4o(
                    image_path,
                    resolved_experiment_id,
                    model=extract_model,
                )
            except Exception as exc:
                draft = TopologyDraftArtifact(
                    image_id=image_id,
                    experiment_id=resolved_experiment_id,
                    source_image=str(image_path),
                    extractor_model=extract_model,
                    topology=TopologyExtraction(
                        warnings=[f"GPT-4o 抽取失败: {repr(exc)}"]
                    ),
                )
            _write_json_model(raw_path, draft)

        review = None if overwrite else _read_json_model(review_path, TopologyReviewArtifact)
        if review is None:
            try:
                review = review_topology_draft_with_gpt4o(
                    image_path,
                    draft,
                    model=review_model,
                )
            except Exception as exc:
                review = TopologyReviewArtifact(
                    image_id=image_id,
                    experiment_id=resolved_experiment_id,
                    source_image=str(image_path),
                    review_model=review_model,
                    review_status="needs_manual_review",
                    is_usable=False,
                    issues=[f"GPT-4o 审核失败: {repr(exc)}"],
                    summary="审核调用失败，已标记为需要人工复核。",
                    corrected_topology=draft.topology,
                )
                review = _finalize_review_artifact(review)
            _write_json_model(review_path, review)

        approved_topology = _approved_topology_from_review(review)
        approved_json_rel: Optional[str] = None
        if approved_topology is not None:
            _write_json_model(approved_path, approved_topology)
            approved_json_rel = str(approved_path.relative_to(resolved_store_dir))
        else:
            approved_path.unlink(missing_ok=True)

        items.append(
            TopologyManifestItem(
                image_id=image_id,
                source_image=_manifest_path_str(Path(image_path), resolved_store_dir),
                classification_path=_manifest_path_str(classification_path, resolved_store_dir),
                raw_json_path=_manifest_path_str(raw_path, resolved_store_dir),
                review_path=_manifest_path_str(review_path, resolved_store_dir),
                approved_json_path=approved_json_rel,
                image_type=review.image_type if review.image_type != "unclear" else classification.image_type,
                classification_confidence=classification.classification_confidence,
                review_status=review.review_status,
                is_usable=review.is_usable,
                review_confidence=review.review_confidence,
                summary=review.summary,
            )
        )

    manifest = TopologyBuildManifest(
        experiment_id=resolved_experiment_id,
        experiment_label=experiment_label_from_id(resolved_experiment_id),
        docx_path=str(docx_path),
        store_dir=str(resolved_store_dir),
        classifier_model=classify_model,
        extractor_model=extract_model,
        review_model=review_model,
        images_total=len(items),
        non_topology_prefiltered_total=sum(
            1 for item in items if item.image_type == "non_topology" and item.raw_json_path is None
        ),
        approved_total=sum(1 for item in items if item.review_status == "approved" and item.approved_json_path),
        approved_with_warnings_total=sum(
            1 for item in items if item.review_status == "approved_with_warnings" and item.approved_json_path
        ),
        needs_manual_review_total=sum(1 for item in items if item.review_status == "needs_manual_review"),
        rejected_non_topology_total=sum(1 for item in items if item.review_status == "rejected_non_topology"),
        items=items,
    )
    _write_json_model(manifest_path, manifest)
    return manifest


def _load_approved_topology_registry(store_root: str, experiment_id: str) -> Dict[str, dict]:
    experiment_dir = _resolve_experiment_dir_for_runtime(store_root, experiment_id)
    approved_dir = experiment_dir / "approved_json"
    registry: Dict[str, dict] = {}

    if not approved_dir.exists():
        return registry

    cache_key = f"{experiment_dir.resolve()}::{experiment_id}"
    cached = _GLOBAL_APPROVED_TOPO_REGISTRY.get(cache_key)
    if cached is not None:
        return cached

    for json_path in sorted(approved_dir.glob("*.json")):
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if data.get("devices") or data.get("links"):
            registry[json_path.stem] = data

    _GLOBAL_APPROVED_TOPO_REGISTRY[cache_key] = registry
    return registry


def _format_topology_as_text(topo_data: dict, label: str = "") -> str:
    header = f"=== 网络拓扑{(' - ' + label) if label else ''} ==="
    lines = [header]

    devices = topo_data.get("devices") or []
    if devices:
        lines.append("\n【设备列表】")
        for device in devices:
            line = f"  {device['name']}  类型={device.get('type', 'unknown')}"
            if device.get("mgmt_ip"):
                line += f"  管理IP={device['mgmt_ip']}"
            lines.append(line)

    interfaces = topo_data.get("interfaces") or []
    if interfaces:
        lines.append("\n【接口信息】")
        for itf in interfaces:
            dev = itf.get("device", "?")
            name = itf.get("name", "")
            port = f"{dev}.{name}" if name else dev
            parts = [f"  {port}"]
            if itf.get("kind"):
                parts.append(f"kind={itf['kind']}")
            if itf.get("ip"):
                parts.append(f"IP={itf['ip']}")
            elif itf.get("ip_raw"):
                parts.append(f"IP(待定)={itf['ip_raw']}")
            if itf.get("mask"):
                parts.append(f"掩码={itf['mask']}")
            elif itf.get("mask_raw"):
                parts.append(f"掩码(待定)={itf['mask_raw']}")
            if itf.get("vlan"):
                parts.append(f"VLAN={itf['vlan']}")
            elif itf.get("vlan_raw"):
                parts.append(f"VLAN(待定)={itf['vlan_raw']}")
            if itf.get("mode"):
                parts.append(f"模式={itf['mode']}")
            if itf.get("allowed_vlans"):
                parts.append(f"允许VLAN={itf['allowed_vlans']}")
            if itf.get("access_vlan"):
                parts.append(f"接入VLAN={itf['access_vlan']}")
            lines.append("  ".join(parts))

    links = topo_data.get("links") or []
    if links:
        lines.append("\n【链路连接】")
        for link in links:
            a = link.get("a") or {}
            b = link.get("b") or {}
            a_str = a.get("device", "?")
            if a.get("interface"):
                a_str += f"({a['interface']})"
            b_str = b.get("device", "?")
            if b.get("interface"):
                b_str += f"({b['interface']})"
            medium = link.get("medium", "")
            suffix = f"  [{medium}]" if medium and medium != "unknown" else ""
            lines.append(f"  {a_str} <-> {b_str}{suffix}")

    subnets = topo_data.get("subnets") or []
    if subnets:
        lines.append("\n【子网】")
        for subnet in subnets:
            lines.append(f"  {subnet.get('cidr', '?')}")

    warnings = topo_data.get("warnings") or []
    if warnings:
        lines.append("\n【不确定信息（请注意）】")
        for warning in warnings:
            lines.append(f"  ! {warning}")

    return "\n".join(lines)


def _render_topology_registry(registry: Dict[str, dict], experiment_id: str) -> str:
    parts = [f"=== 当前实验：{experiment_label_from_id(experiment_id)} ({experiment_id}) ==="]
    for index, (topo_id, topo_data) in enumerate(registry.items(), 1):
        label = f"拓扑{index}"
        if topo_id:
            label = f"{label} / {topo_id}"
        parts.append(_format_topology_as_text(topo_data, label=label))
    return "\n\n".join(parts)


def _resolve_runtime_experiment_id(
    query: str,
    *,
    explicit_experiment_id: Optional[str] = None,
    store_root: str = TOPO_STORE_ROOT,
) -> Optional[str]:
    if explicit_experiment_id:
        return normalize_experiment_id(explicit_experiment_id)

    detected = extract_experiment_id(query)
    if detected:
        return detected

    if TOPO_DEFAULT_EXPERIMENT_ID:
        return normalize_experiment_id(TOPO_DEFAULT_EXPERIMENT_ID)

    available = _list_available_experiment_ids(store_root)
    if len(available) == 1:
        return available[0]
    return None


def LoadTopoRetriever(
    *,
    experiment_id: Optional[str] = None,
    store_root: str = TOPO_STORE_ROOT,
    embedding_model: str = "text-embedding-3-small",
    k: int = 6,
) -> Callable[[str], Dict[str, Any]]:
    """
    返回一个只读取 approved_json 的 retriever。

    `embedding_model` 和 `k` 参数仅为兼容旧调用保留。
    """
    del embedding_model, k

    def _retriever(query: str) -> Dict[str, Any]:
        resolved_experiment_id = _resolve_runtime_experiment_id(
            query,
            explicit_experiment_id=experiment_id,
            store_root=store_root,
        )
        if not resolved_experiment_id:
            available = _list_available_experiment_ids(store_root)
            hint = "、".join(available) if available else "无"
            return {
                "context": "",
                "topo_ids": [],
                "warning": f"未识别到实验编号。请在问题中说明实验号，例如“实验13”或“lab13”。当前可用实验：{hint}",
            }

        registry = _load_approved_topology_registry(store_root, resolved_experiment_id)
        if not registry:
            return {
                "context": "",
                "topo_ids": [],
                "experiment_id": resolved_experiment_id,
                "warning": f"在 {store_root}/{resolved_experiment_id}/approved_json/ 下未找到审核通过的拓扑 JSON。",
            }

        return {
            "context": _render_topology_registry(registry, resolved_experiment_id),
            "topo_ids": list(registry.keys()),
            "experiment_id": resolved_experiment_id,
        }

    return _retriever


def TopoRetriever(
    query: str,
    *,
    experiment_id: Optional[str] = None,
    store_root: str = TOPO_STORE_ROOT,
) -> str:
    """
    提供给 agent.py 使用的全局拓扑函数。
    运行时只读取审核通过的 approved_json。
    """
    result = LoadTopoRetriever(experiment_id=experiment_id, store_root=store_root)(query)
    if result.get("context"):
        return result["context"]
    return result.get("warning", "未找到可用拓扑数据。")


def BuildTopoIndexFromDocxImages(
    docx_path: str,
    persist_dir: str = "topo_store",
    vision_model: str = "gpt-4o",
    embedding_model: str = "text-embedding-3-small",
    overwrite: bool = False,
    min_image_pixels: int = 120 * 120,
    embed_batch_size: int = 32,
) -> str:
    """
    兼容旧接口：构建并返回拓扑存储目录。
    """
    del embedding_model, embed_batch_size

    persist_path = Path(persist_dir)
    explicit_experiment_id = extract_experiment_id(persist_path.name)
    kwargs: Dict[str, Any] = {
        "classify_model": "gpt-4o-mini",
        "extract_model": vision_model,
        "review_model": vision_model,
        "overwrite": overwrite,
        "min_image_pixels": min_image_pixels,
    }
    if explicit_experiment_id:
        kwargs["experiment_id"] = explicit_experiment_id
        kwargs["store_dir"] = persist_dir
    else:
        kwargs["output_root"] = persist_dir

    manifest = build_topology_store(docx_path, **kwargs)
    return manifest.store_dir


def BuildAndLoadTopoRetrieverFromDocxImages(
    docx_path: str,
    persist_dir: str = "topo_store",
    vision_model: str = "gpt-4o",
    embedding_model: str = "text-embedding-3-small",
    overwrite: bool = False,
    k: int = 6,
) -> Tuple[str, Callable[[str], Dict[str, Any]]]:
    """
    一次性：构建拓扑存储并返回只读 approved_json 的 retriever。
    """
    out_dir = BuildTopoIndexFromDocxImages(
        docx_path=docx_path,
        persist_dir=persist_dir,
        vision_model=vision_model,
        embedding_model=embedding_model,
        overwrite=overwrite,
    )
    experiment_id = normalize_experiment_id(Path(out_dir).name)
    retriever = LoadTopoRetriever(
        experiment_id=experiment_id,
        store_root=str(Path(out_dir).parent),
        embedding_model=embedding_model,
        k=k,
    )
    return out_dir, retriever


if __name__ == "__main__":
    manifest = build_topology_store(
        "/Users/baoliliu/Downloads/networking-agent/RAG-Agent/data/实验13：子网划分（详细版）.docx",
        output_root="topo_store",
        classify_model="gpt-4o-mini",
        extract_model="gpt-4o",
        review_model="gpt-4o",
        overwrite=True,
        min_image_pixels=64 * 64,
    )
    print("Topology store saved to:", manifest.store_dir)
