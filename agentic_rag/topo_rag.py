# topo_rag_pack.py
from __future__ import annotations

import os
import io
import re
import json
import base64
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable, Tuple

from pydantic import BaseModel, Field
from tqdm import tqdm

import networkx as nx
from networkx.readwrite import json_graph

from docx import Document as DocxDocument
from PIL import Image

from openai import OpenAI

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS


# =========================
# 1) Topology schema (Pydantic)
# =========================

class Device(BaseModel):
    name: str
    type: str = Field(default="unknown", description="router/switch/host/firewall/server/unknown")
    mgmt_ip: Optional[str] = None

class Interface(BaseModel):
    device: str
    name: str
    ip: Optional[str] = None
    mask: Optional[str] = None
    vlan: Optional[str] = None

class LinkEnd(BaseModel):
    device: str
    interface: Optional[str] = None

class Link(BaseModel):
    a: LinkEnd
    b: LinkEnd
    medium: Optional[str] = "unknown"

class Subnet(BaseModel):
    cidr: str
    members: List[LinkEnd] = Field(default_factory=list)

class TopologyExtraction(BaseModel):
    devices: List[Device] = Field(default_factory=list)
    interfaces: List[Interface] = Field(default_factory=list)
    links: List[Link] = Field(default_factory=list)
    subnets: List[Subnet] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


# =========================
# 2) Helpers: extract images from DOCX
# =========================

SUPPORTED_IMAGE_SUFFIX = {".png", ".jpg", ".jpeg", ".webp"}

def _safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def _sha1_bytes(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()

def _docx_extract_images(docx_path: str, out_dir: str, min_pixels: int = 120 * 120) -> List[str]:
    """
    从 docx 的 relationship 里抽取图片，保存到 out_dir/images/
    返回图片文件路径列表（去重）
    """
    docx_path = str(docx_path)
    root = Path(out_dir)
    img_dir = root / "images"
    _safe_mkdir(img_dir)

    doc = DocxDocument(docx_path)
    extracted: List[str] = []
    seen: set[str] = set()

    # python-docx: relationships
    for rel in doc.part.rels.values():
        # 只取 image rel
        if "image" not in str(rel.reltype):
            continue
        try:
            blob = rel.target_part.blob  # bytes
        except Exception:
            continue

        digest = _sha1_bytes(blob)
        if digest in seen:
            continue
        seen.add(digest)

        # 原始扩展名（可能是 .png/.jpeg 等）
        partname = str(rel.target_part.partname)  # e.g. '/word/media/image1.png'
        suffix = Path(partname).suffix.lower() or ".png"

        # 统一输出为 png/jpg/webp；如果是奇怪格式，尝试用 PIL 转 png
        out_path = img_dir / f"{digest}{suffix}"
        if out_path.exists():
            extracted.append(str(out_path))
            continue

        # 先直接写入
        out_path.write_bytes(blob)

        # 过滤太小的图片（避免把小图标也当拓扑）
        try:
            with Image.open(out_path) as im:
                if (im.size[0] * im.size[1]) < min_pixels:
                    out_path.unlink(missing_ok=True)
                    continue
        except Exception:
            # PIL 打不开就跳过
            out_path.unlink(missing_ok=True)
            continue

        extracted.append(str(out_path))

    return extracted


# =========================
# 3) Multimodal extraction: image -> TopologyExtraction
# =========================

SYSTEM_TOPO = """你是计算机网络实验拓扑图解析器。
任务：从给定拓扑图中提取设备、接口、链路、IP/掩码/VLAN/子网等信息，并输出为严格匹配 schema 的 JSON。
要求：
1) 不要臆造：看不清就填 null/留空，并在 warnings 说明不确定性来源。
2) 设备命名尽量沿用图中标注（如 R1/SW1/PC1）。
3) 接口名/端口号看不清可留空；但链路“设备-设备”关系尽量保留。
4) 只输出结构化内容，不输出额外解释性文本。
"""

USER_TOPO = """请解析这张网络拓扑图，输出设备、接口、链路、IP/掩码/VLAN/子网信息。"""

def _image_to_data_url(image_path: str) -> str:
    p = Path(image_path)
    suffix = p.suffix.lower()
    # openai vision 常见支持 png/jpg/webp；如果不是就转 png
    if suffix not in SUPPORTED_IMAGE_SUFFIX:
        # 尝试转 png
        with Image.open(p) as im:
            buf = io.BytesIO()
            im.convert("RGB").save(buf, format="PNG")
            b = buf.getvalue()
        b64 = base64.b64encode(b).decode("utf-8")
        return f"data:image/png;base64,{b64}"

    mime = "image/png"
    if suffix in {".jpg", ".jpeg"}:
        mime = "image/jpeg"
    elif suffix == ".webp":
        mime = "image/webp"

    b64 = base64.b64encode(p.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{b64}"

def _extract_topology_from_image_openai(image_path: str, vision_model: str = "gpt-4o-mini") -> TopologyExtraction:
    """
    需要 openai>=1.x 且支持 Responses API 的 parse（Structured Outputs）。
    """
    client = OpenAI()
    data_url = _image_to_data_url(image_path)

    resp = client.responses.parse(
        model=vision_model,
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
    return resp.output_parsed


# =========================
# 4) Build graph + facts
# =========================

def _build_nx_graph(topo: TopologyExtraction) -> nx.Graph:
    G = nx.Graph()

    for d in topo.devices:
        G.add_node(d.name, type=d.type, mgmt_ip=d.mgmt_ip)

    iface_map: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for itf in topo.interfaces:
        if itf.device and itf.name:
            iface_map[(itf.device, itf.name)] = {
                "ip": itf.ip,
                "mask": itf.mask,
                "vlan": itf.vlan,
            }

    for lk in topo.links:
        a_dev, b_dev = lk.a.device, lk.b.device
        a_if, b_if = lk.a.interface, lk.b.interface

        if a_dev not in G:
            G.add_node(a_dev, type="unknown")
        if b_dev not in G:
            G.add_node(b_dev, type="unknown")

        attrs = {"medium": lk.medium}
        if a_if:
            attrs["a_if"] = a_if
            attrs["a_if_attrs"] = iface_map.get((a_dev, a_if), {})
        if b_if:
            attrs["b_if"] = b_if
            attrs["b_if_attrs"] = iface_map.get((b_dev, b_if), {})

        G.add_edge(a_dev, b_dev, **attrs)

    return G

def _topology_facts(G: nx.Graph) -> List[str]:
    facts: List[str] = []
    for n, attrs in G.nodes(data=True):
        facts.append(f"DEVICE name={n} type={attrs.get('type')} mgmt_ip={attrs.get('mgmt_ip')}")
    for u, v, attrs in G.edges(data=True):
        facts.append(
            f"LINK {u}({attrs.get('a_if')}) <-> {v}({attrs.get('b_if')}) medium={attrs.get('medium')}"
        )
        a = (attrs.get("a_if_attrs") or {})
        b = (attrs.get("b_if_attrs") or {})
        if attrs.get("a_if") and a.get("ip"):
            facts.append(f"INTERFACE {u}.{attrs.get('a_if')} ip={a.get('ip')} mask={a.get('mask')} vlan={a.get('vlan')}")
        if attrs.get("b_if") and b.get("ip"):
            facts.append(f"INTERFACE {v}.{attrs.get('b_if')} ip={b.get('ip')} mask={b.get('mask')} vlan={b.get('vlan')}")
    return facts


# =========================
# 5) FAISS build with tqdm (batch)
# =========================

def _faiss_from_documents_with_tqdm(
    documents: List[Document],
    embedding: OpenAIEmbeddings,
    batch_size: int = 32,
) -> FAISS:
    texts = [d.page_content for d in documents]
    metadatas = [d.metadata for d in documents]

    vectors: List[List[float]] = []
    print("正在生成向量(Topo Facts)...")

    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
        batch = texts[i : i + batch_size]
        batch_vecs = embedding.embed_documents(batch)
        vectors.extend(batch_vecs)

    print("正在构建FAISS索引(Topo Facts)...")
    return FAISS.from_embeddings(
        text_embeddings=list(zip(texts, vectors)),
        embedding=embedding,
        metadatas=metadatas,
    )


# =========================
# 6) Public API: Build + Load Retriever
# =========================

DEVICE_RE_DEFAULT = re.compile(r"\b(?:R|SW|PC)\d+\b", re.IGNORECASE)

def BuildTopoIndexFromDocxImages(
    docx_path: str,
    persist_dir: str = "topo_index_store",
    vision_model: str = "gpt-4o-mini",
    embedding_model: str = "text-embedding-3-small",
    overwrite: bool = False,
    min_image_pixels: int = 120 * 120,
    embed_batch_size: int = 32,
) -> str:
    """
    1) 从 docx 抽取图片
    2) 图片 -> 结构化 topo JSON（OpenAI vision）
    3) topo -> NetworkX 图 + topo facts
    4) topo facts -> FAISS 向量库并 save_local
    同时落盘：images/ topology_json/ graphs/ manifest.json faiss_topo_index/
    返回 persist_dir
    """
    root = Path(persist_dir)
    _safe_mkdir(root)

    faiss_dir = root / "faiss_topo_index"
    topo_json_dir = root / "topology_json"
    graphs_dir = root / "graphs"
    _safe_mkdir(topo_json_dir)
    _safe_mkdir(graphs_dir)

    # 如果已有索引且不覆盖，直接返回
    if faiss_dir.exists() and (faiss_dir / "index.faiss").exists() and not overwrite:
        print(f"[跳过构建] 检测到已有索引：{faiss_dir}")
        return str(root)

    # 覆盖则清理旧内容
    if overwrite and faiss_dir.exists():
        # 只清理 faiss 目录，其他内容保留
        for fp in faiss_dir.glob("*"):
            try:
                fp.unlink()
            except Exception:
                pass

    # 1) 抽图
    images = _docx_extract_images(docx_path, out_dir=str(root), min_pixels=min_image_pixels)
    if not images:
        raise RuntimeError("未从 docx 中抽取到可用图片（可能图片太小/格式不支持/文档无图片）。")

    # 2) 逐图解析 topo
    all_fact_docs: List[Document] = []
    manifest: Dict[str, Any] = {
        "docx_path": str(docx_path),
        "persist_dir": str(root),
        "vision_model": vision_model,
        "embedding_model": embedding_model,
        "topologies": [],  # list of {topo_id, image_path, json_path, graph_path, n_facts, warnings}
    }

    for idx, img_path in enumerate(tqdm(images, desc="Parsing topology images")):
        topo_id = f"topo_{idx:03d}"

        try:
            topo = _extract_topology_from_image_openai(img_path, vision_model=vision_model)
        except Exception as e:
            # 解析失败也写 manifest，方便你排查
            topo = TopologyExtraction(warnings=[f"OpenAI解析失败: {repr(e)}"])

        # 若没有提取到任何设备/链路，可以选择跳过（避免污染索引）
        if (not topo.devices) and (not topo.links):
            topo.warnings.append("解析结果为空（无 devices/links），可能不是拓扑图或标签不可读；已跳过入库。")
            # 仍然落盘 JSON，便于你复核
            json_path = topo_json_dir / f"{topo_id}.json"
            json_path.write_text(json.dumps(topo.model_dump(), ensure_ascii=False, indent=2), encoding="utf-8")

            manifest["topologies"].append(
                {
                    "topo_id": topo_id,
                    "image_path": str(img_path),
                    "json_path": str(json_path),
                    "graph_path": None,
                    "n_facts": 0,
                    "warnings": topo.warnings,
                }
            )
            continue

        # 3) topo -> graph + facts
        G = _build_nx_graph(topo)
        facts = _topology_facts(G)

        # 4) 落盘 topo json + graph json
        json_path = topo_json_dir / f"{topo_id}.json"
        json_path.write_text(json.dumps(topo.model_dump(), ensure_ascii=False, indent=2), encoding="utf-8")

        graph_path = graphs_dir / f"{topo_id}.graph.json"
        graph_path.write_text(json.dumps(json_graph.node_link_data(G), ensure_ascii=False, indent=2), encoding="utf-8")

        # 5) facts -> Documents
        for f in facts:
            prefix = f.split(" ", 1)[0] if f else "FACT"
            all_fact_docs.append(
                Document(
                    page_content=f,
                    metadata={
                        "type": "topo_fact",
                        "fact_type": prefix,
                        "topo_id": topo_id,
                        "image_path": str(img_path),
                    },
                )
            )

        manifest["topologies"].append(
            {
                "topo_id": topo_id,
                "image_path": str(img_path),
                "json_path": str(json_path),
                "graph_path": str(graph_path),
                "n_facts": len(facts),
                "warnings": topo.warnings,
            }
        )

    if not all_fact_docs:
        raise RuntimeError("所有图片都未得到有效拓扑 facts，未能构建向量库。请检查 docx 图片是否为拓扑图且清晰可读。")

    # 6) facts -> FAISS
    embeddings = OpenAIEmbeddings(model=embedding_model)
    db = _faiss_from_documents_with_tqdm(all_fact_docs, embeddings, batch_size=embed_batch_size)

    _safe_mkdir(faiss_dir)
    db.save_local(str(faiss_dir))

    # 7) 写 manifest
    (root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"拓扑向量索引已保存到：{faiss_dir}/")
    return str(root)


def LoadTopoRetriever(
    persist_dir: str,
    embedding_model: str = "text-embedding-3-small",
    k: int = 6,
    device_regex: re.Pattern = DEVICE_RE_DEFAULT,
    with_graph_context: bool = True,
) -> Callable[[str], Dict[str, Any]]:
    """
    加载本地 topo FAISS 索引，并返回一个可调用函数 TopoRetriever(query)->dict：
      {
        "docs": List[Document],              # topo facts 命中
        "context": str,                      # 拼好的上下文（facts + graph_context）
        "graph_context": str,                # 可选：图结构上下文
        "topo_ids": List[str],               # 命中的拓扑id
      }
    """
    root = Path(persist_dir)
    faiss_dir = root / "faiss_topo_index"
    manifest_path = root / "manifest.json"
    graphs_dir = root / "graphs"

    if not (faiss_dir.exists() and (faiss_dir / "index.faiss").exists()):
        raise RuntimeError(f"未找到 FAISS 索引：{faiss_dir}. 请先运行 BuildTopoIndexFromDocxImages().")

    embeddings = OpenAIEmbeddings(model=embedding_model)
    db = FAISS.load_local(
        str(faiss_dir),
        embeddings,
        allow_dangerous_deserialization=True,
    )
    retriever = db.as_retriever(search_kwargs={"k": k})

    manifest = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    # 预加载所有图（也可按 topo_id 懒加载）
    graphs: Dict[str, nx.Graph] = {}
    if with_graph_context and graphs_dir.exists():
        for gp in graphs_dir.glob("*.graph.json"):
            topo_id = gp.name.replace(".graph.json", "")
            try:
                data = json.loads(gp.read_text(encoding="utf-8"))
                graphs[topo_id] = json_graph.node_link_graph(data)
            except Exception:
                continue

    def _extract_devices(q: str) -> List[str]:
        ds = device_regex.findall(q or "")
        # 统一成大写，避免 R1/r1 不一致
        return list(dict.fromkeys([d.upper() for d in ds]))

    def _graph_context_for(topo_id: str, devices: List[str], hop: int = 1, max_lines: int = 30) -> str:
        G = graphs.get(topo_id)
        if not G or not devices:
            return ""

        # 取提到的设备及其 1-hop 邻居
        nodes = set()
        for d in devices:
            if d in G:
                nodes.add(d)
                nodes.update(list(G.neighbors(d)))

        if not nodes:
            return ""

        sg = G.subgraph(nodes)
        lines = []
        for u, v, attrs in sg.edges(data=True):
            lines.append(
                f"{u} <-> {v} (a_if={attrs.get('a_if')}, b_if={attrs.get('b_if')}, medium={attrs.get('medium')})"
            )

        if len(lines) > max_lines:
            lines = lines[:max_lines] + ["...(省略更多链路)"]

        return "\n".join(lines)

    def TopoRetriever(query: str) -> Dict[str, Any]:
        # 注意：新版 LangChain retriever 支持 invoke
        try:
            docs = retriever.invoke(query)
        except Exception:
            # 兼容旧版
            docs = retriever.get_relevant_documents(query)

        topo_ids = []
        for d in docs:
            tid = (d.metadata or {}).get("topo_id")
            if tid and tid not in topo_ids:
                topo_ids.append(tid)

        devices = _extract_devices(query)

        graph_ctx_parts = []
        if with_graph_context and topo_ids and devices:
            for tid in topo_ids[:3]:  # 最多取前三个相关拓扑，避免太长
                gctx = _graph_context_for(tid, devices)
                if gctx:
                    graph_ctx_parts.append(f"[GRAPH {tid}]\n{gctx}")

        graph_context = "\n\n".join(graph_ctx_parts).strip()

        facts_context = "\n".join([d.page_content for d in docs]).strip()
        context = facts_context
        if graph_context:
            context = (facts_context + "\n\n" + graph_context).strip()

        return {
            "docs": docs,
            "context": context,
            "graph_context": graph_context,
            "topo_ids": topo_ids,
        }

    return TopoRetriever


# =========================
# 7) Optional: one-call wrapper (build + load)
# =========================

def BuildAndLoadTopoRetrieverFromDocxImages(
    docx_path: str,
    persist_dir: str = "topo_index_store",
    vision_model: str = "gpt-5o",
    embedding_model: str = "text-embedding-3-small",
    overwrite: bool = False,
    k: int = 6,
) -> Tuple[str, Callable[[str], Dict[str, Any]]]:
    """
    一次性：构建索引 + 返回可检索函数
    """
    out_dir = BuildTopoIndexFromDocxImages(
        docx_path=docx_path,
        persist_dir=persist_dir,
        vision_model=vision_model,
        embedding_model=embedding_model,
        overwrite=overwrite,
    )
    retr = LoadTopoRetriever(
        persist_dir=out_dir,
        embedding_model=embedding_model,
        k=k,
        with_graph_context=True,
    )
    return out_dir, retr

out_dir = BuildTopoIndexFromDocxImages(
    docx_path="/Users/baoliliu/Downloads/networking-agent/RAG-Agent/data/实验13：子网划分（详细版）.docx",
    persist_dir="topo_store_lab13",
    vision_model="gpt-5-mini",
    embedding_model="text-embedding-3-small",
    overwrite=True,
)
print("Topo index saved to:", out_dir)
