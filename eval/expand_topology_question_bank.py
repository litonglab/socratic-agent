"""
基于 topo_store/*/approved_json 自动扩充拓扑题库。

目标：
1) 覆盖更多实验（lab1/lab2/...）
2) 题目尽量使用稳定字段（设备/链路/子网），避免依赖不确定字段
3) 自动去重并延续 TQ 编号

示例：
  python eval/expand_topology_question_bank.py
  python eval/expand_topology_question_bank.py --max-new-per-topology 4
  python eval/expand_topology_question_bank.py --output eval/topology_question_bank.json
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Dict, List, Tuple


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TOPO_ROOT = ROOT / "topo_store"
DEFAULT_BANK = ROOT / "eval" / "topology_question_bank.json"
_TYPE_ORDER = ["concept", "config", "troubleshooting", "calculation", "unknown"]


def _resolve(path_text: str) -> Path:
    path = Path(path_text)
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def _canonical(text: str) -> str:
    s = (text or "").strip().lower()
    s = re.sub(r"\s+", "", s)
    s = re.sub(r"[，。！？、,.!?：:；;（）()【】\\[\\]\"“”‘’`']", "", s)
    return s


def _lab_label(experiment_id: str) -> str:
    m = re.search(r"lab(\d+)$", experiment_id or "", re.IGNORECASE)
    if m:
        return f"实验{int(m.group(1))}"
    return experiment_id


def _device_type_count(devices: List[dict]) -> Dict[str, int]:
    c = Counter((d.get("type") or "unknown") for d in devices)
    return {
        "router": c.get("router", 0),
        "switch": c.get("switch", 0),
        "host": c.get("host", 0),
        "unknown": c.get("unknown", 0),
    }


def _build_graph(links: List[dict]) -> Tuple[Dict[str, List[str]], Dict[Tuple[str, str], Tuple[str, str]]]:
    graph: Dict[str, List[str]] = defaultdict(list)
    edge_meta: Dict[Tuple[str, str], Tuple[str, str]] = {}
    for link in links:
        a = (link.get("a") or {}).get("device")
        b = (link.get("b") or {}).get("device")
        ia = (link.get("a") or {}).get("interface")
        ib = (link.get("b") or {}).get("interface")
        if not a or not b:
            continue
        graph[a].append(b)
        graph[b].append(a)
        edge_meta[(a, b)] = (ia, ib)
        edge_meta[(b, a)] = (ib, ia)
    return graph, edge_meta


def _shortest_path(graph: Dict[str, List[str]], src: str, dst: str) -> List[str]:
    if src == dst:
        return [src]
    q = deque([src])
    prev = {src: None}
    while q:
        cur = q.popleft()
        for nxt in graph.get(cur, []):
            if nxt in prev:
                continue
            prev[nxt] = cur
            if nxt == dst:
                path = [dst]
                while path[-1] is not None:
                    p = prev[path[-1]]
                    if p is None:
                        break
                    path.append(p)
                return list(reversed(path))
            q.append(nxt)
    return []


def _safe_member_text(member: dict) -> str:
    dev = member.get("device", "?")
    itf = member.get("interface")
    return f"{dev}.{itf}" if itf else str(dev)


def generate_candidates(
    *,
    experiment_id: str,
    source_rel: str,
    topo_id: str,
    topo_data: dict,
) -> List[dict]:
    """为单个拓扑 JSON 生成候选题。"""
    devices = topo_data.get("devices") or []
    links = topo_data.get("links") or []
    subnets = topo_data.get("subnets") or []
    if not devices and not links:
        return []

    label = _lab_label(experiment_id)
    candidates: List[dict] = []
    tcount = _device_type_count(devices)
    graph, edge_meta = _build_graph(links)

    # 1) 设备类型计数题
    if devices:
        reference = (
            f"根据该拓扑的设备清单，路由器 {tcount['router']} 台、交换机 {tcount['switch']} 台、"
            f"主机 {tcount['host']} 台；其余未明确类型设备 {tcount['unknown']} 台。"
        )
        candidates.append(
            {
                "question": f"在{label}的拓扑 {topo_id} 中，路由器、交换机、主机分别有多少台？",
                "reference": reference,
                "type": "calculation",
                "difficulty": "easy",
            }
        )

    # 2) 链路总数题
    if links:
        candidates.append(
            {
                "question": f"在{label}的拓扑 {topo_id} 中，抽取结果一共包含多少条设备链路？",
                "reference": (
                    f"根据结构化链路列表统计，该拓扑共抽取到 {len(links)} 条链路；"
                    "可通过 links 数组逐条核对两端设备与接口。"
                ),
                "type": "calculation",
                "difficulty": "easy",
            }
        )

    # 3) 最高度设备邻接题
    if graph:
        degree_sorted = sorted(graph.items(), key=lambda kv: (-len(kv[1]), kv[0]))
        center, neighbors = degree_sorted[0]
        uniq_neighbors = sorted(set(neighbors))
        n_txt = "、".join(uniq_neighbors)
        candidates.append(
            {
                "question": f"在{label}的拓扑 {topo_id} 中，设备 {center} 直接连接了哪些设备？",
                "reference": (
                    f"根据链路关系，{center} 直接邻接的设备为：{n_txt}。"
                    "该结论来自以该设备为端点的所有链路条目汇总。"
                ),
                "type": "concept",
                "difficulty": "medium",
            }
        )

    # 4) 主机接入端口题（优先有接口名）
    host_links = []
    for link in links:
        a = link.get("a") or {}
        b = link.get("b") or {}
        da, db = a.get("device"), b.get("device")
        ia, ib = a.get("interface"), b.get("interface")
        if not da or not db:
            continue
        if da.startswith("PC") or "pc" in da.lower():
            host_links.append((da, db, ia, ib))
        elif db.startswith("PC") or "pc" in db.lower():
            host_links.append((db, da, ib, ia))

    if host_links:
        host, peer, host_if, peer_if = sorted(
            host_links,
            key=lambda x: (x[0], x[1], str(x[3] or "")),
        )[0]
        peer_port = f"{peer}.{peer_if}" if peer_if else peer
        candidates.append(
            {
                "question": f"在{label}的拓扑 {topo_id} 中，主机 {host} 是通过哪个对端设备/端口接入网络的？",
                "reference": (
                    f"从链路映射可见，{host} 连接到 {peer_port}。"
                    "若主机链路中断，应优先检查该接入端口及对应物理连线状态。"
                ),
                "type": "troubleshooting",
                "difficulty": "medium",
            }
        )

    # 5) 子网成员题
    valid_subnets = [s for s in subnets if s.get("cidr") and (s.get("members") or [])]
    if valid_subnets:
        subnet = sorted(valid_subnets, key=lambda s: s.get("cidr"))[0]
        members = subnet.get("members") or []
        member_txt = "、".join(_safe_member_text(m) for m in members[:8])
        candidates.append(
            {
                "question": f"在{label}的拓扑 {topo_id} 中，子网 {subnet['cidr']} 的成员有哪些？",
                "reference": (
                    f"根据 subnets 列表，{subnet['cidr']} 的成员包括：{member_txt}。"
                    "成员信息由设备名与接口名（若有）组成。"
                ),
                "type": "concept",
                "difficulty": "easy",
            }
        )

    # 6) 路径排障题（选两台不同主机）
    hosts = sorted([d.get("name") for d in devices if (d.get("type") == "host" and d.get("name"))])
    if len(hosts) >= 2 and graph:
        src, dst = hosts[0], hosts[-1]
        path = _shortest_path(graph, src, dst)
        if len(path) >= 3:
            path_txt = " -> ".join(path)
            candidates.append(
                {
                    "question": f"在{label}的拓扑 {topo_id} 中，若仅按当前链路关系，从 {src} 到 {dst} 的最短设备路径是什么？",
                    "reference": (
                        f"在无权链路假设下，最短路径为：{path_txt}。"
                        "排障时可按该路径逐跳检查链路与接口状态。"
                    ),
                    "type": "troubleshooting",
                    "difficulty": "medium",
                }
            )

    # 7) 路由器互联校验题
    routers = sorted([d.get("name") for d in devices if d.get("type") == "router" and d.get("name")])
    if len(routers) >= 2 and graph:
        r1, r2 = routers[0], routers[1]
        if r2 in graph.get(r1, []):
            i1, i2 = edge_meta.get((r1, r2), (None, None))
            p1 = f"{r1}.{i1}" if i1 else r1
            p2 = f"{r2}.{i2}" if i2 else r2
            candidates.append(
                {
                    "question": f"在{label}的拓扑 {topo_id} 中，若要验证两台路由器互联是否正常，应优先核查哪对接口？",
                    "reference": (
                        f"该拓扑里两台路由器直接通过 {p1} 与 {p2} 相连。"
                        "进行互联验证时，应优先检查这对接口的物理连通与三层配置一致性。"
                    ),
                    "type": "config",
                    "difficulty": "medium",
                }
            )

    output = []
    for row in candidates:
        output.append(
            {
                "question": row["question"],
                "reference": row["reference"],
                "experiment_id": experiment_id,
                "type": row["type"],
                "difficulty": row["difficulty"],
                "requires_topology": True,
                "source": source_rel,
            }
        )
    return output


def pick_diverse_questions(candidates: List[dict], max_n: int) -> List[dict]:
    if len(candidates) <= max_n:
        return candidates
    by_type: Dict[str, List[dict]] = defaultdict(list)
    for row in candidates:
        by_type[row.get("type", "concept")].append(row)

    picked: List[dict] = []
    for t in _TYPE_ORDER:
        if by_type.get(t):
            picked.append(by_type[t][0])
        if len(picked) >= max_n:
            return picked[:max_n]

    # 补齐
    flat = []
    for rows in by_type.values():
        flat.extend(rows)
    for row in flat:
        if row in picked:
            continue
        picked.append(row)
        if len(picked) >= max_n:
            break
    return picked[:max_n]


def next_tq_id(existing: List[dict]) -> int:
    max_num = 0
    for row in existing:
        rid = str(row.get("id", ""))
        m = re.search(r"(\d+)$", rid)
        if m:
            max_num = max(max_num, int(m.group(1)))
    return max_num + 1


def format_tq_id(n: int) -> str:
    return f"TQ{n:02d}" if n < 100 else f"TQ{n}"


def main() -> None:
    parser = argparse.ArgumentParser(description="自动扩充拓扑题库（跨实验）")
    parser.add_argument("--topo-root", type=str, default=str(DEFAULT_TOPO_ROOT), help="topo_store 路径")
    parser.add_argument("--bank", type=str, default=str(DEFAULT_BANK), help="现有题库路径")
    parser.add_argument("--output", type=str, default=str(DEFAULT_BANK), help="输出题库路径")
    parser.add_argument("--max-new-per-topology", type=int, default=4, help="每个拓扑最多新增题数")
    parser.add_argument("--min-reference-len", type=int, default=20, help="参考答案最短长度")
    args = parser.parse_args()

    topo_root = _resolve(args.topo_root)
    bank_path = _resolve(args.bank)
    output_path = _resolve(args.output)

    if not topo_root.exists():
        raise FileNotFoundError(f"找不到 topo_store：{topo_root}")
    if not bank_path.exists():
        raise FileNotFoundError(f"找不到题库文件：{bank_path}")

    bank = json.loads(bank_path.read_text(encoding="utf-8"))
    existing_keys = {_canonical(str(x.get("question", ""))) for x in bank}

    topo_files = sorted(topo_root.glob("*/approved_json/*.json"))
    if not topo_files:
        print("未找到任何 approved_json 文件。")
        return

    next_id = next_tq_id(bank)
    new_rows: List[dict] = []
    coverage_counter = Counter()

    for topo_file in topo_files:
        experiment_id = topo_file.parent.parent.name
        source_rel = str(topo_file.relative_to(ROOT))
        topo_id = topo_file.stem

        try:
            topo_data = json.loads(topo_file.read_text(encoding="utf-8"))
        except Exception:
            continue

        candidates = generate_candidates(
            experiment_id=experiment_id,
            source_rel=source_rel,
            topo_id=topo_id,
            topo_data=topo_data,
        )
        selected = pick_diverse_questions(candidates, max_n=max(0, args.max_new_per_topology))

        added_this_topo = 0
        for row in selected:
            key = _canonical(row["question"])
            if not key or key in existing_keys:
                continue
            if len(str(row.get("reference", "")).strip()) < args.min_reference_len:
                continue

            row = dict(row)
            row["id"] = format_tq_id(next_id)
            next_id += 1
            new_rows.append(row)
            existing_keys.add(key)
            coverage_counter[experiment_id] += 1
            added_this_topo += 1

        if added_this_topo:
            pass

    merged = list(bank) + new_rows
    # 按 id 数值排序，便于审阅
    def _id_num(row: dict) -> int:
        m = re.search(r"(\d+)$", str(row.get("id", "")))
        return int(m.group(1)) if m else 10**9

    merged = sorted(merged, key=_id_num)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")

    labs = sorted({p.parent.parent.name for p in topo_files})
    print(f"扫描拓扑文件：{len(topo_files)}")
    print(f"覆盖实验：{', '.join(labs)}")
    print(f"原题数：{len(bank)}")
    print(f"新增题数：{len(new_rows)}")
    print(f"合并后题数：{len(merged)}")
    print(f"输出：{output_path}")
    if coverage_counter:
        print("各实验新增题量：")
        for lab, cnt in sorted(coverage_counter.items()):
            print(f"  - {lab}: {cnt}")


if __name__ == "__main__":
    main()
