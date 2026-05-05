import { useState } from "react"
import {
  ChevronDown,
  ChevronRight,
  BookOpen,
  Network,
  Globe,
  Wrench,
} from "lucide-react"
import { cn } from "@/lib/utils"
import type { ToolTrace } from "@/lib/api"

interface Props {
  traces?: ToolTrace[]
}

const TOOL_META: Record<
  string,
  { label: string; icon: React.ReactNode }
> = {
  retrieve_course_docs: {
    label: "课程文档检索",
    icon: <BookOpen className="w-3.5 h-3.5" />,
  },
  rag_retrieve: {
    label: "课程文档检索",
    icon: <BookOpen className="w-3.5 h-3.5" />,
  },
  retrieve: {
    label: "课程文档检索",
    icon: <BookOpen className="w-3.5 h-3.5" />,
  },
  get_topology_context: {
    label: "实验拓扑",
    icon: <Network className="w-3.5 h-3.5" />,
  },
  topology: {
    label: "实验拓扑",
    icon: <Network className="w-3.5 h-3.5" />,
  },
  web_search: {
    label: "联网搜索",
    icon: <Globe className="w-3.5 h-3.5" />,
  },
  websearch: {
    label: "联网搜索",
    icon: <Globe className="w-3.5 h-3.5" />,
  },
}

function metaFor(tool?: string) {
  if (!tool) return { label: "工具", icon: <Wrench className="w-3.5 h-3.5" /> }
  return (
    TOOL_META[tool] || {
      label: tool,
      icon: <Wrench className="w-3.5 h-3.5" />,
    }
  )
}

function truncate(s: string | undefined, max: number) {
  if (!s) return ""
  const trimmed = s.trim()
  return trimmed.length > max ? trimmed.slice(0, max) + "…" : trimmed
}

export default function TraceList({ traces }: Props) {
  const [open, setOpen] = useState(false)
  if (!traces || traces.length === 0) return null

  return (
    <div className="mt-1 px-1 max-w-[82%]">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="flex items-center gap-1 text-xs text-[hsl(var(--muted-foreground))] hover:text-[hsl(var(--primary))]"
        aria-expanded={open}
      >
        {open ? (
          <ChevronDown className="w-3.5 h-3.5" />
        ) : (
          <ChevronRight className="w-3.5 h-3.5" />
        )}
        <Wrench className="w-3.5 h-3.5" />
        <span>调用了 {traces.length} 个工具</span>
        <span className="opacity-70">（点击展开依据）</span>
      </button>
      {open && (
        <ol className="mt-1.5 space-y-1.5 border-l-2 border-[hsl(var(--border))] pl-3">
          {traces.map((t, i) => {
            const m = metaFor(t.tool)
            return (
              <li key={i} className="text-xs">
                <div className="flex items-center gap-1.5 font-semibold text-[hsl(var(--ink-700,#4D3D3A))]">
                  <span className="text-[hsl(var(--primary))]">{m.icon}</span>
                  <span>{m.label}</span>
                  <span className="font-normal text-[hsl(var(--muted-foreground))]">
                    #{i + 1}
                  </span>
                </div>
                {t.input && (
                  <div className="mt-0.5 text-[hsl(var(--muted-foreground))]">
                    <span className="font-medium">输入：</span>
                    <span
                      className={cn(
                        "ml-1 font-mono text-[11px] break-words",
                      )}
                    >
                      {truncate(t.input, 160)}
                    </span>
                  </div>
                )}
                {t.output && (
                  <details className="mt-0.5">
                    <summary className="cursor-pointer text-[hsl(var(--muted-foreground))] hover:text-[hsl(var(--primary))]">
                      <span className="font-medium">输出：</span>
                      <span className="ml-1">{truncate(t.output, 60)}</span>
                    </summary>
                    <pre className="mt-1 p-2 rounded-md bg-[hsl(var(--muted))] text-[11px] font-mono overflow-x-auto whitespace-pre-wrap break-words max-h-64">
                      {t.output}
                    </pre>
                  </details>
                )}
              </li>
            )
          })}
        </ol>
      )}
    </div>
  )
}
