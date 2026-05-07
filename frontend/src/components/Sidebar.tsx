import { useMemo, useState, useRef, useEffect } from "react"
import {
  Edit3,
  MoreHorizontal,
  Archive,
  Trash2,
  PanelLeftClose,
  Search,
  Pencil,
  X as XIcon,
  Check,
  Loader2,
} from "lucide-react"
import { Button } from "@/components/ui/button"
import { Separator } from "@/components/ui/separator"
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover"
import { cn } from "@/lib/utils"
import { groupSessionsByUpdatedAt } from "@/lib/sessionGrouping"
import type { AuthUser, SessionMeta } from "@/lib/api"
import UserCard from "./UserCard"
import BrandLogo from "./BrandLogo"

interface Props {
  user: AuthUser
  sessions: SessionMeta[]
  activeId: string | null
  /** 哪些 sid 正在流式生成 → 该行右侧显示转圈 */
  streamingSessions?: ReadonlySet<string>
  /** 哪些 sid 已完成但用户未查看 → 该行右侧显示红点 */
  unseenSessions?: ReadonlySet<string>
  onNew: () => void
  onSelect: (id: string) => void
  onDelete: (id: string) => void
  onArchive: (id: string) => void
  onUnarchive: (id: string) => void
  onRename: (id: string, title: string) => void | Promise<void>
  onLogout: () => void
  onCollapse?: () => void
  /** 由父组件持有的 ref，用于实现 Cmd/Ctrl+K 聚焦搜索 */
  searchInputRef?: React.RefObject<HTMLInputElement | null>
}

export default function Sidebar(props: Props) {
  const {
    user,
    sessions,
    activeId,
    streamingSessions,
    unseenSessions,
    onNew,
    onSelect,
    onDelete,
    onArchive,
    onUnarchive,
    onRename,
    onLogout,
    onCollapse,
    searchInputRef,
  } = props

  const [query, setQuery] = useState("")
  const [renamingId, setRenamingId] = useState<string | null>(null)

  const visible = useMemo(() => sessions.filter((s) => !s.archived), [sessions])
  const archived = useMemo(() => sessions.filter((s) => !!s.archived), [sessions])

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase()
    if (!q) return visible
    return visible.filter((s) => (s.title || "").toLowerCase().includes(q))
  }, [visible, query])

  const groups = useMemo(() => groupSessionsByUpdatedAt(filtered), [filtered])

  return (
    <aside className="h-full flex flex-col bg-white/96 border-r border-[hsl(var(--border))]">
      <div className="flex items-center gap-3 px-4 py-3 border-b border-[hsl(var(--border))]">
        <BrandLogo size={36} />
        <div className="font-bold text-lg text-[hsl(var(--primary))] flex-1 min-w-0 truncate">
          NetRUC Agent
        </div>
        {onCollapse && (
          <button
            type="button"
            onClick={onCollapse}
            className="grid place-items-center w-8 h-8 rounded-md text-[hsl(var(--muted-foreground))] hover:bg-[hsl(var(--accent))] hover:text-[hsl(var(--primary))]"
            title="收起侧栏"
            aria-label="收起侧栏"
          >
            <PanelLeftClose className="w-4 h-4" />
          </button>
        )}
      </div>

      <div className="px-3 py-2.5 space-y-2">
        <Button variant="outline" size="sm" className="w-full justify-start gap-2" onClick={onNew}>
          <Edit3 className="w-4 h-4" />
          新建会话
        </Button>
        <div className="relative">
          <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-[hsl(var(--muted-foreground))]" />
          <input
            ref={searchInputRef}
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="搜索会话标题…   ⌘K"
            className="w-full h-8 pl-8 pr-7 rounded-md bg-white border border-[hsl(var(--border))] text-sm outline-none focus:border-[hsl(var(--primary))] focus:ring-1 focus:ring-[hsl(var(--primary)/0.25)]"
            aria-label="搜索会话"
          />
          {query && (
            <button
              type="button"
              onClick={() => setQuery("")}
              className="absolute right-1.5 top-1/2 -translate-y-1/2 text-[hsl(var(--muted-foreground))] hover:text-[hsl(var(--primary))]"
              aria-label="清空搜索"
            >
              <XIcon className="w-3.5 h-3.5" />
            </button>
          )}
        </div>
      </div>

      <Separator />

      <div className="flex-1 min-h-0 overflow-y-auto px-2 py-2">
        {visible.length === 0 && (
          <div className="text-xs text-[hsl(var(--muted-foreground))] px-2 py-2">
            暂无会话，点击上方"新建会话"开始
          </div>
        )}
        {visible.length > 0 && groups.length === 0 && (
          <div className="text-xs text-[hsl(var(--muted-foreground))] px-2 py-2">
            没有匹配"{query}"的会话
          </div>
        )}
        {groups.map((g) => (
          <div key={g.key} className="mb-2">
            <div className="text-[11px] font-semibold text-[hsl(var(--muted-foreground))] uppercase tracking-wide px-2 mb-1">
              {g.label}
            </div>
            <div className="space-y-0.5">
              {g.items.map((s) => (
                <SessionRow
                  key={s.session_id}
                  s={s}
                  active={s.session_id === activeId}
                  streaming={streamingSessions?.has(s.session_id) ?? false}
                  unseen={unseenSessions?.has(s.session_id) ?? false}
                  renaming={renamingId === s.session_id}
                  onSelect={() => onSelect(s.session_id)}
                  onArchive={() => onArchive(s.session_id)}
                  onDelete={() => onDelete(s.session_id)}
                  onStartRename={() => setRenamingId(s.session_id)}
                  onCancelRename={() => setRenamingId(null)}
                  onCommitRename={async (title) => {
                    setRenamingId(null)
                    await onRename(s.session_id, title)
                  }}
                />
              ))}
            </div>
          </div>
        ))}
      </div>

      <UserCard
        user={user}
        archived={archived}
        onLogout={onLogout}
        onSelectArchived={onSelect}
        onUnarchive={onUnarchive}
        onDeleteArchived={onDelete}
      />
    </aside>
  )
}

interface RowProps {
  s: SessionMeta
  active: boolean
  /** 该会话正在后台流式生成：右侧显示小转圈 */
  streaming?: boolean
  /** 该会话已完成但用户未查看：右侧显示红点 */
  unseen?: boolean
  renaming: boolean
  onSelect: () => void
  onArchive: () => void
  onDelete: () => void
  onStartRename: () => void
  onCancelRename: () => void
  onCommitRename: (title: string) => void | Promise<void>
}

function RenameInput({
  initial,
  active,
  onCommit,
  onCancel,
}: {
  initial: string
  active: boolean
  onCommit: (v: string) => void | Promise<void>
  onCancel: () => void
}) {
  const [draft, setDraft] = useState(() => initial)
  const inputRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    inputRef.current?.focus()
    inputRef.current?.select()
  }, [])

  return (
    <div
      className={cn(
        "flex items-center gap-1 rounded-lg px-1 py-0.5",
        active ? "bg-[hsl(var(--primary)/0.10)]" : "bg-[hsl(var(--accent))]",
      )}
    >
      <input
        ref={inputRef}
        value={draft}
        onChange={(e) => setDraft(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter") {
            e.preventDefault()
            const v = draft.trim()
            if (v) void onCommit(v)
            else onCancel()
          } else if (e.key === "Escape") {
            e.preventDefault()
            onCancel()
          }
        }}
        maxLength={80}
        className="flex-1 min-w-0 text-sm bg-white border border-[hsl(var(--primary)/0.4)] rounded px-2 py-1 outline-none focus:ring-1 focus:ring-[hsl(var(--primary)/0.3)]"
        aria-label="重命名会话"
      />
      <button
        type="button"
        className="shrink-0 grid place-items-center w-6 h-6 rounded text-emerald-700 hover:bg-emerald-50"
        onClick={() => {
          const v = draft.trim()
          if (v) void onCommit(v)
          else onCancel()
        }}
        title="保存"
        aria-label="保存"
      >
        <Check className="w-3.5 h-3.5" />
      </button>
      <button
        type="button"
        className="shrink-0 grid place-items-center w-6 h-6 rounded text-[hsl(var(--muted-foreground))] hover:bg-[hsl(var(--accent))]"
        onClick={onCancel}
        title="取消"
        aria-label="取消"
      >
        <XIcon className="w-3.5 h-3.5" />
      </button>
    </div>
  )
}

function SessionRow({
  s,
  active,
  streaming = false,
  unseen = false,
  renaming,
  onSelect,
  onArchive,
  onDelete,
  onStartRename,
  onCancelRename,
  onCommitRename,
}: RowProps) {
  if (renaming) {
    return (
      <RenameInput
        initial={s.title || ""}
        active={active}
        onCommit={onCommitRename}
        onCancel={onCancelRename}
      />
    )
  }

  return (
    <div
      className={cn(
        "group flex items-center gap-1 rounded-lg px-1 transition-colors",
        active ? "bg-[hsl(var(--primary)/0.10)]" : "hover:bg-[hsl(var(--accent))]",
      )}
    >
      <button
        className={cn(
          "flex-1 min-w-0 text-left px-2 py-1.5 truncate text-sm",
          active
            ? "text-[hsl(var(--primary))] font-bold"
            : "text-[hsl(var(--ink-900,#2A1F1D))]",
        )}
        onClick={onSelect}
        onDoubleClick={onStartRename}
        title={s.title}
      >
        {s.title || "新会话"}
      </button>
      {/* 状态指示：streaming 优先展示转圈；否则若 unseen 显示红点。
          位置在更多操作按钮之前；hover 时更多按钮会与之并存。 */}
      {streaming ? (
        <span
          className="shrink-0 grid place-items-center w-6 h-6 text-[hsl(var(--primary))]"
          title="正在生成中…"
          aria-label="正在生成中"
        >
          <Loader2 className="w-3.5 h-3.5 animate-spin" />
        </span>
      ) : unseen ? (
        <span
          className="shrink-0 grid place-items-center w-6 h-6"
          title="已完成生成，点击查看"
          aria-label="未读：已完成生成"
        >
          <span className="block w-2 h-2 rounded-full bg-[hsl(var(--primary))]" />
        </span>
      ) : null}
      <Popover>
        <PopoverTrigger asChild>
          <button
            className={cn(
              "shrink-0 grid place-items-center w-7 h-7 rounded-md text-[hsl(var(--muted-foreground))] transition-opacity hover:bg-[hsl(var(--primary)/0.10)] hover:text-[hsl(var(--primary))]",
              "opacity-0 group-hover:opacity-100 group-focus-within:opacity-100 focus-visible:opacity-100 data-[state=open]:opacity-100",
            )}
            aria-label="更多操作"
          >
            <MoreHorizontal className="w-4 h-4" />
          </button>
        </PopoverTrigger>
        <PopoverContent align="end" side="bottom" sideOffset={4} className="w-36">
          <button
            className="w-full flex items-center gap-2 px-2.5 py-1.5 text-sm rounded-md hover:bg-[hsl(var(--accent))]"
            onClick={onStartRename}
          >
            <Pencil className="w-4 h-4" />
            重命名
          </button>
          <button
            className="w-full flex items-center gap-2 px-2.5 py-1.5 text-sm rounded-md hover:bg-[hsl(var(--accent))]"
            onClick={onArchive}
          >
            <Archive className="w-4 h-4" />
            归档
          </button>
          <button
            className="w-full flex items-center gap-2 px-2.5 py-1.5 text-sm rounded-md hover:bg-[hsl(var(--accent))] text-[hsl(var(--destructive))]"
            onClick={onDelete}
          >
            <Trash2 className="w-4 h-4" />
            删除
          </button>
        </PopoverContent>
      </Popover>
    </div>
  )
}
