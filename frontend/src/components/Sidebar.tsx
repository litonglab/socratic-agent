import { Edit3, MoreHorizontal, Archive, Trash2 } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Separator } from "@/components/ui/separator"
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover"
import { cn } from "@/lib/utils"
import type { AuthUser, SessionMeta } from "@/lib/api"
import UserCard from "./UserCard"

interface Props {
  user: AuthUser
  sessions: SessionMeta[]
  activeId: string | null
  onNew: () => void
  onSelect: (id: string) => void
  onDelete: (id: string) => void
  onArchive: (id: string) => void
  onUnarchive: (id: string) => void
  onLogout: () => void
}

export default function Sidebar(props: Props) {
  const { user, sessions, activeId, onNew, onSelect, onDelete, onArchive, onUnarchive, onLogout } =
    props
  const visible = sessions.filter((s) => !s.archived)
  const archived = sessions.filter((s) => !!s.archived)

  return (
    <aside className="h-full flex flex-col bg-white/96 border-r border-[hsl(var(--border))]">
      {/* 顶部 logo */}
      <div className="flex items-center gap-3 px-4 py-3 border-b border-[hsl(var(--border))]">
        <div className="w-9 h-9 rounded-xl bg-gradient-to-b from-[#A42A1E] to-[#861A11] text-white font-extrabold text-lg grid place-items-center shadow-sm">
          N
        </div>
        <div className="font-bold text-lg text-[hsl(var(--primary))]">NetRUC Agent</div>
      </div>

      {/* 新建会话 */}
      <div className="px-3 py-2.5">
        <Button variant="outline" size="sm" className="w-full justify-start gap-2" onClick={onNew}>
          <Edit3 className="w-4 h-4" />
          新建会话
        </Button>
      </div>

      <Separator />

      {/* 会话列表（撑满 + 滚动） */}
      <div className="flex-1 min-h-0 overflow-y-auto px-2 py-2">
        <div className="text-xs font-bold text-[hsl(var(--primary))] px-2 mb-1">◌ 会话列表</div>
        {visible.length === 0 && (
          <div className="text-xs text-[hsl(var(--muted-foreground))] px-2 py-2">
            暂无会话，点击上方"新建会话"开始
          </div>
        )}
        <div className="space-y-0.5">
          {visible.map((s) => {
            const isActive = s.session_id === activeId
            return (
              <div
                key={s.session_id}
                className={cn(
                  "group flex items-center gap-1 rounded-lg px-1 transition-colors",
                  isActive ? "bg-[hsl(var(--primary)/0.10)]" : "hover:bg-[hsl(var(--accent))]",
                )}
              >
                <button
                  className={cn(
                    "flex-1 min-w-0 text-left px-2 py-1.5 truncate text-sm",
                    isActive
                      ? "text-[hsl(var(--primary))] font-bold"
                      : "text-[hsl(var(--ink-900,#2A1F1D))]",
                  )}
                  onClick={() => onSelect(s.session_id)}
                  title={s.title}
                >
                  {s.title || "新会话"}
                </button>
                <Popover>
                  <PopoverTrigger asChild>
                    <button
                      className={cn(
                        "shrink-0 grid place-items-center w-7 h-7 rounded-md text-[hsl(var(--muted-foreground))] transition-opacity hover:bg-[hsl(var(--primary)/0.10)] hover:text-[hsl(var(--primary))]",
                        "opacity-0 group-hover:opacity-100 data-[state=open]:opacity-100",
                      )}
                      aria-label="更多操作"
                    >
                      <MoreHorizontal className="w-4 h-4" />
                    </button>
                  </PopoverTrigger>
                  <PopoverContent align="end" side="bottom" sideOffset={4} className="w-36">
                    <button
                      className="w-full flex items-center gap-2 px-2.5 py-1.5 text-sm rounded-md hover:bg-[hsl(var(--accent))]"
                      onClick={() => onArchive(s.session_id)}
                    >
                      <Archive className="w-4 h-4" />
                      归档
                    </button>
                    <button
                      className="w-full flex items-center gap-2 px-2.5 py-1.5 text-sm rounded-md hover:bg-[hsl(var(--accent))] text-[hsl(var(--destructive))]"
                      onClick={() => onDelete(s.session_id)}
                    >
                      <Trash2 className="w-4 h-4" />
                      删除
                    </button>
                  </PopoverContent>
                </Popover>
              </div>
            )
          })}
        </div>
      </div>

      {/* 用户卡片：固定在 sidebar 底部 */}
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
