import { useState } from "react"
import { ChevronUp, LogOut, ArchiveRestore, Trash2, ChevronDown, ChevronRight } from "lucide-react"
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover"
import { Button } from "@/components/ui/button"
import { Separator } from "@/components/ui/separator"
import type { AuthUser, SessionMeta } from "@/lib/api"

interface Props {
  user: AuthUser
  archived: SessionMeta[]
  onLogout: () => void
  onSelectArchived: (id: string) => void
  onUnarchive: (id: string) => void
  onDeleteArchived: (id: string) => void
}

export default function UserCard({
  user,
  archived,
  onLogout,
  onSelectArchived,
  onUnarchive,
  onDeleteArchived,
}: Props) {
  const initial = (user.username?.[0] || "U").toUpperCase()
  const className = (user.profile?.class_name || "").trim() || "未填写班级"
  const [archOpen, setArchOpen] = useState(false)

  return (
    <div className="border-t border-[hsl(var(--border))] bg-white/97 backdrop-blur-sm px-2 py-2">
      <Popover>
        <PopoverTrigger asChild>
          <button className="w-full flex items-center gap-2.5 rounded-xl border border-[hsl(var(--border))] bg-gradient-to-b from-white to-[#FAF5F3] px-2.5 py-2 hover:from-[#FFF9F7] hover:to-[#F4E7E3] hover:border-[#D0BFBB] transition-colors text-left">
            <div className="shrink-0 w-8 h-8 rounded-full bg-gradient-to-b from-[#A42A1E] to-[#861A11] text-white font-bold text-sm grid place-items-center shadow-sm">
              {initial}
            </div>
            <div className="flex-1 min-w-0 leading-tight">
              <div className="font-bold text-sm text-[hsl(var(--primary))] truncate">{user.username}</div>
              <div className="text-xs text-[hsl(var(--muted-foreground))] truncate">{className}</div>
            </div>
            <ChevronUp className="w-4 h-4 text-[hsl(var(--muted-foreground))] shrink-0" />
          </button>
        </PopoverTrigger>
        <PopoverContent
          align="start"
          side="top"
          sideOffset={8}
          className="w-[var(--radix-popover-trigger-width)] p-2"
        >
          <div className="text-xs text-[hsl(var(--muted-foreground))] px-2 py-1.5">
            会话历史与状态以后端数据库为准
          </div>
          <Separator className="my-1" />

          {/* 已归档折叠 */}
          <button
            type="button"
            className="w-full flex items-center justify-between px-2 py-1.5 text-sm rounded-md hover:bg-[hsl(var(--accent))]"
            onClick={() => setArchOpen((v) => !v)}
          >
            <span className="flex items-center gap-2">
              {archOpen ? <ChevronDown className="w-3.5 h-3.5" /> : <ChevronRight className="w-3.5 h-3.5" />}
              已归档（{archived.length}）
            </span>
          </button>
          {archOpen && (
            <div className="max-h-[40vh] overflow-y-auto px-1 py-1">
              {archived.length === 0 && (
                <div className="text-xs text-[hsl(var(--muted-foreground))] px-2 py-1.5">
                  暂无归档会话
                </div>
              )}
              {archived.map((s) => (
                <div
                  key={s.session_id}
                  className="group flex items-center gap-1 rounded-md hover:bg-[hsl(var(--accent))]"
                >
                  <button
                    className="flex-1 min-w-0 text-left px-2 py-1 text-xs truncate"
                    onClick={() => onSelectArchived(s.session_id)}
                    title={s.title}
                  >
                    {s.title || s.session_id}
                  </button>
                  <button
                    className="shrink-0 p-1 text-[hsl(var(--muted-foreground))] hover:text-[hsl(var(--primary))]"
                    onClick={() => onUnarchive(s.session_id)}
                    title="恢复"
                  >
                    <ArchiveRestore className="w-3.5 h-3.5" />
                  </button>
                  <button
                    className="shrink-0 p-1 text-[hsl(var(--muted-foreground))] hover:text-[hsl(var(--destructive))]"
                    onClick={() => onDeleteArchived(s.session_id)}
                    title="删除"
                  >
                    <Trash2 className="w-3.5 h-3.5" />
                  </button>
                </div>
              ))}
            </div>
          )}

          <Separator className="my-1" />
          <Button
            variant="ghost"
            size="sm"
            className="w-full justify-start gap-2"
            onClick={onLogout}
          >
            <LogOut className="w-4 h-4" />
            退出登录
          </Button>
        </PopoverContent>
      </Popover>
    </div>
  )
}
