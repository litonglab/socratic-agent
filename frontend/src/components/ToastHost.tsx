import { useEffect, useState } from "react"
import { CheckCircle2, AlertCircle, Info, X } from "lucide-react"
import { cn } from "@/lib/utils"
import {
  type Toast,
  type ToastType,
  dismissToast,
  getToastSnapshot,
  subscribeToasts,
} from "@/lib/toast"

const TYPE_STYLES: Record<ToastType, { wrap: string; icon: React.ReactNode }> = {
  info: {
    wrap: "border-[hsl(var(--border))] bg-white text-[hsl(var(--ink-700,#4D3D3A))]",
    icon: <Info className="w-4 h-4 text-[hsl(var(--primary))]" />,
  },
  success: {
    wrap: "border-emerald-200 bg-emerald-50 text-emerald-900",
    icon: <CheckCircle2 className="w-4 h-4 text-emerald-600" />,
  },
  error: {
    wrap: "border-red-200 bg-red-50 text-red-900",
    icon: <AlertCircle className="w-4 h-4 text-red-600" />,
  },
}

export default function ToastHost() {
  // 用初始化函数同步外部模块状态，避免 effect 内 setState 误触 lint
  const [list, setList] = useState<Toast[]>(() => getToastSnapshot())

  useEffect(() => subscribeToasts(setList), [])

  if (list.length === 0) return null

  return (
    <div
      className="pointer-events-none fixed top-4 right-4 z-[100] flex w-[min(320px,calc(100vw-2rem))] flex-col gap-2"
      role="region"
      aria-label="通知"
    >
      {list.map((t) => {
        const s = TYPE_STYLES[t.type]
        return (
          <div
            key={t.id}
            className={cn(
              "pointer-events-auto flex items-start gap-2 rounded-xl border px-3 py-2 text-sm shadow-lg",
              s.wrap,
            )}
            role={t.type === "error" ? "alert" : "status"}
          >
            <span className="mt-0.5 shrink-0">{s.icon}</span>
            <span className="flex-1 break-words leading-snug">{t.text}</span>
            <button
              type="button"
              onClick={() => dismissToast(t.id)}
              className="shrink-0 -m-1 rounded-md p-1 opacity-60 hover:opacity-100"
              aria-label="关闭通知"
            >
              <X className="w-3.5 h-3.5" />
            </button>
          </div>
        )
      })}
    </div>
  )
}
