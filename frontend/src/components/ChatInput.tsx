import { useState, useRef, useEffect } from "react"
import { Send, Paperclip, Loader2, Globe, X, Square } from "lucide-react"
import { Button } from "@/components/ui/button"
import { cn } from "@/lib/utils"

export interface AttachedImage {
  id: string
  base64: string // pure base64 (no data:)
  mime: string
  dataUrl: string // for preview
}

interface Props {
  disabled?: boolean
  /** 是否处于流式输出中：true 时把发送按钮替换为"停止" */
  streaming?: boolean
  websearch: boolean
  onWebsearchChange: (v: boolean) => void
  onSend: (text: string, images: AttachedImage[]) => void
  onStop?: () => void
}

const MAX_IMAGES = 4
const MAX_BYTES = 8 * 1024 * 1024 // 8MB

export default function ChatInput({
  disabled,
  streaming,
  websearch,
  onWebsearchChange,
  onSend,
  onStop,
}: Props) {
  const [text, setText] = useState("")
  const [images, setImages] = useState<AttachedImage[]>([])
  const [error, setError] = useState<string | null>(null)
  const taRef = useRef<HTMLTextAreaElement>(null)
  const fileRef = useRef<HTMLInputElement>(null)

  // auto-resize textarea
  useEffect(() => {
    const ta = taRef.current
    if (!ta) return
    ta.style.height = "auto"
    ta.style.height = Math.min(ta.scrollHeight, 200) + "px"
  }, [text])

  function submit() {
    const v = text.trim()
    if ((!v && images.length === 0) || disabled) return
    onSend(v, images)
    setText("")
    setImages([])
    setError(null)
  }

  function handleKey(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
      e.preventDefault()
      submit()
    }
  }

  async function addFiles(files: FileList | File[]) {
    const arr = Array.from(files).filter((f) => f.type.startsWith("image/"))
    if (arr.length === 0) return
    const next: AttachedImage[] = []
    for (const f of arr) {
      if (images.length + next.length >= MAX_IMAGES) {
        setError(`最多上传 ${MAX_IMAGES} 张图片`)
        break
      }
      if (f.size > MAX_BYTES) {
        setError(`图片 ${f.name} 超过 ${(MAX_BYTES / 1024 / 1024).toFixed(0)}MB`)
        continue
      }
      const dataUrl = await readAsDataURL(f)
      const base64 = dataUrl.split(",", 2)[1] || ""
      next.push({
        id: Math.random().toString(36).slice(2),
        base64,
        mime: f.type || "image/png",
        dataUrl,
      })
    }
    if (next.length) setImages((prev) => [...prev, ...next])
  }

  function onPaste(e: React.ClipboardEvent) {
    if (!e.clipboardData?.files?.length) return
    const imgs = Array.from(e.clipboardData.files).filter((f) => f.type.startsWith("image/"))
    if (imgs.length) {
      e.preventDefault()
      void addFiles(imgs)
    }
  }

  function removeImage(id: string) {
    setImages((prev) => prev.filter((x) => x.id !== id))
  }

  return (
    <div className="px-6 pb-6">
      <div className="max-w-3xl mx-auto">
        {/* 错误提示 */}
        {error && (
          <div className="text-xs text-[hsl(var(--destructive))] bg-red-50 border border-red-200 rounded-md px-2 py-1 mb-1.5">
            {error}
          </div>
        )}

        {/* 输入框容器 */}
        <div className="rounded-2xl border border-[hsl(var(--border))] bg-white shadow-[0_4px_14px_rgba(38,22,19,0.06)] focus-within:border-[hsl(var(--primary))] focus-within:ring-2 focus-within:ring-[hsl(var(--primary)/0.18)]">
          {/* 缩略图 */}
          {images.length > 0 && (
            <div className="flex flex-wrap gap-2 px-3 pt-2.5">
              {images.map((img) => (
                <div key={img.id} className="relative group">
                  <img
                    src={img.dataUrl}
                    alt="preview"
                    className="w-16 h-16 object-cover rounded-md border border-[hsl(var(--border))]"
                  />
                  <button
                    type="button"
                    onClick={() => removeImage(img.id)}
                    className="absolute -top-1.5 -right-1.5 w-4 h-4 rounded-full bg-[hsl(var(--destructive))] text-white grid place-items-center opacity-90 hover:opacity-100"
                    aria-label="删除图片"
                  >
                    <X className="w-2.5 h-2.5" />
                  </button>
                </div>
              ))}
            </div>
          )}

          {/* textarea 独占一行 */}
          <textarea
            ref={taRef}
            value={text}
            onChange={(e) => setText(e.target.value)}
            onKeyDown={handleKey}
            onPaste={onPaste}
            placeholder="输入你的问题，例如：我对子网划分感到困惑（可粘贴图片）"
            rows={1}
            className="block w-full resize-none bg-transparent outline-none text-sm leading-relaxed max-h-[200px] px-3.5 pt-3 pb-1"
          />

          {/* 工具栏：左侧 Paperclip + 联网搜索 chip，右侧 Send */}
          <div className="flex items-center gap-1.5 px-2 pb-2">
            <button
              type="button"
              onClick={() => fileRef.current?.click()}
              className="shrink-0 w-8 h-8 grid place-items-center rounded-lg text-[hsl(var(--muted-foreground))] hover:bg-[hsl(var(--accent))] hover:text-[hsl(var(--primary))]"
              title="上传图片（也可粘贴）"
            >
              <Paperclip className="w-4 h-4" />
            </button>
            <input
              ref={fileRef}
              type="file"
              accept="image/*"
              multiple
              hidden
              onChange={(e) => {
                if (e.target.files) void addFiles(e.target.files)
                e.target.value = ""
              }}
            />
            <button
              type="button"
              role="switch"
              aria-checked={websearch}
              onClick={() => onWebsearchChange(!websearch)}
              className={cn(
                "inline-flex items-center gap-1.5 text-xs px-2.5 py-1 rounded-full border transition-colors",
                websearch
                  ? "bg-[hsl(var(--primary)/0.08)] border-[hsl(var(--primary)/0.3)] text-[hsl(var(--primary))]"
                  : "bg-[hsl(var(--accent))]/40 border-transparent text-[hsl(var(--muted-foreground))] hover:bg-[hsl(var(--accent))]",
              )}
              title={websearch ? "联网搜索：已开启" : "联网搜索：未开启"}
            >
              <Globe className="w-3.5 h-3.5" />
              联网搜索
              <span
                aria-hidden="true"
                className={cn(
                  "ml-0.5 inline-block w-7 h-3.5 rounded-full transition-colors relative",
                  websearch ? "bg-[hsl(var(--primary))]" : "bg-[hsl(var(--border))]",
                )}
              >
                <span
                  className={cn(
                    "absolute top-0.5 left-0.5 w-2.5 h-2.5 rounded-full bg-white transition-transform",
                    websearch && "translate-x-3.5",
                  )}
                />
              </span>
            </button>
            <div className="flex-1" />
            {streaming && onStop ? (
              <Button
                size="icon"
                onClick={onStop}
                variant="outline"
                className="shrink-0 w-9 h-9 rounded-xl border-[hsl(var(--primary)/0.4)] text-[hsl(var(--primary))] hover:bg-[hsl(var(--primary)/0.08)]"
                title="停止生成"
                aria-label="停止生成"
              >
                <Square className="w-4 h-4 fill-current" />
              </Button>
            ) : (
              <Button
                size="icon"
                onClick={submit}
                disabled={disabled || (!text.trim() && images.length === 0)}
                className="shrink-0 w-9 h-9 rounded-xl"
                title="发送 (Enter)"
                aria-label="发送"
              >
                {disabled ? (
                  <Loader2 className="w-4 h-4 animate-spin" />
                ) : (
                  <Send className="w-4 h-4" />
                )}
              </Button>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}

function readAsDataURL(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const r = new FileReader()
    r.onload = () => resolve(String(r.result))
    r.onerror = () => reject(r.error)
    r.readAsDataURL(file)
  })
}
