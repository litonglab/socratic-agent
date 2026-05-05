import { useState, useRef, useEffect } from "react"
import { Send, Paperclip, Loader2, Globe, X } from "lucide-react"
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
  websearch: boolean
  onWebsearchChange: (v: boolean) => void
  onSend: (text: string, images: AttachedImage[]) => void
}

const MAX_IMAGES = 4
const MAX_BYTES = 8 * 1024 * 1024 // 8MB

export default function ChatInput({ disabled, websearch, onWebsearchChange, onSend }: Props) {
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
        {/* 联网搜索 chip（输入框上方右对齐） */}
        <div className="flex justify-end mb-1.5">
          <button
            type="button"
            onClick={() => onWebsearchChange(!websearch)}
            className={cn(
              "inline-flex items-center gap-1.5 text-xs px-2.5 py-1 rounded-full border transition-colors",
              websearch
                ? "bg-[hsl(var(--primary)/0.10)] border-[hsl(var(--primary)/0.35)] text-[hsl(var(--primary))]"
                : "bg-white border-[hsl(var(--border))] text-[hsl(var(--muted-foreground))] hover:border-[hsl(var(--primary)/0.4)] hover:text-[hsl(var(--primary))]",
            )}
            title={websearch ? "联网搜索：已开启" : "联网搜索：未开启"}
          >
            <Globe className="w-3.5 h-3.5" />
            联网搜索
            <span
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
        </div>

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

          <div className="flex items-end gap-1 px-2 py-2">
            <button
              type="button"
              onClick={() => fileRef.current?.click()}
              className="shrink-0 w-9 h-9 grid place-items-center rounded-lg text-[hsl(var(--muted-foreground))] hover:bg-[hsl(var(--accent))] hover:text-[hsl(var(--primary))]"
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
            <textarea
              ref={taRef}
              value={text}
              onChange={(e) => setText(e.target.value)}
              onKeyDown={handleKey}
              onPaste={onPaste}
              placeholder="输入你的问题，例如：我对子网划分感到困惑（可粘贴图片）"
              rows={1}
              className="flex-1 resize-none bg-transparent outline-none text-sm py-2 leading-relaxed max-h-[200px] px-1"
            />
            <Button
              size="icon"
              onClick={submit}
              disabled={disabled || (!text.trim() && images.length === 0)}
              className="shrink-0 w-9 h-9 rounded-xl"
            >
              {disabled ? <Loader2 className="w-4 h-4 animate-spin" /> : <Send className="w-4 h-4" />}
            </Button>
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
