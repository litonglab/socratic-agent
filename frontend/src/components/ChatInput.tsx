import { useState, useRef, useEffect, useMemo } from "react"
import {
  Send,
  Paperclip,
  Loader2,
  Globe,
  X,
  Square,
  FileText,
  FileArchive,
  FileSpreadsheet,
  File as FileIcon,
} from "lucide-react"
import { Button } from "@/components/ui/button"
import { cn } from "@/lib/utils"
import { useHotkeys, type HotkeyBinding } from "@/hooks/useHotkeys"

/** 统一的附件抽象：图片走 OCR，其它走文件抽取。 */
export interface Attachment {
  id: string
  /** "image" 走 dataUrl 缩略图；"file" 显示为 chip。 */
  kind: "image" | "file"
  name: string
  size: number
  mime: string
  /** 纯 base64（不带 data: 前缀），上传给后端用 */
  base64: string
  /** 仅图片：data URL，用于本地预览 */
  dataUrl?: string
}

/** 兼容老调用方：图片附件类型别名。 */
export type AttachedImage = Attachment

interface Props {
  disabled?: boolean
  /** 是否处于流式输出中：true 时把发送按钮替换为"停止" */
  streaming?: boolean
  websearch: boolean
  onWebsearchChange: (v: boolean) => void
  onSend: (text: string, attachments: Attachment[]) => void
  onStop?: () => void
}

const MAX_ATTACHMENTS = 4
const MAX_BYTES = 10 * 1024 * 1024 // 10MB

// 后端可解析的非图片扩展名（与 agentic_rag/file_extract.py PLAIN_TEXT_EXTS + 文档类对齐）
const FILE_EXT_WHITELIST = new Set<string>([
  ".pdf",
  ".docx",
  ".doc",
  ".xlsx",
  ".pptx",
  ".zip",
  ".txt",
  ".md",
  ".markdown",
  ".rst",
  ".log",
  ".json",
  ".yaml",
  ".yml",
  ".toml",
  ".ini",
  ".cfg",
  ".conf",
  ".env",
  ".csv",
  ".tsv",
  ".py",
  ".js",
  ".jsx",
  ".ts",
  ".tsx",
  ".java",
  ".c",
  ".cc",
  ".cpp",
  ".h",
  ".hpp",
  ".go",
  ".rs",
  ".rb",
  ".php",
  ".sh",
  ".bash",
  ".zsh",
  ".html",
  ".htm",
  ".xml",
  ".sql",
])

const ACCEPT_ATTR = "image/*," + Array.from(FILE_EXT_WHITELIST).join(",")

function getExt(name: string): string {
  const idx = name.lastIndexOf(".")
  return idx >= 0 ? name.slice(idx).toLowerCase() : ""
}

function formatSize(bytes: number): string {
  if (bytes < 1024) return `${bytes}B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)}KB`
  return `${(bytes / 1024 / 1024).toFixed(1)}MB`
}

/** 按扩展名挑一个合适的 lucide 图标。 */
function pickFileIcon(name: string) {
  const ext = getExt(name)
  if (ext === ".zip") return FileArchive
  if (ext === ".xlsx" || ext === ".csv" || ext === ".tsv") return FileSpreadsheet
  if (
    ext === ".pdf" ||
    ext === ".docx" ||
    ext === ".doc" ||
    ext === ".pptx" ||
    ext === ".txt" ||
    ext === ".md"
  )
    return FileText
  return FileIcon
}

export default function ChatInput({
  disabled,
  streaming,
  websearch,
  onWebsearchChange,
  onSend,
  onStop,
}: Props) {
  const [text, setText] = useState("")
  const [attachments, setAttachments] = useState<Attachment[]>([])
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

  // "/" 聚焦输入框（默认在 input/textarea/contenteditable 内不触发，避免和打字冲突）
  const hotkeys = useMemo<HotkeyBinding[]>(
    () => [
      {
        key: "/",
        handler: () => taRef.current?.focus(),
      },
    ],
    [],
  )
  useHotkeys(hotkeys)

  function submit() {
    const v = text.trim()
    if ((!v && attachments.length === 0) || disabled) return
    onSend(v, attachments)
    setText("")
    setAttachments([])
    setError(null)
  }

  function handleKey(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
      e.preventDefault()
      submit()
    }
  }

  async function addFiles(files: FileList | File[]) {
    const arr = Array.from(files)
    if (arr.length === 0) return
    const next: Attachment[] = []
    for (const f of arr) {
      if (attachments.length + next.length >= MAX_ATTACHMENTS) {
        setError(`最多上传 ${MAX_ATTACHMENTS} 个附件`)
        break
      }
      if (f.size > MAX_BYTES) {
        setError(`文件 ${f.name} 超过 ${(MAX_BYTES / 1024 / 1024).toFixed(0)}MB`)
        continue
      }
      const isImage = (f.type || "").startsWith("image/")
      const ext = getExt(f.name)
      if (!isImage && !FILE_EXT_WHITELIST.has(ext)) {
        setError(`暂不支持的文件类型：${f.name}`)
        continue
      }
      try {
        const dataUrl = await readAsDataURL(f)
        const base64 = dataUrl.split(",", 2)[1] || ""
        next.push({
          id: Math.random().toString(36).slice(2),
          kind: isImage ? "image" : "file",
          name: f.name || (isImage ? "image.png" : "file"),
          size: f.size,
          mime: f.type || (isImage ? "image/png" : "application/octet-stream"),
          base64,
          dataUrl: isImage ? dataUrl : undefined,
        })
      } catch (err) {
        setError(`读取失败：${f.name}（${(err as Error).message || "unknown"}）`)
      }
    }
    if (next.length) setAttachments((prev) => [...prev, ...next])
  }

  function onPaste(e: React.ClipboardEvent) {
    if (!e.clipboardData?.files?.length) return
    // 仅拦截图片粘贴；其它文件粘贴体验各浏览器差异较大，避免误吞
    const imgs = Array.from(e.clipboardData.files).filter((f) => f.type.startsWith("image/"))
    if (imgs.length) {
      e.preventDefault()
      void addFiles(imgs)
    }
  }

  function removeAttachment(id: string) {
    setAttachments((prev) => prev.filter((x) => x.id !== id))
  }

  const images = attachments.filter((a) => a.kind === "image")
  const files = attachments.filter((a) => a.kind === "file")

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
          {/* 附件预览：图片缩略图 + 文件 chip */}
          {(images.length > 0 || files.length > 0) && (
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
                    onClick={() => removeAttachment(img.id)}
                    className="absolute -top-1.5 -right-1.5 w-4 h-4 rounded-full bg-[hsl(var(--destructive))] text-white grid place-items-center opacity-90 hover:opacity-100"
                    aria-label="删除图片"
                  >
                    <X className="w-2.5 h-2.5" />
                  </button>
                </div>
              ))}
              {files.map((f) => {
                const Icon = pickFileIcon(f.name)
                return (
                  <div
                    key={f.id}
                    className="relative group inline-flex items-center gap-1.5 max-w-[220px] h-8 px-2 rounded-md border border-[hsl(var(--border))] bg-[hsl(var(--accent))]/40 text-xs text-[hsl(var(--ink-700,#4D3D3A))]"
                    title={`${f.name} · ${formatSize(f.size)}`}
                  >
                    <Icon className="w-3.5 h-3.5 shrink-0 text-[hsl(var(--primary))]" />
                    <span className="truncate">{f.name}</span>
                    <span className="shrink-0 text-[hsl(var(--muted-foreground))]">
                      {formatSize(f.size)}
                    </span>
                    <button
                      type="button"
                      onClick={() => removeAttachment(f.id)}
                      className="absolute -top-1.5 -right-1.5 w-4 h-4 rounded-full bg-[hsl(var(--destructive))] text-white grid place-items-center opacity-90 hover:opacity-100"
                      aria-label="删除附件"
                    >
                      <X className="w-2.5 h-2.5" />
                    </button>
                  </div>
                )
              })}
            </div>
          )}

          {/* textarea 独占一行 */}
          <textarea
            ref={taRef}
            value={text}
            onChange={(e) => setText(e.target.value)}
            onKeyDown={handleKey}
            onPaste={onPaste}
            placeholder="输入你的问题，例如：我对子网划分感到困惑（可粘贴图片，也可上传 PDF / Word / ZIP）"
            rows={1}
            className="block w-full resize-none bg-transparent outline-none text-sm leading-relaxed max-h-[200px] px-3.5 pt-3 pb-1"
          />

          {/* 工具栏：左侧 Paperclip + 联网搜索 chip，右侧 Send */}
          <div className="flex items-center gap-1.5 px-2 pb-2">
            <button
              type="button"
              onClick={() => fileRef.current?.click()}
              className="shrink-0 w-8 h-8 grid place-items-center rounded-lg text-[hsl(var(--muted-foreground))] hover:bg-[hsl(var(--accent))] hover:text-[hsl(var(--primary))]"
              title="上传附件（图片可粘贴；支持 PDF / Word / Excel / PPT / ZIP / 文本）"
            >
              <Paperclip className="w-4 h-4" />
            </button>
            <input
              ref={fileRef}
              type="file"
              accept={ACCEPT_ATTR}
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
                disabled={disabled || (!text.trim() && attachments.length === 0)}
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
