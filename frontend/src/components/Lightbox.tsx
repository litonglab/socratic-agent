import { useEffect } from "react"
import { X } from "lucide-react"

interface Props {
  src: string | null
  onClose: () => void
}

/**
 * 简易图片放大查看器：
 * - 全屏遮罩 + 居中图片
 * - 点击遮罩 / 按 ESC 关闭；点图片本身不关闭
 * - 不依赖第三方库，键盘可达
 */
export default function Lightbox({ src, onClose }: Props) {
  useEffect(() => {
    if (!src) return
    function onKey(e: KeyboardEvent) {
      if (e.key === "Escape") onClose()
    }
    window.addEventListener("keydown", onKey)
    // 防止背后滚动
    const prev = document.body.style.overflow
    document.body.style.overflow = "hidden"
    return () => {
      window.removeEventListener("keydown", onKey)
      document.body.style.overflow = prev
    }
  }, [src, onClose])

  if (!src) return null

  return (
    <div
      className="fixed inset-0 z-[120] bg-black/75 backdrop-blur-sm flex items-center justify-center p-6 cursor-zoom-out"
      role="dialog"
      aria-modal="true"
      aria-label="图片预览"
      onClick={onClose}
    >
      <button
        type="button"
        onClick={onClose}
        className="absolute top-4 right-4 grid place-items-center w-9 h-9 rounded-full bg-white/15 text-white hover:bg-white/30"
        aria-label="关闭预览（ESC）"
        title="关闭（ESC）"
      >
        <X className="w-4 h-4" />
      </button>
      <img
        src={src}
        alt="预览"
        className="max-w-[92vw] max-h-[88vh] rounded-lg shadow-2xl object-contain cursor-default"
        onClick={(e) => e.stopPropagation()}
        draggable={false}
      />
    </div>
  )
}
