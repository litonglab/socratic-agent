import { cn } from "@/lib/utils"

interface Props {
  /** 像素尺寸，默认 36 */
  size?: number
  /** 圆角类，默认 rounded-xl */
  roundedClass?: string
  className?: string
  /** 优先使用 public/favicon.svg；为 false 时回退渲染纯文字 N */
  useImage?: boolean
}

/**
 * NetRUC Agent 品牌徽标。
 * 同时用于：登录页、侧栏顶部、聊天空状态。
 */
export default function BrandLogo({
  size = 36,
  roundedClass = "rounded-xl",
  className,
  useImage = true,
}: Props) {
  const style = { width: size, height: size }
  return (
    <div
      className={cn(
        "shrink-0 grid place-items-center bg-gradient-to-b from-[#A42A1E] to-[#861A11] text-white font-extrabold shadow-sm",
        roundedClass,
        className,
      )}
      style={style}
      aria-label="NetRUC Agent"
    >
      {useImage ? (
        <img
          src="/favicon.svg"
          alt=""
          className="select-none"
          style={{ width: size * 0.7, height: size * 0.7 }}
          draggable={false}
        />
      ) : (
        <span style={{ fontSize: size * 0.5 }}>N</span>
      )}
    </div>
  )
}
