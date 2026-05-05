import ReactMarkdown from "react-markdown"
import remarkGfm from "remark-gfm"
import rehypeHighlight from "rehype-highlight"
import rehypeRaw from "rehype-raw"
import "highlight.js/styles/github.css"
import { cn } from "@/lib/utils"

interface Props {
  content: string
  className?: string
  /** 用户消息：白底深字反向（在红色气泡里）；assistant 默认深字 */
  inverse?: boolean
}

export default function Markdown({ content, className, inverse }: Props) {
  return (
    <div
      className={cn(
        "prose-base max-w-none leading-relaxed",
        inverse ? "text-white" : "text-[hsl(var(--foreground))]",
        className,
      )}
    >
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        rehypePlugins={[rehypeHighlight, rehypeRaw]}
        components={{
          p: ({ children }) => (
            <p className="my-2 first:mt-0 last:mb-0 whitespace-pre-wrap">{children}</p>
          ),
          h1: ({ children }) => <h1 className="text-xl font-bold mt-3 mb-2">{children}</h1>,
          h2: ({ children }) => <h2 className="text-lg font-bold mt-3 mb-2">{children}</h2>,
          h3: ({ children }) => <h3 className="text-base font-bold mt-2 mb-1">{children}</h3>,
          ul: ({ children }) => <ul className="list-disc list-outside pl-5 my-2 space-y-0.5">{children}</ul>,
          ol: ({ children }) => <ol className="list-decimal list-outside pl-5 my-2 space-y-0.5">{children}</ol>,
          li: ({ children }) => <li className="leading-relaxed">{children}</li>,
          a: ({ href, children }) => (
            <a
              href={href}
              target="_blank"
              rel="noreferrer noopener"
              className={cn(
                "underline underline-offset-2",
                inverse ? "text-white/90 hover:text-white" : "text-[hsl(var(--primary))] hover:opacity-80",
              )}
            >
              {children}
            </a>
          ),
          strong: ({ children }) => <strong className="font-bold">{children}</strong>,
          em: ({ children }) => <em className="italic">{children}</em>,
          blockquote: ({ children }) => (
            <blockquote
              className={cn(
                "border-l-4 pl-3 my-2 italic opacity-90",
                inverse ? "border-white/40" : "border-[hsl(var(--border))]",
              )}
            >
              {children}
            </blockquote>
          ),
          hr: () => <hr className={cn("my-3", inverse ? "border-white/30" : "border-[hsl(var(--border))]")} />,
          code: ({ inline, className, children, ...props }: any) => {
            if (inline) {
              return (
                <code
                  className={cn(
                    "px-1.5 py-0.5 rounded text-[0.9em] font-mono",
                    inverse ? "bg-white/15 text-white" : "bg-[hsl(var(--accent))] text-[hsl(var(--foreground))]",
                  )}
                  {...props}
                >
                  {children}
                </code>
              )
            }
            return (
              <code className={cn("font-mono text-xs", className)} {...props}>
                {children}
              </code>
            )
          },
          pre: ({ children }) => (
            <pre
              className={cn(
                "rounded-lg p-3 my-2 overflow-x-auto text-xs",
                inverse ? "bg-black/30" : "bg-[#FAF6F4] border border-[hsl(var(--border))]",
              )}
            >
              {children}
            </pre>
          ),
          table: ({ children }) => (
            <div className="overflow-x-auto my-2">
              <table className="border-collapse text-sm">{children}</table>
            </div>
          ),
          th: ({ children }) => (
            <th
              className={cn(
                "border px-2 py-1 text-left font-semibold",
                inverse ? "border-white/30 bg-white/10" : "border-[hsl(var(--border))] bg-[hsl(var(--accent))]",
              )}
            >
              {children}
            </th>
          ),
          td: ({ children }) => (
            <td className={cn("border px-2 py-1", inverse ? "border-white/20" : "border-[hsl(var(--border))]")}>
              {children}
            </td>
          ),
        }}
      >
        {content}
      </ReactMarkdown>
    </div>
  )
}
