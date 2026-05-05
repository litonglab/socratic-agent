import * as React from "react"
import { cn } from "@/lib/utils"

export type InputProps = React.InputHTMLAttributes<HTMLInputElement>

export const Input = React.forwardRef<HTMLInputElement, InputProps>(
  ({ className, type = "text", ...props }, ref) => (
    <input
      ref={ref}
      type={type}
      className={cn(
        "flex h-10 w-full rounded-lg border border-[hsl(var(--input))] bg-white px-3 py-2 text-sm",
        "placeholder:text-[hsl(var(--muted-foreground))]",
        "focus-visible:outline-none focus-visible:border-[hsl(var(--ring))] focus-visible:ring-2 focus-visible:ring-[hsl(var(--ring)/0.18)]",
        "disabled:cursor-not-allowed disabled:opacity-50",
        className
      )}
      {...props}
    />
  )
)
Input.displayName = "Input"
