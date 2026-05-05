import * as React from "react"
import { cn } from "@/lib/utils"

export const Card = React.forwardRef<HTMLDivElement, React.HTMLAttributes<HTMLDivElement>>(
  ({ className, ...props }, ref) => (
    <div
      ref={ref}
      className={cn(
        "rounded-2xl border border-[hsl(var(--border))] bg-white shadow-[0_8px_24px_rgba(38,22,19,0.08)]",
        className
      )}
      {...props}
    />
  )
)
Card.displayName = "Card"

export const CardHeader = ({ className, ...props }: React.HTMLAttributes<HTMLDivElement>) => (
  <div className={cn("flex flex-col gap-1.5 p-6 pb-3", className)} {...props} />
)
export const CardTitle = ({ className, ...props }: React.HTMLAttributes<HTMLDivElement>) => (
  <h3 className={cn("text-lg font-semibold tracking-tight text-[hsl(var(--foreground))]", className)} {...props} />
)
export const CardDescription = ({ className, ...props }: React.HTMLAttributes<HTMLDivElement>) => (
  <p className={cn("text-sm text-[hsl(var(--muted-foreground))]", className)} {...props} />
)
export const CardContent = ({ className, ...props }: React.HTMLAttributes<HTMLDivElement>) => (
  <div className={cn("p-6 pt-0", className)} {...props} />
)
export const CardFooter = ({ className, ...props }: React.HTMLAttributes<HTMLDivElement>) => (
  <div className={cn("flex items-center p-6 pt-0", className)} {...props} />
)
