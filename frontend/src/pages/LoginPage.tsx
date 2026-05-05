import { useMemo, useState } from "react"
import { useNavigate } from "react-router-dom"
import { Lock, User, AlertCircle, Loader2 } from "lucide-react"
import { login, register } from "@/lib/api"
import type { AuthState } from "@/hooks/useAuth"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { RegisterRules, type RegisterField } from "@/lib/validate"

interface Props {
  auth: AuthState
}

const REG_FIELDS: Array<{
  key: RegisterField
  label: string
  placeholder: string
  type?: string
}> = [
  { key: "username", label: "用户名", placeholder: "字母数字，最多 10 位" },
  { key: "password", label: "密码", placeholder: "至少 8 位字母+数字", type: "password" },
  { key: "name", label: "姓名", placeholder: "请输入姓名" },
  { key: "student_id", label: "学号", placeholder: "请输入学号" },
  { key: "nickname", label: "昵称", placeholder: "请输入昵称" },
  { key: "class_name", label: "班级", placeholder: "如：计算机网络1班" },
  { key: "email", label: "邮箱", placeholder: "name@example.com", type: "email" },
]

export default function LoginPage({ auth }: Props) {
  const navigate = useNavigate()
  const [tab, setTab] = useState<"login" | "register">("login")

  // ===== login =====
  const [u, setU] = useState("")
  const [p, setP] = useState("")
  const [err, setErr] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)

  async function onLogin(e: React.FormEvent) {
    e.preventDefault()
    setErr(null)
    setLoading(true)
    try {
      const res = await login(u, p)
      auth.setSession(res.token, res.user)
      navigate("/")
    } catch (ex) {
      setErr((ex as Error).message || "登录失败")
    } finally {
      setLoading(false)
    }
  }

  // ===== register =====
  const [reg, setReg] = useState<Record<RegisterField, string>>({
    username: "",
    password: "",
    name: "",
    student_id: "",
    nickname: "",
    class_name: "",
    email: "",
  })
  const [submitAttempted, setSubmitAttempted] = useState(false)
  const [regErr, setRegErr] = useState<string | null>(null)
  const [regLoading, setRegLoading] = useState(false)

  // 实时错误
  const liveErrors = useMemo(() => {
    const out: Partial<Record<RegisterField, string | null>> = {}
    for (const f of REG_FIELDS) {
      const liveFn = RegisterRules[f.key].live as (v: string) => string | null
      out[f.key] = liveFn(reg[f.key])
    }
    return out
  }, [reg])

  // strict 错误（只在提交后展示空值校验）
  const strictErrors = useMemo(() => {
    if (!submitAttempted) return liveErrors
    const out: Partial<Record<RegisterField, string | null>> = {}
    for (const f of REG_FIELDS) {
      const strictFn = RegisterRules[f.key].strict
      out[f.key] = strictFn(reg[f.key])
    }
    return out
  }, [reg, submitAttempted, liveErrors])

  const allValid = useMemo(() => {
    return REG_FIELDS.every((f) => RegisterRules[f.key].strict(reg[f.key]) === null)
  }, [reg])

  async function onRegister(e: React.FormEvent) {
    e.preventDefault()
    setRegErr(null)
    setSubmitAttempted(true)
    if (!allValid) return
    setRegLoading(true)
    try {
      const res = await register(reg)
      auth.setSession(res.token, res.user)
      navigate("/")
    } catch (ex) {
      setRegErr((ex as Error).message || "注册失败")
    } finally {
      setRegLoading(false)
    }
  }

  return (
    <div className="min-h-screen flex items-center justify-center px-4 py-8">
      <div className="w-full max-w-md">
        <div className="text-center mb-6">
          <div className="inline-flex items-center justify-center w-14 h-14 rounded-2xl bg-gradient-to-b from-[#A42A1E] to-[#861A11] text-white text-2xl font-extrabold shadow-md mb-3">
            N
          </div>
          <h1 className="text-3xl font-bold text-[hsl(var(--primary))]">NetRUC Agent</h1>
        </div>

        <div className="rounded-2xl border border-[hsl(var(--border))] bg-white shadow-[0_12px_32px_rgba(53,29,24,0.10)] overflow-hidden">
          <Tabs value={tab} onValueChange={(v) => setTab(v as "login" | "register")}>
            <div className="px-6 pt-3">
              <TabsList>
                <TabsTrigger value="login">登录</TabsTrigger>
                <TabsTrigger value="register">注册</TabsTrigger>
              </TabsList>
            </div>

            <TabsContent value="login">
              <form onSubmit={onLogin} className="px-6 pb-6 pt-2 space-y-4">
                <FieldWithIcon icon={<User className="w-4 h-4" />} label="用户名" id="login-username">
                  <Input
                    id="login-username"
                    placeholder="请输入用户名"
                    value={u}
                    onChange={(e) => setU(e.target.value)}
                    autoComplete="username"
                    className="pl-9"
                  />
                </FieldWithIcon>
                <FieldWithIcon icon={<Lock className="w-4 h-4" />} label="密码" id="login-password">
                  <Input
                    id="login-password"
                    type="password"
                    placeholder="请输入密码"
                    value={p}
                    onChange={(e) => setP(e.target.value)}
                    autoComplete="current-password"
                    className="pl-9"
                  />
                </FieldWithIcon>
                {err && (
                  <div className="flex items-start gap-2 text-sm text-[hsl(var(--destructive))] bg-red-50 border border-red-200 rounded-lg px-3 py-2">
                    <AlertCircle className="w-4 h-4 mt-0.5 shrink-0" />
                    <span>{err}</span>
                  </div>
                )}
                <Button type="submit" className="w-full" disabled={loading || !u || !p}>
                  {loading && <Loader2 className="w-4 h-4 animate-spin" />}
                  登录
                </Button>
              </form>
            </TabsContent>

            <TabsContent value="register">
              <form onSubmit={onRegister} className="px-6 pb-6 pt-2 space-y-3">
                {REG_FIELDS.map((f) => {
                  const error = strictErrors[f.key]
                  return (
                    <div key={f.key} className="space-y-1">
                      <Label htmlFor={`reg-${f.key}`}>{f.label}</Label>
                      <Input
                        id={`reg-${f.key}`}
                        type={f.type || "text"}
                        placeholder={f.placeholder}
                        value={reg[f.key]}
                        onChange={(e) => setReg({ ...reg, [f.key]: e.target.value })}
                        className={error ? "border-[hsl(var(--destructive))]" : ""}
                      />
                      {error && (
                        <div className="flex items-center gap-1 text-xs text-[hsl(var(--destructive))]">
                          <AlertCircle className="w-3 h-3" />
                          {error}
                        </div>
                      )}
                    </div>
                  )
                })}
                {regErr && (
                  <div className="flex items-start gap-2 text-sm text-[hsl(var(--destructive))] bg-red-50 border border-red-200 rounded-lg px-3 py-2">
                    <AlertCircle className="w-4 h-4 mt-0.5 shrink-0" />
                    <span>{regErr}</span>
                  </div>
                )}
                <Button type="submit" className="w-full" disabled={regLoading}>
                  {regLoading && <Loader2 className="w-4 h-4 animate-spin" />}
                  注册
                </Button>
              </form>
            </TabsContent>
          </Tabs>
        </div>
      </div>
    </div>
  )
}

function FieldWithIcon({
  icon,
  label,
  id,
  children,
}: {
  icon: React.ReactNode
  label: string
  id: string
  children: React.ReactNode
}) {
  return (
    <div className="space-y-1.5">
      <Label htmlFor={id}>{label}</Label>
      <div className="relative">
        <span className="absolute left-3 top-1/2 -translate-y-1/2 text-[hsl(var(--muted-foreground))]">
          {icon}
        </span>
        {children}
      </div>
    </div>
  )
}
