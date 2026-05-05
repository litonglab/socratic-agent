/**
 * 注册表单字段校验：与 streamlit 版规则保持一致。
 * - liveErr：边输入边校验（空值不报错，避免噪声）
 * - strictErr：提交时校验（空值也报错）
 */

const ALPHANUMERIC = /^[A-Za-z0-9]+$/
const EMAIL = /^[^@\s]+@[^@\s]+\.[^@\s]+$/

export const RegisterRules = {
  username: {
    label: "用户名",
    live: (v: string) => {
      const s = (v || "").trim()
      if (!s) return null
      if (s.length > 10) return "用户名最多 10 位字母数字"
      if (!ALPHANUMERIC.test(s)) return "用户名仅支持字母和数字"
      return null
    },
    strict: (v: string) => {
      const s = (v || "").trim()
      if (!s) return "请输入用户名"
      if (!ALPHANUMERIC.test(s)) return "用户名仅支持字母和数字"
      if (s.length > 10) return "用户名最多 10 位字母数字"
      return null
    },
  },
  password: {
    label: "密码",
    live: (v: string) => {
      if (!v) return null
      if (v.length < 8) return "密码需至少 8 位"
      if (!/[A-Za-z]/.test(v) || !/\d/.test(v)) return "密码需同时包含字母和数字"
      if (!ALPHANUMERIC.test(v)) return "密码仅支持字母和数字"
      return null
    },
    strict: (v: string) => {
      if (!v) return "请输入密码"
      if (v.length < 8) return "密码需至少 8 位"
      if (!/[A-Za-z]/.test(v) || !/\d/.test(v)) return "密码需同时包含字母和数字"
      if (!ALPHANUMERIC.test(v)) return "密码仅支持字母和数字"
      return null
    },
  },
  name: {
    label: "姓名",
    live: () => null,
    strict: (v: string) => ((v || "").trim() ? null : "请输入姓名"),
  },
  student_id: {
    label: "学号",
    live: (v: string) => {
      const s = (v || "").trim()
      if (!s) return null
      if (s.length > 32) return "学号长度不能超过 32 位"
      if (!ALPHANUMERIC.test(s)) return "学号仅支持字母和数字"
      return null
    },
    strict: (v: string) => {
      const s = (v || "").trim()
      if (!s) return "请输入学号"
      if (s.length > 32) return "学号长度不能超过 32 位"
      if (!ALPHANUMERIC.test(s)) return "学号仅支持字母和数字"
      return null
    },
  },
  nickname: {
    label: "昵称",
    live: () => null,
    strict: (v: string) => ((v || "").trim() ? null : "请输入昵称"),
  },
  class_name: {
    label: "班级",
    live: () => null,
    strict: (v: string) => ((v || "").trim() ? null : "请输入班级"),
  },
  email: {
    label: "邮箱",
    live: (v: string) => {
      const s = (v || "").trim()
      if (!s) return null
      return EMAIL.test(s) ? null : "请输入有效邮箱地址"
    },
    strict: (v: string) => {
      const s = (v || "").trim()
      if (!s) return "请输入邮箱"
      return EMAIL.test(s) ? null : "请输入有效邮箱地址"
    },
  },
} as const

export type RegisterField = keyof typeof RegisterRules
