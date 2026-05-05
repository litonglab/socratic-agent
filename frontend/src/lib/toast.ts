export type ToastType = "info" | "success" | "error"

export interface Toast {
  id: string
  type: ToastType
  text: string
  durationMs: number
}

type Listener = (toasts: Toast[]) => void

let _seq = 0
let _toasts: Toast[] = []
const _listeners = new Set<Listener>()

function emit() {
  for (const l of _listeners) l([..._toasts])
}

export function dismissToast(id: string) {
  _toasts = _toasts.filter((t) => t.id !== id)
  emit()
}

export function getToastSnapshot(): Toast[] {
  return [..._toasts]
}

export function subscribeToasts(listener: Listener): () => void {
  _listeners.add(listener)
  return () => {
    _listeners.delete(listener)
  }
}

function show(text: string, type: ToastType = "info", durationMs = 3000) {
  const id = `toast_${++_seq}`
  const t: Toast = { id, type, text, durationMs }
  _toasts = [..._toasts, t]
  emit()
  if (durationMs > 0) {
    window.setTimeout(() => dismissToast(id), durationMs)
  }
  return id
}

export const toast = {
  show,
  info: (text: string, durationMs?: number) => show(text, "info", durationMs ?? 3000),
  success: (text: string, durationMs?: number) => show(text, "success", durationMs ?? 2500),
  error: (text: string, durationMs?: number) => show(text, "error", durationMs ?? 4000),
  dismiss: dismissToast,
}
