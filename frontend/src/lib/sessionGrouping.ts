import type { SessionMeta } from "@/lib/api"

export interface SessionGroup {
  key: string
  label: string
  items: SessionMeta[]
}

export function startOfDay(d: Date): number {
  return new Date(d.getFullYear(), d.getMonth(), d.getDate()).getTime()
}

export function groupSessionsByUpdatedAt(items: SessionMeta[]): SessionGroup[] {
  const now = new Date()
  const today = startOfDay(now)
  const yesterday = today - 24 * 3600 * 1000
  const sevenDays = today - 7 * 24 * 3600 * 1000
  const thirtyDays = today - 30 * 24 * 3600 * 1000

  const buckets: Record<string, SessionGroup> = {
    today: { key: "today", label: "今天", items: [] },
    yesterday: { key: "yesterday", label: "昨天", items: [] },
    week: { key: "week", label: "7 天内", items: [] },
    month: { key: "month", label: "30 天内", items: [] },
    older: { key: "older", label: "更早", items: [] },
    unknown: { key: "unknown", label: "其他", items: [] },
  }

  for (const s of items) {
    const ts = s.updated_at ? Date.parse(s.updated_at) : NaN
    if (Number.isNaN(ts)) {
      buckets.unknown.items.push(s)
      continue
    }
    const dayStart = startOfDay(new Date(ts))
    if (dayStart >= today) buckets.today.items.push(s)
    else if (dayStart >= yesterday) buckets.yesterday.items.push(s)
    else if (dayStart >= sevenDays) buckets.week.items.push(s)
    else if (dayStart >= thirtyDays) buckets.month.items.push(s)
    else buckets.older.items.push(s)
  }

  for (const k of Object.keys(buckets)) {
    buckets[k].items.sort((a, b) => {
      const ta = a.updated_at ? Date.parse(a.updated_at) : 0
      const tb = b.updated_at ? Date.parse(b.updated_at) : 0
      return tb - ta
    })
  }

  return [
    buckets.today,
    buckets.yesterday,
    buckets.week,
    buckets.month,
    buckets.older,
    buckets.unknown,
  ].filter((g) => g.items.length > 0)
}
