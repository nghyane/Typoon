// SettingsSheet — tabbed reader preferences (Layout / Image /
// Behavior). Mirrors truyendrive's settings dialog. Each tab is a
// flat list of fields bound to the `useReaderSettings` store; no
// local state — single source of truth.

import { useState } from 'react'
import {
  ArrowDown, ArrowLeftRight, ArrowRight,
} from 'lucide-react'

import { Modal } from '@shared/ui/Modal'
import { cn } from '@shared/lib/cn'
import {
  PAGE_WIDTH_MIN, PAGE_WIDTH_MAX,
  PAGE_GAP_MAX, STRIP_MARGIN_MAX, PRELOAD_AHEAD_MAX,
  useReaderSettings,
  type Direction, type ImageFit,
} from './store'


type Tab = 'layout' | 'image' | 'behavior'


interface Props {
  open:    boolean
  onClose: () => void
  workId:  number
}


export function SettingsSheet({ open, onClose, workId }: Props) {
  const [tab, setTab] = useState<Tab>('layout')

  return (
    <Modal open={open} onClose={onClose} title="Cài đặt đọc" size="sm">
      <div className="flex flex-col">
        <Tabs tab={tab} setTab={setTab} />
        <div className="px-5 py-4">
          {tab === 'layout'   && <LayoutTab workId={workId} />}
          {tab === 'image'    && <ImageTab />}
          {tab === 'behavior' && <BehaviorTab />}
        </div>
      </div>
    </Modal>
  )
}


// ── Tab strip ──────────────────────────────────────────────────


function Tabs({
  tab, setTab,
}: {
  tab: Tab
  setTab: (t: Tab) => void
}) {
  const items: Array<{ id: Tab; label: string }> = [
    { id: 'layout',   label: 'Bố cục' },
    { id: 'image',    label: 'Ảnh' },
    { id: 'behavior', label: 'Hành vi' },
  ]
  return (
    <div className="flex items-center gap-1 px-5 pt-3 border-b border-border-soft">
      {items.map((it) => (
        <button
          key={it.id}
          onClick={() => setTab(it.id)}
          className={cn(
            'h-9 px-3 text-sm rounded-t-md transition-colors cursor-pointer',
            'border-b-2 -mb-px',
            tab === it.id
              ? 'border-accent text-text font-medium'
              : 'border-transparent text-text-muted hover:text-text',
          )}
        >
          {it.label}
        </button>
      ))}
    </div>
  )
}


// ── Layout tab ─────────────────────────────────────────────────


function LayoutTab({ workId }: { workId: number }) {
  const directionByWork = useReaderSettings((s) => s.directionByWork)
  const setDirection    = useReaderSettings((s) => s.setDirection)
  const dir = directionByWork[String(workId)] ?? 'ttb'

  const pageWidth   = useReaderSettings((s) => s.pageWidth)
  const setPageWidth= useReaderSettings((s) => s.setPageWidth)
  const pageGap     = useReaderSettings((s) => s.pageGap)
  const setPageGap  = useReaderSettings((s) => s.setPageGap)
  const stripMargin = useReaderSettings((s) => s.stripMargin)
  const setStripMargin = useReaderSettings((s) => s.setStripMargin)

  return (
    <div className="space-y-5">
      <Section title="Hướng đọc" hint="Cài đặt riêng cho work này.">
        <div className="grid grid-cols-3 gap-2">
          <DirectionTile current={dir} value="ltr"
            label="Trái → phải" icon={<ArrowRight size={18} />}
            onClick={() => setDirection(workId, 'ltr')} />
          <DirectionTile current={dir} value="rtl"
            label="Phải → trái" icon={<ArrowLeftRight size={18} className="rotate-180" />}
            onClick={() => setDirection(workId, 'rtl')} />
          <DirectionTile current={dir} value="ttb"
            label="Cuộn dọc" icon={<ArrowDown size={18} />}
            onClick={() => setDirection(workId, 'ttb')} />
        </div>
      </Section>

      <Section title="Bề rộng trang" hint={`${pageWidth}px`}>
        <input
          type="range"
          min={PAGE_WIDTH_MIN}
          max={PAGE_WIDTH_MAX}
          step={20}
          value={pageWidth}
          onChange={(e) => setPageWidth(Number(e.target.value))}
          className="w-full h-2 appearance-none bg-surface-2 rounded-full accent-accent cursor-pointer"
        />
      </Section>

      {dir === 'ttb' && (
        <>
          <Section title="Khoảng cách giữa các trang" hint={`${pageGap}px`}>
            <input
              type="range"
              min={0} max={PAGE_GAP_MAX} step={2}
              value={pageGap}
              onChange={(e) => setPageGap(Number(e.target.value))}
              className="w-full h-2 appearance-none bg-surface-2 rounded-full accent-accent cursor-pointer"
            />
          </Section>

          <Section title="Lề trên / dưới" hint={`${stripMargin}px`}>
            <input
              type="range"
              min={0} max={STRIP_MARGIN_MAX} step={2}
              value={stripMargin}
              onChange={(e) => setStripMargin(Number(e.target.value))}
              className="w-full h-2 appearance-none bg-surface-2 rounded-full accent-accent cursor-pointer"
            />
          </Section>
        </>
      )}
    </div>
  )
}


function DirectionTile({
  current, value, label, icon, onClick,
}: {
  current: Direction
  value:   Direction
  label:   string
  icon:    React.ReactNode
  onClick: () => void
}) {
  const active = current === value
  return (
    <button
      onClick={onClick}
      className={cn(
        'flex flex-col items-center gap-1.5 py-3 rounded-md',
        'border transition-colors cursor-pointer',
        active
          ? 'bg-accent-bg border-accent text-accent-text'
          : 'bg-surface-2 border-transparent text-text-muted hover:text-text hover:bg-hover',
      )}
    >
      {icon}
      <span className="text-xs font-medium">{label}</span>
    </button>
  )
}


// ── Image tab ─────────────────────────────────────────────────


function ImageTab() {
  const imageFit    = useReaderSettings((s) => s.imageFit)
  const setImageFit = useReaderSettings((s) => s.setImageFit)
  const preloadAhead= useReaderSettings((s) => s.preloadAhead)
  const setPreloadAhead = useReaderSettings((s) => s.setPreloadAhead)

  const FITS: Array<{ id: ImageFit; label: string; hint: string }> = [
    { id: 'width',  label: 'Vừa bề rộng',  hint: 'Tốt cho webtoon dọc.' },
    { id: 'height', label: 'Vừa chiều cao', hint: 'Tốt cho manga 2 trang.' },
    { id: 'free',   label: 'Tự do',         hint: 'Giữ kích thước gốc.' },
  ]

  return (
    <div className="space-y-5">
      <Section title="Vừa khung">
        <div className="grid grid-cols-3 gap-2">
          {FITS.map((f) => (
            <button
              key={f.id}
              onClick={() => setImageFit(f.id)}
              className={cn(
                'flex flex-col items-start gap-0.5 px-3 py-2.5 rounded-md text-left',
                'border transition-colors cursor-pointer',
                imageFit === f.id
                  ? 'bg-accent-bg border-accent text-accent-text'
                  : 'bg-surface-2 border-transparent text-text-muted hover:text-text hover:bg-hover',
              )}
            >
              <span className="text-xs font-medium">{f.label}</span>
              <span className="text-xs opacity-60">{f.hint}</span>
            </button>
          ))}
        </div>
      </Section>

      <Section
        title="Tải trước trang"
        hint={preloadAhead === 0 ? 'Tắt' : `${preloadAhead} trang`}
      >
        <input
          type="range"
          min={0} max={PRELOAD_AHEAD_MAX} step={1}
          value={preloadAhead}
          onChange={(e) => setPreloadAhead(Number(e.target.value))}
          className="w-full h-2 appearance-none bg-surface-2 rounded-full accent-accent cursor-pointer"
        />
      </Section>
    </div>
  )
}


// ── Behavior tab ──────────────────────────────────────────────


function BehaviorTab() {
  const click  = useReaderSettings((s) => s.clickTurnPage)
  const setClick = useReaderSettings((s) => s.setClickTurnPage)
  const swipe  = useReaderSettings((s) => s.swipeGestures)
  const setSwipe = useReaderSettings((s) => s.setSwipeGestures)
  const resume = useReaderSettings((s) => s.resumePosition)
  const setResume = useReaderSettings((s) => s.setResumePosition)

  return (
    <div className="space-y-2">
      <Toggle
        label="Bấm/chạm để lật trang"
        hint="Bấm mép trái/phải để đi trước/sau, giữa để hiện thanh điều khiển."
        on={click}
        onChange={setClick}
      />
      <Toggle
        label="Vuốt ngang để lật trang"
        hint="Trên chế độ trái ↔ phải, vuốt ngang ≥ 60px chuyển trang."
        on={swipe}
        onChange={setSwipe}
      />
      <Toggle
        label="Nhớ vị trí đọc"
        hint="Mở lại chương sẽ tự quay về đúng chỗ đã dừng."
        on={resume}
        onChange={setResume}
      />
    </div>
  )
}


// ── Atoms ─────────────────────────────────────────────────────


function Section({
  title, hint, children,
}: {
  title:    string
  hint?:    string
  children: React.ReactNode
}) {
  return (
    <div>
      <div className="flex items-center justify-between mb-2">
        <span className="text-xs uppercase tracking-wide text-text-subtle">
          {title}
        </span>
        {hint && <span className="text-xs text-text-subtle tabular">{hint}</span>}
      </div>
      {children}
    </div>
  )
}


function Toggle({
  label, hint, on, onChange,
}: {
  label:    string
  hint?:    string
  on:       boolean
  onChange: (v: boolean) => void
}) {
  return (
    <label
      className={cn(
        'flex items-start gap-3 py-2.5 cursor-pointer',
        'hover:bg-hover -mx-2 px-2 rounded-md',
      )}
    >
      <input
        type="checkbox"
        checked={on}
        onChange={(e) => onChange(e.target.checked)}
        className="mt-0.5 size-4 accent-accent cursor-pointer"
      />
      <div className="min-w-0 flex-1">
        <p className="text-sm text-text">{label}</p>
        {hint && <p className="text-xs text-text-subtle mt-0.5">{hint}</p>}
      </div>
    </label>
  )
}
