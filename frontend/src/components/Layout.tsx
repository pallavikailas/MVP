import { Outlet, useLocation, Link } from 'react-router-dom'
import { motion, AnimatePresence } from 'framer-motion'
import { useAuditStore } from '../hooks/useAuditStore'

const STAGES = [
  { key: 'upload',        label: 'Configure',        path: '/audit',   num: 1, subs: [] },
  { key: 'model_probe',   label: 'Model Probe',      path: '/audit',   num: 2, subs: ['Cartography', 'Constitution'] },
  { key: 'dataset_probe', label: 'Dataset Analysis', path: '/audit',   num: 3, subs: ['Cartography', 'Proxy Hunt'] },
  { key: 'review',        label: 'Results',          path: '/results', num: 4, subs: [] },
]

const STAGE_ORDER = STAGES.map(s => s.key)

export default function Layout() {
  const { stage, activeSubStage, loading } = useAuditStore()
  const loc = useLocation()
  const currentIdx = STAGE_ORDER.indexOf(stage)

  return (
    <div className="min-h-screen bg-night flex flex-col">
      {/* Top nav */}
      <header className="border-b border-white/5 sticky top-0 z-50 bg-night/90 backdrop-blur-md">
        <div className="max-w-7xl mx-auto px-6 flex flex-col">
          <div className="h-14 flex items-center justify-between">
            <Link to="/" className="font-display font-bold text-lg text-white flex items-center gap-2">
              <span className="text-lens">◈</span> FairLens
            </Link>

            {/* Stage progress */}
            <div className="hidden md:flex items-center gap-1">
              {STAGES.map((s, i) => {
                const done = i < currentIdx
                const active = i === currentIdx
                return (
                  <div key={s.key} className="flex items-center">
                    <div className={`
                      flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-mono transition-all
                      ${active ? 'bg-lens/20 text-lens-light border border-lens/30' : ''}
                      ${done ? 'text-signal-green' : ''}
                      ${!active && !done ? 'text-white/20' : ''}
                    `}>
                      {done ? '✓' : active && loading
                        ? <motion.span animate={{ rotate: 360 }} transition={{ duration: 1, repeat: Infinity, ease: 'linear' }} className="inline-block text-lens-light">⟳</motion.span>
                        : s.num}
                      <span className={active ? 'text-lens-light' : ''}>{s.label}</span>
                    </div>
                    {i < STAGES.length - 1 && (
                      <div className={`w-4 h-px mx-0.5 ${i < currentIdx ? 'bg-signal-green/40' : 'bg-white/10'}`} />
                    )}
                  </div>
                )
              })}
            </div>

            <div className="text-xs font-mono text-white/30">
              Google Solution Challenge 2026
            </div>
          </div>

          {/* Sub-stage indicator strip — only visible while a phase is running */}
          <AnimatePresence>
            {loading && activeSubStage && (() => {
              const activeStage = STAGES.find(s => s.key === stage)
              if (!activeStage?.subs.length) return null
              return (
                <motion.div
                  key={stage}
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: 'auto' }}
                  exit={{ opacity: 0, height: 0 }}
                  className="flex items-center gap-1 pb-2 overflow-hidden">
                  <span className="text-white/20 text-xs font-mono mr-2">running:</span>
                  {activeStage.subs.map((sub, i) => {
                    const isActive = activeSubStage === sub
                    const isDone   = activeStage.subs.indexOf(activeSubStage) > i
                    return (
                      <div key={sub} className="flex items-center gap-1">
                        <span className={`text-xs font-mono px-2 py-0.5 rounded transition-all
                          ${isActive ? 'bg-lens/25 text-lens-light border border-lens/40 font-semibold' :
                            isDone   ? 'text-signal-green/70' : 'text-white/20'}`}>
                          {isDone ? '✓ ' : isActive ? '▶ ' : ''}{sub}
                        </span>
                        {i < activeStage.subs.length - 1 && (
                          <span className="text-white/15 text-xs">→</span>
                        )}
                      </div>
                    )
                  })}
                </motion.div>
              )
            })()}
          </AnimatePresence>
        </div>
      </header>

      <main className="flex-1">
        <motion.div
          key={loc.pathname}
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3 }}
        >
          <Outlet />
        </motion.div>
      </main>
    </div>
  )
}
