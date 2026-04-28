import { useState, useEffect, useRef } from 'react'
import { useNavigate } from 'react-router-dom'
import { motion, AnimatePresence } from 'framer-motion'
import { useAuditStore } from '../hooks/useAuditStore'
import { streamRedTeam, exportPdfReport } from '../utils/api'

type AgentEvent = {
  node: string
  iteration?: number
  log?: string[]
  status?: string
  results?: any
}

const NODE_META: Record<string, { label: string; icon: string; color: string }> = {
  attack:       { label: 'Attack Agent',    icon: '⚔',  color: '#ef4444' },
  evaluate:     { label: 'Evaluator Agent', icon: '🔬', color: '#f97316' },
  decide_patch: { label: 'Decision Agent',  icon: '⚖',  color: '#eab308' },
  patch:        { label: 'Patcher Agent',   icon: '🔧', color: '#3b82f6' },
  validate:     { label: 'Validator Agent', icon: '✓',  color: '#10b981' },
  report:       { label: 'Report Agent',    icon: '📋', color: '#7c3aed' },
  complete:     { label: 'Complete',        icon: '✅', color: '#10b981' },
}

function AgentNode({ name, active, done }: { name: string; active: boolean; done: boolean }) {
  const meta = NODE_META[name] || { label: name, icon: '◈', color: '#7c3aed' }
  return (
    <motion.div animate={{ scale: active ? 1.05 : 1 }}
      className="flex items-center gap-3 p-3 rounded-xl border transition-all"
      style={{
        borderColor: active ? meta.color + '60' : done ? '#10b98130' : '#ffffff10',
        background: active ? meta.color + '10' : 'transparent',
      }}>
      <div className="w-8 h-8 rounded-lg flex items-center justify-center text-sm flex-shrink-0"
        style={{ background: meta.color + '20', border: `1px solid ${meta.color}40` }}>
        {active ? <span className="animate-spin text-xs inline-block">⟳</span> : done ? '✓' : meta.icon}
      </div>
      <div>
        <div className="text-xs font-mono" style={{ color: active ? meta.color : done ? '#10b981' : '#ffffff44' }}>
          {meta.label}
        </div>
        {active && (
          <div className="flex gap-0.5 mt-1">
            {[0, 1, 2].map(i => (
              <motion.div key={i} className="w-1 h-1 rounded-full"
                style={{ background: meta.color }}
                animate={{ opacity: [0.3, 1, 0.3] }}
                transition={{ duration: 0.8, repeat: Infinity, delay: i * 0.2 }} />
            ))}
          </div>
        )}
      </div>
    </motion.div>
  )
}

function LogLine({ line, index }: { line: string; index: number }) {
  const getColor = (t: string) => {
    if (t.includes('Attack')) return '#ef4444'
    if (t.includes('Evaluator')) return '#f97316'
    if (t.includes('Decision')) return '#eab308'
    if (t.includes('Patcher')) return '#3b82f6'
    if (t.includes('Validator')) return '#10b981'
    if (t.includes('Report')) return '#7c3aed'
    if (t.includes('Failed') || t.includes('error')) return '#ef4444'
    if (t.includes('improved') || t.includes('Done')) return '#10b981'
    return '#ffffff50'
  }
  return (
    <motion.div initial={{ opacity: 0, x: -8 }} animate={{ opacity: 1, x: 0 }}
      transition={{ delay: index * 0.03 }}
      className="flex items-start gap-2 py-1 font-mono text-xs border-b border-white/3 last:border-0">
      <span className="text-white/20 flex-shrink-0">{String(index + 1).padStart(3, '0')}</span>
      <span style={{ color: getColor(line) }}>{line}</span>
    </motion.div>
  )
}

export default function RedTeamPage() {
  const nav = useNavigate()
  const store = useAuditStore()
  const [allLogs, setAllLogs] = useState<string[]>([])
  const [activeNode, setActiveNode] = useState<string | null>(null)
  const [doneNodes, setDoneNodes] = useState<Set<string>>(new Set())
  const [finalResults, setFinalResults] = useState<any>(null)
  const [running, setRunning] = useState(false)
  const [started, setStarted] = useState(false)
  const [stoppedByUser, setStoppedByUser] = useState(false)
  const stopRef = useRef<(() => void) | null>(null)
  const logRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight
  }, [allLogs])

  const hasModel = !!store.modelFile || (store.modelType !== 'sklearn' && !!store.modelEndpoint)

  const startRedTeam = () => {
    if (!hasModel) return
    if (!store.datasetFile && store.datasetSource === 'upload') return
    if (store.confirmedBiases.length === 0) return
    setRunning(true); setStarted(true); setStoppedByUser(false)
    setAllLogs([]); setDoneNodes(new Set()); setFinalResults(null)

    const auditResults = {
      crossAnalysis: store.crossAnalysisResults,
      cartography:   store.cartographyResults,
      constitution:  store.constitutionResults,
      proxy:         store.proxyResults,
    }

    // Biases from Phase 1 (model probe) + Phase 2 (dataset probe) — sent to redteam separately
    const modelProbeBiases   = store.modelProbeResults?.model_biases   ?? []
    const datasetProbeBiases = store.datasetProbeResults?.dataset_biases ?? []

    const stop = streamRedTeam(
      store.modelFile,
      store.datasetFile,
      store.protectedCols,
      store.targetCol,
      store.confirmedBiases,
      auditResults,
      (event: AgentEvent) => {
        if (event.log?.length) setAllLogs(prev => [...prev, ...event.log!])
        if (event.node && event.node !== 'complete' && event.node !== 'error') setActiveNode(event.node)
        if (event.node === 'error') {
          setActiveNode(null)
          setRunning(false)
        }
        if (event.status === 'done' || event.node === 'complete') {
          setActiveNode(null)
          setDoneNodes(new Set(Object.keys(NODE_META)))
          setRunning(false)
          setFinalResults(event.results)
          store.setRedteamResults(event.results)
          store.setStage('done')
        }
      },
      store.datasetSource,
      store.datasetUrl,
      store.modelType,
      store.modelEndpoint,
      store.llmApiKey,
      store.hfToken,
      modelProbeBiases,
      datasetProbeBiases,
    )
    stopRef.current = stop
  }

  const AGENT_NODES = ['attack', 'evaluate', 'decide_patch', 'patch', 'validate', 'report']
  const report = finalResults?.final_report || finalResults || {}
  const validation = report.validation || finalResults?.validation_results || {}
  const improved = validation.improved || []
  const patchesApplied = report.patches_applied ?? finalResults?.patch_results?.applied?.length ?? 0
  const fairnessDelta = report.remediated_fairness || {}
  const modelArtifact = report.patched_model_artifact || null

  const downloadPatchedModel = () => {
    if (!modelArtifact?.available || !modelArtifact?.pickle_b64) return
    const bytes = Uint8Array.from(atob(modelArtifact.pickle_b64), c => c.charCodeAt(0))
    const blob = new Blob([bytes], { type: 'application/octet-stream' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = modelArtifact.filename || 'fairlens-remediated-model.pkl'
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    URL.revokeObjectURL(url)
  }

  return (
    <div className="max-w-6xl mx-auto px-6 py-10">
      <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} className="mb-8">
        <div className="text-xs font-mono text-signal-red mb-1">Stage 6 — Red-Team Agent</div>
        <h1 className="font-display font-bold text-white text-3xl mb-2">Adversarial Bias Attack & Fix</h1>
        <p className="text-white/40 text-sm max-w-2xl">
          The agent generates adversarial probes targeting confirmed biases, evaluates their impact,
          applies mitigation patches, then validates that fixes hold.
        </p>
      </motion.div>

      {/* Confirmed biases */}
      <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.1 }}
        className="glass rounded-2xl p-5 border border-white/5 mb-6">
        <div className="text-xs font-mono text-white/30 mb-3 uppercase tracking-wider">
          Targeting {store.confirmedBiases.length} confirmed biases
        </div>
        <div className="flex flex-wrap gap-2">
          {store.confirmedBiases.map((b: any, i: number) => (
            <div key={i} className="flex items-center gap-2 bg-signal-red/10 border border-signal-red/20 rounded-lg px-3 py-1.5">
              <span className="w-1.5 h-1.5 rounded-full bg-signal-red" />
              <span className="text-signal-red text-xs font-mono">{b.attribute}</span>
            </div>
          ))}
        </div>
      </motion.div>

      <div className="grid md:grid-cols-3 gap-6">
        {/* Agent pipeline */}
        <div className="glass rounded-2xl p-5 border border-white/5">
          <h3 className="font-display font-semibold text-white text-sm mb-4">Agent Pipeline</h3>
          <div className="space-y-2">
            {AGENT_NODES.map(node => (
              <AgentNode key={node} name={node} active={activeNode === node} done={doneNodes.has(node)} />
            ))}
          </div>

          {!started && !hasModel && (
            <div className="mt-6 p-3 rounded-xl bg-signal-red/10 border border-signal-red/20 text-signal-red text-xs font-mono">
              ⚠ A model is required for red-team analysis. Go back and upload a .pkl file or configure a model endpoint.
            </div>
          )}

          {!started && hasModel && (
            <motion.button whileHover={{ scale: 1.01 }} whileTap={{ scale: 0.99 }}
              onClick={startRedTeam}
              className="w-full mt-6 py-3 rounded-xl bg-signal-red hover:bg-signal-red/90 text-white font-display font-semibold text-sm transition-all">
              ⊘ Launch Red-Team Agent
            </motion.button>
          )}

          {running && (
            <button onClick={() => {
              stopRef.current?.()
              setRunning(false)
              setActiveNode(null)
              setStarted(false)
              setStoppedByUser(true)
            }}
              className="w-full mt-4 py-2.5 rounded-xl bg-signal-red/10 border border-signal-red/30 text-signal-red hover:bg-signal-red/20 text-sm font-mono transition-all flex items-center justify-center gap-2">
              <span className="w-2.5 h-2.5 rounded-sm bg-signal-red inline-block" />
              Stop agent
            </button>
          )}

          {stoppedByUser && !running && !finalResults && (
            <div className="mt-4 p-3 rounded-xl bg-white/5 border border-white/10 text-white/40 text-xs font-mono">
              Agent stopped. Results may be incomplete.
            </div>
          )}

          {finalResults && (
            <div className="mt-4 p-3 rounded-xl bg-signal-green/5 border border-signal-green/20">
              <div className="text-signal-green font-mono text-xs font-semibold mb-1">✓ Agent complete</div>
              <div className="text-white/40 text-xs">
                {patchesApplied} patches · {store.confirmedBiases.length} biases targeted
              </div>
            </div>
          )}
        </div>

        {/* Live log */}
        <div className="md:col-span-2 glass rounded-2xl border border-white/5 overflow-hidden flex flex-col" style={{ maxHeight: '460px' }}>
          <div className="flex items-center justify-between px-5 py-3 border-b border-white/5">
            <h3 className="font-display font-semibold text-white text-sm">Agent Activity Log</h3>
            {running && (
              <div className="flex items-center gap-2">
                <span className="w-1.5 h-1.5 rounded-full bg-signal-red animate-pulse" />
                <span className="text-signal-red text-xs font-mono">LIVE</span>
              </div>
            )}
          </div>
          <div ref={logRef} className="flex-1 overflow-y-auto px-5 py-4 min-h-[200px]">
            {!started && (
              <div className="text-center py-12">
                <div className="text-4xl mb-4">⊘</div>
                <div className="text-white/30 text-sm font-mono">Launch the agent to start</div>
              </div>
            )}
            {allLogs.map((line, i) => <LogLine key={i} line={line} index={i} />)}
            {running && (
              <div className="flex items-center gap-2 mt-3 text-xs font-mono text-white/30">
                <motion.span animate={{ opacity: [0.3, 1, 0.3] }} transition={{ duration: 1, repeat: Infinity }}>▋</motion.span>
                Agent running...
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Results */}
      <AnimatePresence>
        {finalResults && (
          <motion.div initial={{ opacity: 0, y: 24 }} animate={{ opacity: 1, y: 0 }} className="mt-8">
            {/* Divider */}
            <div className="flex items-center gap-3 mb-6">
              <div className="flex-1 h-px bg-white/10" />
              <span className="text-white/20 text-xs font-mono uppercase tracking-widest">Report</span>
              <div className="flex-1 h-px bg-white/10" />
            </div>

            {/* Success / warning banner */}
            {improved.length > 0 ? (
              <div className="mb-6 p-4 rounded-2xl bg-signal-green/10 border border-signal-green/30 flex items-start gap-3">
                <span className="text-signal-green text-xl mt-0.5">✓</span>
                <div>
                  <div className="text-signal-green font-display font-semibold text-sm mb-0.5">
                    {improved.length} bias{improved.length > 1 ? 'es' : ''} successfully remediated
                  </div>
                  <div className="text-white/50 text-xs font-mono">
                    The agent applied mitigation patches and confirmed improvement across {improved.length} attribute{improved.length > 1 ? 's' : ''}.
                    {(validation.regressed?.length > 0) && ` ${validation.regressed.length} worsened.`}
                    {(validation.unchanged?.length > 0) && ` ${validation.unchanged.length} unchanged.`}
                  </div>
                </div>
              </div>
            ) : (
              <div className="mb-6 p-4 rounded-2xl bg-white/5 border border-white/10 flex items-start gap-3">
                <span className="text-white/30 text-xl mt-0.5">◈</span>
                <div className="text-white/50 text-xs font-mono">
                  {patchesApplied > 0
                    ? 'Corrections were applied via post-hoc factors embedded in the remediated model. Bias scores reflect estimated improvement.'
                    : 'No patches were applied — no confirmed biases required remediation.'}
                </div>
              </div>
            )}

            {/* Summary cards */}
            <div className="grid md:grid-cols-4 gap-4 mb-6">
              {[
                { label: 'Iterations',      value: report.iterations ?? 1,                                    color: 'text-white' },
                { label: 'Patches Applied', value: patchesApplied,                                            color: 'text-signal-green' },
                { label: 'Biases Improved', value: improved.length,                                           color: 'text-signal-green' },
                { label: 'Patches Failed',  value: Array.isArray(report.patches_failed) ? report.patches_failed.length : (report.patches_failed ?? 0),
                  color: (Array.isArray(report.patches_failed) ? report.patches_failed.length : (report.patches_failed ?? 0)) > 0 ? 'text-signal-red' : 'text-white/30' },
              ].map((card, i) => (
                <div key={i} className="glass rounded-2xl p-5 border border-white/5">
                  <div className="text-white/30 text-xs font-mono mb-2">{card.label}</div>
                  <div className={`font-display font-bold text-3xl ${card.color}`}>{card.value ?? 0}</div>
                </div>
              ))}
            </div>

            {/* Mitigation plan table */}
            {report.mitigation_plan?.length > 0 && (
              <div className="glass rounded-2xl p-5 border border-white/5 mb-6">
                <div className="text-xs font-mono text-white/30 uppercase tracking-wider mb-4">Mitigation Plan</div>
                <div className="overflow-x-auto">
                  <table className="w-full text-xs font-mono">
                    <thead>
                      <tr className="border-b border-white/5">
                        <th className="text-left text-white/30 pb-3 pr-4 font-normal">Attribute</th>
                        <th className="text-left text-white/30 pb-3 pr-4 font-normal">Strategy</th>
                        <th className="text-left text-white/30 pb-3 pr-4 font-normal">SPD</th>
                        <th className="text-left text-white/30 pb-3 pr-4 font-normal">Source</th>
                        <th className="text-left text-white/30 pb-3 font-normal">Rationale</th>
                      </tr>
                    </thead>
                    <tbody>
                      {report.mitigation_plan.map((item: any, i: number) => (
                        <tr key={i} className="border-b border-white/3 last:border-0">
                          <td className="py-3 pr-4 text-signal-red font-semibold">{item.attribute}</td>
                          <td className="py-3 pr-4">
                            <span className="px-2 py-0.5 rounded bg-lens/10 border border-lens/20 text-lens-light">
                              {item.strategy?.replace(/_/g, ' ')}
                            </span>
                          </td>
                          <td className="py-3 pr-4 text-white/70">
                            {item.disparity != null ? item.disparity.toFixed(4) : '—'}
                          </td>
                          <td className="py-3 pr-4 text-white/40">{item.bias_source ?? '—'}</td>
                          <td className="py-3 text-white/60 max-w-xs truncate">{item.rationale}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}

            {/* Per-attribute before/after */}
            {(() => {
              const rows: { attribute: string; before: number; after: number; status: string }[] = []
              const seen = new Set<string>()

              // Priority: measured improvements/regressions with actual before/after numbers
              for (const a of [...(validation.improved || []), ...(validation.regressed || [])]) {
                if (typeof a === 'object' && a.attribute && !seen.has(a.attribute)) {
                  const delta = (a.before ?? 0) - (a.after ?? a.before ?? 0)
                  rows.push({ attribute: a.attribute, before: a.before ?? 0, after: a.after ?? a.before ?? 0, status: delta > 0.001 ? 'improved' : delta < -0.001 ? 'regressed' : 'unchanged' })
                  seen.add(a.attribute)
                }
              }
              // Unchanged — may just be strings
              for (const a of (validation.unchanged || [])) {
                const attr = typeof a === 'string' ? a : a?.attribute
                if (attr && !seen.has(attr)) {
                  const plan = report.mitigation_plan?.find((m: any) => m.attribute === attr)
                  const spd = plan?.disparity ?? 0
                  rows.push({ attribute: attr, before: spd, after: spd, status: 'unchanged' })
                  seen.add(attr)
                }
              }

              if (!rows.length) return null
              return (
                <div className="glass rounded-2xl p-5 border border-white/5 mb-6">
                  <div className="text-xs font-mono text-white/30 uppercase tracking-wider mb-4">Per-Attribute Bias Change</div>
                  {fairnessDelta.before_avg_spd != null && (
                    <div className="grid md:grid-cols-3 gap-4 mb-5 pb-5 border-b border-white/5">
                      <div>
                        <div className="text-white/30 text-xs font-mono mb-1">Avg SPD Before</div>
                        <div className="text-white font-display text-2xl">{fairnessDelta.before_avg_spd.toFixed(3)}</div>
                      </div>
                      <div>
                        <div className="text-white/30 text-xs font-mono mb-1">Avg SPD After</div>
                        <div className="text-signal-green font-display text-2xl">{fairnessDelta.after_avg_spd?.toFixed(3) ?? '—'}</div>
                      </div>
                      <div>
                        <div className="text-white/30 text-xs font-mono mb-1">Improvement</div>
                        <div className="text-lens-light font-display text-2xl">
                          {fairnessDelta.improvement != null ? `[−${fairnessDelta.improvement.toFixed(3)}]` : '—'}
                        </div>
                      </div>
                    </div>
                  )}
                  <div className="space-y-3">
                    {rows.map((item, i) => {
                      const before = item.before
                      const after  = item.after
                      const delta  = before - after
                      const statusColor = item.status === 'improved' ? '#10b981' : item.status === 'regressed' ? '#ef4444' : '#ffffff30'
                      const barMax = Math.max(before, after, 0.001)
                      return (
                        <div key={i} className="flex items-center gap-3 py-1.5 border-b border-white/3 last:border-0">
                          <div className="w-24 flex-shrink-0 text-xs font-mono text-white/70 truncate">{item.attribute}</div>
                          {/* Before bar */}
                          <div className="flex-1 flex items-center gap-1.5">
                            <div className="flex-1 relative h-2 rounded-sm bg-white/5">
                              <div className="absolute left-0 top-0 h-full rounded-sm"
                                style={{ width: `${Math.max((before / barMax) * 100, 2)}%`, background: '#ef444460' }} />
                            </div>
                            <span className="text-white/40 text-xs font-mono w-12 text-right tabular-nums">{before.toFixed(3)}</span>
                          </div>
                          <span className="text-white/20 text-xs font-mono">→</span>
                          {/* After bar */}
                          <div className="flex-1 flex items-center gap-1.5">
                            <div className="flex-1 relative h-2 rounded-sm bg-white/5">
                              <div className="absolute left-0 top-0 h-full rounded-sm"
                                style={{ width: `${Math.max((after / barMax) * 100, 2)}%`, background: statusColor }} />
                            </div>
                            <span className="text-xs font-mono w-12 text-right tabular-nums" style={{ color: statusColor }}>{after.toFixed(3)}</span>
                          </div>
                          {/* Delta badge */}
                          <div className="w-20 flex-shrink-0 text-right">
                            <span className="text-xs font-mono px-2 py-0.5 rounded-md"
                              style={{ background: statusColor + '20', color: statusColor }}>
                              {item.status === 'improved' ? `−${Math.abs(delta).toFixed(3)}` : item.status === 'regressed' ? `+${Math.abs(delta).toFixed(3)}` : '—'}
                            </span>
                          </div>
                        </div>
                      )
                    })}
                  </div>
                </div>
              )
            })()}

            {/* Use the improved model */}
            <div className="glass rounded-2xl p-5 border border-white/5 mb-6">
              <div className="text-xs font-mono text-white/30 uppercase tracking-wider mb-3">Use the Improved Model</div>
              <p className="text-white/50 text-xs font-mono mb-4">
                {modelArtifact?.available
                  ? 'The patched model includes per-group correction factors applied automatically at prediction time.'
                  : (modelArtifact?.message || 'No patched model artifact was generated for this run.')}
              </p>
              {modelArtifact?.available ? (
                <div className="space-y-3">
                  <button onClick={downloadPatchedModel}
                    className="w-full py-2.5 rounded-xl bg-signal-green hover:bg-signal-green/90 text-white font-display font-semibold text-sm transition-all flex items-center justify-center gap-2">
                    <span>↓</span> Download Remediated Model (.pkl)
                  </button>
                  <div className="text-xs font-mono text-white/30 space-y-1 bg-white/3 rounded-xl p-3 border border-white/5">
                    <div className="text-white/40 mb-1.5">How to use:</div>
                    <div>1. <span className="text-lens-light">joblib.load('fairlens-remediated-model.pkl')</span></div>
                    <div>2. <span className="text-lens-light">model.predict(X)</span> — corrections automatic</div>
                  </div>
                </div>
              ) : (
                report.mitigation_plan?.length > 0 && (
                  <div className="text-xs font-mono text-white/30 space-y-1.5">
                    <div className="text-white/40 mb-2">Mitigation measures to implement:</div>
                    {(report.mitigation_plan || []).map((item: any, i: number) => (
                      <div key={i} className="flex items-start gap-2">
                        <span className="text-lens-light">•</span>
                        <span><span className="text-white/60">{item.attribute}</span>: {item.strategy?.replace(/_/g, ' ')} — {item.rationale}</span>
                      </div>
                    ))}
                  </div>
                )
              )}
            </div>

            <div className="flex gap-4">
              <button onClick={() => nav('/audit')}
                className="flex-1 py-3 rounded-xl border border-white/10 text-white/60 hover:text-white font-mono text-sm transition-all">
                ← New Audit
              </button>
              <button onClick={async () => {
                try {
                  const fullReport = {
                    ...(store.crossAnalysisResults?.cartography ?? store.cartographyResults ?? {}),
                    constitution: store.crossAnalysisResults?.constitution ?? store.constitutionResults ?? null,
                    proxy_hunt: store.crossAnalysisResults?.proxy ?? store.proxyResults ?? null,
                    model_probe: store.modelProbeResults ?? null,
                    dataset_probe: store.datasetProbeResults ?? null,
                    cross_synthesis: store.crossAnalysisResults?.cross_synthesis ?? null,
                    redteam: finalResults,
                  }
                  await exportPdfReport(fullReport)
                } catch (e: any) {
                  alert(`PDF export failed: ${e.message}`)
                }
              }} className="flex-1 py-3 rounded-xl bg-lens hover:bg-lens/90 text-white font-display font-semibold text-sm glow-lens transition-all">
                ↓ Export Full Report (PDF)
              </button>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}
