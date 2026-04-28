import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { motion, AnimatePresence } from 'framer-motion'
import { useAuditStore } from '../hooks/useAuditStore'
import { runModelProbe, runDatasetProbe } from '../utils/api'

export default function AuditPage() {
  const nav   = useNavigate()
  const store = useAuditStore()

  const [hfModel,   setHfModel]   = useState('')
  const [hfToken,   setHfToken]   = useState('')
  const [hfDataset, setHfDataset] = useState('')

  const canRun = !!(hfModel || hfDataset)

  const runAudit = async () => {
    if (!canRun || store.loading) return
    store.setError(null)
    store.setLoading(true)
    store.setModelType('huggingface')
    store.setModelEndpoint(hfModel)
    store.setHfToken(hfToken)
    store.setDatasetSource('huggingface')
    store.setDatasetUrl(hfDataset)

    try {
      if (hfModel) {
        store.setStage('model_probe')
        const modelProbe = await runModelProbe(null, 'huggingface', hfModel, '', hfToken)
        store.setModelProbeResults(modelProbe)
      }

      if (hfDataset) {
        store.setStage('dataset_probe')
        const datasetProbe = await runDatasetProbe(null, ['auto'], 'auto', 'huggingface', hfDataset)
        store.setDatasetProbeResults(datasetProbe)
        store.setProtectedCols(
          datasetProbe.detected_protected_cols?.length
            ? datasetProbe.detected_protected_cols
            : ['auto']
        )
        store.setTargetCol(datasetProbe.detected_target_col || 'auto')
      }

      store.setStage('review')
      nav('/results')
    } catch (e: any) {
      store.setError(e.message)
    } finally {
      store.setLoading(false)
    }
  }

  return (
    <div className="max-w-xl mx-auto px-6 py-12">
      <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} className="mb-10">
        <div className="text-xs font-mono text-lens-light mb-2">Step 1 of 2</div>
        <h1 className="font-display font-bold text-white text-3xl mb-2">Configure Audit</h1>
        <p className="text-white/40 text-sm">
          Enter a HuggingFace model and/or dataset. At least one is required.
          Phases that need a missing input are skipped automatically.
        </p>
      </motion.div>

      {/* HuggingFace Model */}
      <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.05 }}
        className="mb-6 glass rounded-2xl p-5 border border-white/8">
        <div className="flex items-center gap-2 mb-4">
          <span className="text-2xl">🤗</span>
          <div>
            <div className="text-sm font-mono font-semibold text-white">HuggingFace Model</div>
            <div className="text-xs text-white/30">Any model from HuggingFace Hub — auto-detected</div>
          </div>
        </div>
        <div className="space-y-3">
          <input
            value={hfModel}
            onChange={e => setHfModel(e.target.value)}
            placeholder="e.g. unitary/toxic-bert or google/gemma-3-1b-it"
            className="w-full bg-white/5 border border-white/10 rounded-xl px-4 py-3 text-sm text-white placeholder-white/20 focus:outline-none focus:border-lens/50 font-mono"
          />
          <input
            value={hfToken}
            onChange={e => setHfToken(e.target.value)}
            placeholder="HuggingFace token (hf_...) — required for gated models"
            type="password"
            className="w-full bg-white/5 border border-white/10 rounded-xl px-4 py-3 text-sm text-white placeholder-white/20 focus:outline-none focus:border-lens/50 font-mono"
          />
        </div>
      </motion.div>

      {/* HuggingFace Dataset */}
      <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.1 }}
        className="mb-8 glass rounded-2xl p-5 border border-white/8">
        <div className="flex items-center gap-2 mb-4">
          <span className="text-2xl">📊</span>
          <div>
            <div className="text-sm font-mono font-semibold text-white">HuggingFace Dataset</div>
            <div className="text-xs text-white/30">Dataset name from huggingface.co/datasets</div>
          </div>
        </div>
        <input
          value={hfDataset}
          onChange={e => setHfDataset(e.target.value)}
          placeholder="e.g. csv/csv or Rowan/hellaswag"
          className="w-full bg-white/5 border border-white/10 rounded-xl px-4 py-3 text-sm text-white placeholder-white/20 focus:outline-none focus:border-lens/50 font-mono"
        />
      </motion.div>

      {/* Pipeline info */}
      <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.15 }}
        className="glass rounded-2xl p-4 border border-lens/10 mb-8 flex items-start gap-3">
        <div className="text-lens text-xl flex-shrink-0">✦</div>
        <div>
          <div className="text-lens-light font-mono text-xs font-semibold mb-1">What happens next</div>
          <div className="text-white/40 text-xs leading-relaxed space-y-1">
            <div>
              <span className="text-white/60">Phase 1 — Model Probe:</span> Tests the model on a neutral reference
              dataset to surface intrinsic bias. <span className="text-white/30 italic">Requires a model.</span>
            </div>
            <div>
              <span className="text-white/60">Phase 2 — Dataset Analysis:</span> Scans the dataset for structural
              bias, proxy chains, and demographic imbalances. <span className="text-white/30 italic">Requires a dataset.</span>
            </div>
          </div>
        </div>
      </motion.div>

      {/* Error */}
      <AnimatePresence>
        {store.error && (
          <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
            className="bg-signal-red/10 border border-signal-red/30 rounded-xl p-4 mb-6 text-signal-red text-sm font-mono">
            ⚠ {store.error}
          </motion.div>
        )}
      </AnimatePresence>

      {/* Loading */}
      <AnimatePresence>
        {store.loading && (
          <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }}
            className="glass rounded-2xl border border-lens/20 mb-6 px-5 py-4 flex items-center gap-3">
            <motion.div className="w-2 h-2 rounded-full bg-lens flex-shrink-0"
              animate={{ opacity: [1, 0.3, 1] }} transition={{ duration: 1.2, repeat: Infinity }} />
            <span className="text-lens-light font-mono text-sm">
              {store.stage === 'model_probe'   ? 'Phase 1 — Running model probe…' :
               store.stage === 'dataset_probe' ? 'Phase 2 — Analysing dataset…'  : 'Initialising…'}
            </span>
          </motion.div>
        )}
      </AnimatePresence>

      <motion.button
        whileHover={canRun && !store.loading ? { scale: 1.01 } : {}}
        whileTap={canRun && !store.loading ? { scale: 0.99 } : {}}
        disabled={!canRun || store.loading}
        onClick={runAudit}
        className={`w-full py-4 rounded-xl font-display font-semibold text-base transition-all
          ${canRun && !store.loading
            ? 'bg-lens hover:bg-lens/90 text-white glow-lens cursor-pointer'
            : 'bg-white/5 text-white/20 cursor-not-allowed'}`}>
        {store.loading ? 'Analysing...' : 'Run Bias Analysis →'}
      </motion.button>

      <p className="text-center text-white/20 text-xs mt-3 font-mono">
        Powered by Gemini 2.5 Flash · Google Cloud · Auto-detects protected attributes
      </p>
    </div>
  )
}
