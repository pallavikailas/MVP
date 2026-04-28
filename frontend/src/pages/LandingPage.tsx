import { useNavigate } from 'react-router-dom'
import { motion } from 'framer-motion'

// Set this to your deployed full prototype URL
const FULL_PROTOTYPE_URL = import.meta.env.VITE_FULL_PROTOTYPE_URL || ''

const EXAMPLES = [
  { role: 'Job applicant',   decision: 'REJECTED',    attr: 'Female, 32',      real: 'Same qualifications as hired male counterpart' },
  { role: 'Loan applicant',  decision: 'DENIED',      attr: 'ZIP: 60619',      real: 'Statistically identical credit profile to approved applicant' },
  { role: 'Medical patient', decision: 'LOW PRIORITY', attr: 'Black male, 45', real: 'Healthcare cost proxy used — correlated with race' },
]

const PIPELINE = [
  { icon: '🤗', title: 'HuggingFace Model',   desc: 'Enter any model from HuggingFace Hub — auto-detected and loaded' },
  { icon: '📊', title: 'HuggingFace Dataset', desc: 'Enter any dataset from HuggingFace Hub — analyzed for structural bias' },
  { icon: '⬡',  title: 'Bias Analysis',       desc: 'Model and dataset probed across identity slices — hotspots surfaced' },
  { icon: '📋', title: 'Results Report',      desc: 'Bias cartography, proxy chains, and compliance flags in one view' },
]

export default function LandingPage() {
  const nav = useNavigate()

  return (
    <div className="min-h-screen bg-night overflow-hidden">
      {/* Grid background */}
      <div className="fixed inset-0 opacity-[0.03]"
        style={{ backgroundImage: 'linear-gradient(#7c3aed 1px, transparent 1px), linear-gradient(90deg, #7c3aed 1px, transparent 1px)', backgroundSize: '60px 60px' }} />

      {/* Glow orb */}
      <div className="fixed top-1/3 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[600px] h-[600px] rounded-full opacity-10"
        style={{ background: 'radial-gradient(circle, #7c3aed 0%, transparent 70%)' }} />

      <div className="relative z-10 max-w-5xl mx-auto px-6 pt-16 pb-32">

        {/* MVP banner */}
        <motion.div initial={{ opacity: 0, y: -10 }} animate={{ opacity: 1, y: 0 }}
          className="flex items-center justify-between mb-8 bg-lens/5 border border-lens/20 rounded-2xl px-5 py-3">
          <div className="flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-lens-light animate-pulse flex-shrink-0" />
            <span className="text-xs font-mono text-white/50">
              MVP Demo — HuggingFace inputs only
            </span>
          </div>
          {FULL_PROTOTYPE_URL ? (
            <a
              href={FULL_PROTOTYPE_URL}
              target="_blank"
              rel="noopener noreferrer"
              className="text-xs font-mono font-semibold text-lens-light hover:text-white border border-lens/30 hover:border-lens/60 px-3 py-1.5 rounded-lg transition-all">
              Try Full Prototype →
            </a>
          ) : (
            <span className="text-xs font-mono text-white/25 border border-white/10 px-3 py-1.5 rounded-lg">
              Full prototype coming soon
            </span>
          )}
        </motion.div>

        {/* Nav */}
        <motion.div initial={{ opacity: 0, y: -20 }} animate={{ opacity: 1, y: 0 }}
          className="flex items-center justify-between mb-20">
          <span className="font-display font-bold text-xl text-white flex items-center gap-2">
            <span className="text-lens text-2xl">◈</span> FairLens
          </span>
          <span className="text-xs font-mono text-white/30 border border-white/10 px-3 py-1.5 rounded-full">
            Google Solution Challenge 2026
          </span>
        </motion.div>

        {/* Hero */}
        <motion.div initial={{ opacity: 0, y: 30 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.1 }}
          className="text-center mb-20">
          <div className="inline-flex items-center gap-2 bg-lens/10 border border-lens/20 text-lens-light text-xs font-mono px-4 py-2 rounded-full mb-8">
            <span className="w-1.5 h-1.5 rounded-full bg-lens-light animate-pulse" />
            Powered by Gemini 2.5 Flash · Vertex AI · Google Cloud Run
          </div>

          <h1 className="font-display font-bold text-white mb-6" style={{ fontSize: 'clamp(2.5rem, 6vw, 5rem)', lineHeight: 1.1 }}>
            Programs that decide your life<br />
            <span className="text-lens">shouldn't be unfair.</span>
          </h1>

          <p className="text-white/50 text-lg max-w-2xl mx-auto mb-12 leading-relaxed">
            FairLens detects, maps, and fixes hidden bias in AI decision systems —
            before they affect real people's jobs, loans, and healthcare.
          </p>

          <div className="flex items-center justify-center gap-4 flex-wrap">
            <motion.button
              whileHover={{ scale: 1.02 }} whileTap={{ scale: 0.98 }}
              onClick={() => nav('/audit')}
              className="bg-lens hover:bg-lens/90 text-white font-display font-semibold px-8 py-4 rounded-xl text-base glow-lens transition-all">
              Start Bias Audit →
            </motion.button>
            {FULL_PROTOTYPE_URL && (
              <a
                href={FULL_PROTOTYPE_URL}
                target="_blank"
                rel="noopener noreferrer"
                className="text-white/50 hover:text-white/80 text-sm border border-white/15 hover:border-white/30 px-6 py-4 rounded-xl transition-all font-mono">
                Full Prototype (all inputs) →
              </a>
            )}
          </div>
        </motion.div>

        {/* Real-world examples */}
        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.3 }}
          className="mb-24">
          <p className="text-center text-xs font-mono text-white/25 uppercase tracking-widest mb-8">
            Real decisions. Hidden discrimination.
          </p>
          <div className="grid md:grid-cols-3 gap-4">
            {EXAMPLES.map((ex, i) => (
              <motion.div key={i}
                initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.4 + i * 0.1 }}
                className="glass rounded-2xl p-5 border border-white/5 hover:border-signal-red/20 transition-colors group">
                <div className="flex items-center justify-between mb-3">
                  <span className="text-xs font-mono text-white/30">{ex.role}</span>
                  <span className="text-xs font-mono text-signal-red bg-signal-red/10 px-2 py-0.5 rounded border border-signal-red/20">{ex.decision}</span>
                </div>
                <div className="text-sm text-white/70 font-mono mb-2">{ex.attr}</div>
                <div className="text-xs text-white/30 leading-relaxed">{ex.real}</div>
                <div className="mt-3 h-px bg-gradient-to-r from-signal-red/30 to-transparent group-hover:from-signal-red/60 transition-all" />
              </motion.div>
            ))}
          </div>
        </motion.div>

        {/* MVP flow */}
        <div className="mb-24">
          <p className="text-center text-xs font-mono text-white/25 uppercase tracking-widest mb-3">The MVP flow</p>
          <h2 className="text-center font-display font-bold text-white text-3xl mb-12">
            Two inputs. One verdict.
          </h2>
          <div className="grid md:grid-cols-4 gap-4">
            {PIPELINE.map((step, i) => (
              <motion.div key={i}
                initial={{ opacity: 0, y: 20 }} whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }} transition={{ delay: i * 0.1 }}
                className="glass rounded-2xl p-6 border border-white/5 hover:border-lens/30 transition-all group relative overflow-hidden">
                <div className="absolute inset-0 opacity-0 group-hover:opacity-100 transition-opacity"
                  style={{ background: 'radial-gradient(circle at 30% 30%, rgba(124,58,237,0.05) 0%, transparent 60%)' }} />
                <div className="text-2xl mb-4 relative">{step.icon}</div>
                <div className="text-xs font-mono text-lens-light mb-2 relative">{String(i + 1).padStart(2, '0')}</div>
                <h3 className="font-display font-semibold text-white text-sm mb-2 relative">{step.title}</h3>
                <p className="text-xs text-white/40 leading-relaxed relative">{step.desc}</p>
              </motion.div>
            ))}
          </div>
        </div>

        {/* CTA */}
        <motion.div initial={{ opacity: 0 }} whileInView={{ opacity: 1 }} viewport={{ once: true }}
          className="text-center glass rounded-3xl p-12 border border-lens/10">
          <h2 className="font-display font-bold text-white text-3xl mb-4">
            Ready to audit your model?
          </h2>
          <p className="text-white/40 mb-8">Enter a HuggingFace model and dataset. Get a full bias report in minutes.</p>
          <div className="flex items-center justify-center gap-4 flex-wrap">
            <motion.button
              whileHover={{ scale: 1.02 }} whileTap={{ scale: 0.98 }}
              onClick={() => nav('/audit')}
              className="bg-lens hover:bg-lens/90 text-white font-display font-semibold px-10 py-4 rounded-xl text-base glow-lens transition-all">
              Launch FairLens →
            </motion.button>
            {FULL_PROTOTYPE_URL && (
              <a
                href={FULL_PROTOTYPE_URL}
                target="_blank"
                rel="noopener noreferrer"
                className="text-white/40 hover:text-white/70 text-sm transition-colors font-mono">
                Full prototype →
              </a>
            )}
          </div>
        </motion.div>
      </div>
    </div>
  )
}
