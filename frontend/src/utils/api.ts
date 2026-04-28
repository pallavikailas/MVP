/// <reference types="vite/client" />
const BASE = import.meta.env.VITE_API_BASE_URL || ''

// ── Phase 1: Model probe on embedded reference dataset ────────────────────────
export async function runModelProbe(
  modelFile: File | null,
  modelType: string = 'sklearn',
  apiEndpoint: string = '',
  llmApiKey: string = '',
  hfToken: string = '',
  protectedCols: string[] = [],
): Promise<any> {
  const fd = new FormData()
  if (modelFile) fd.append('model_file', modelFile)
  fd.append('model_type', modelType)
  if (apiEndpoint) fd.append('api_endpoint', apiEndpoint)
  if (llmApiKey)   fd.append('llm_api_key', llmApiKey)
  if (hfToken)     fd.append('hf_token', hfToken)
  if (protectedCols.length) fd.append('protected_cols', protectedCols.join(','))
  const res = await fetch(`${BASE}/api/v1/model-probe/run`, { method: 'POST', body: fd })
  if (!res.ok) throw new Error(await res.text())
  return res.json()
}

// ── Phase 2: Dataset-only bias probe ─────────────────────────────────────────
export async function runDatasetProbe(
  datasetFile: File | null,
  protectedCols: string[],
  targetCol: string,
  datasetSource: string = 'upload',
  datasetUrl: string = '',
): Promise<any> {
  const fd = new FormData()
  if (datasetFile) fd.append('dataset_file', datasetFile)
  fd.append('protected_cols', protectedCols.length > 0 ? protectedCols.join(',') : 'auto')
  fd.append('target_col', targetCol || 'auto')
  fd.append('dataset_source', datasetSource)
  if (datasetUrl) fd.append('dataset_url', datasetUrl)
  const res = await fetch(`${BASE}/api/v1/dataset-probe/run`, { method: 'POST', body: fd })
  if (!res.ok) throw new Error(await res.text())
  return res.json()
}

// ── Phase 3: Cross-analysis (model × user dataset) ───────────────────────────
export async function runCrossAnalysis(
  modelProbeResults: any,
  datasetProbeResults: any,
  modelFile: File | null,
  datasetFile: File | null,
  modelType: string = 'sklearn',
  apiEndpoint: string = '',
  llmApiKey: string = '',
  hfToken: string = '',
  datasetSource: string = 'upload',
  datasetUrl: string = '',
  protectedCols: string[] = [],
  targetCol: string = 'auto',
): Promise<any> {
  const fd = new FormData()
  fd.append('model_probe_results',   JSON.stringify(modelProbeResults))
  fd.append('dataset_probe_results', JSON.stringify(datasetProbeResults))
  if (modelFile)   fd.append('model_file',   modelFile)
  if (datasetFile) fd.append('dataset_file', datasetFile)
  fd.append('model_type',     modelType)
  if (apiEndpoint) fd.append('api_endpoint', apiEndpoint)
  if (llmApiKey)   fd.append('llm_api_key',  llmApiKey)
  if (hfToken)     fd.append('hf_token',     hfToken)
  fd.append('dataset_source', datasetSource)
  if (datasetUrl) fd.append('dataset_url', datasetUrl)
  fd.append('protected_cols', protectedCols.length > 0 ? protectedCols.join(',') : 'auto')
  fd.append('target_col', targetCol || 'auto')
  const res = await fetch(`${BASE}/api/v1/cross-analysis/run`, { method: 'POST', body: fd })
  if (!res.ok) throw new Error(await res.text())
  return res.json()
}

export async function exportPdfReport(result: any): Promise<void> {
  const res = await fetch(`${BASE}/api/v1/reports/pdf`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(result),
  })
  if (!res.ok) throw new Error(await res.text())
  const blob = await res.blob()
  const url  = URL.createObjectURL(blob)
  const a    = document.createElement('a')
  a.href     = url
  a.download = `fairlens-report-${result?.audit_id ?? 'audit'}.pdf`
  document.body.appendChild(a)
  a.click()
  document.body.removeChild(a)
  URL.revokeObjectURL(url)
}

export function streamRedTeam(
  modelFile: File | null,
  datasetFile: File | null,
  protectedCols: string[],
  targetCol: string,
  confirmedBiases: any[],
  auditResults: any,
  onEvent: (e: any) => void,
  datasetSource: string = 'upload',
  datasetUrl: string = '',
  modelType: string = 'sklearn',
  apiEndpoint: string = '',
  llmApiKey: string = '',
  hfToken: string = '',
  modelProbeBiases: any[] = [],
  datasetProbeBiases: any[] = [],
): () => void {
  const fd = new FormData()
  if (modelFile)   fd.append('model_file',   modelFile)
  if (datasetFile) fd.append('dataset_file', datasetFile)
  fd.append('protected_cols', protectedCols.length > 0 ? protectedCols.join(',') : 'auto')
  fd.append('target_col', targetCol || 'auto')
  fd.append('confirmed_biases', JSON.stringify(confirmedBiases))
  fd.append('model_probe_biases',   JSON.stringify(modelProbeBiases))
  fd.append('dataset_probe_biases', JSON.stringify(datasetProbeBiases))

  // Include slice_metrics so the red-team cartography fallback has SPD data
  const auditSummary = {
    cartography: {
      summary:      auditResults?.crossAnalysis?.cartography?.summary,
      audit_id:     auditResults?.crossAnalysis?.cartography?.audit_id,
      slice_metrics: auditResults?.crossAnalysis?.cartography?.slice_metrics ?? [],
    },
    constitution: { summary: auditResults?.crossAnalysis?.constitution?.summary },
    proxy:        { summary: auditResults?.crossAnalysis?.proxy?.summary },
    model_probe:  { biases: modelProbeBiases },
    dataset_probe: { biases: datasetProbeBiases },
  }
  fd.append('audit_results', JSON.stringify(auditSummary))
  fd.append('dataset_source', datasetSource)
  if (datasetUrl)  fd.append('dataset_url',  datasetUrl)
  fd.append('model_type', modelType)
  if (apiEndpoint) fd.append('api_endpoint', apiEndpoint)
  if (llmApiKey)   fd.append('llm_api_key',  llmApiKey)
  if (hfToken)     fd.append('hf_token',     hfToken)

  const controller = new AbortController()
  fetch(`${BASE}/api/v1/redteam/run`, { method: 'POST', body: fd, signal: controller.signal })
    .then(async (res) => {
      if (!res.ok) {
        const errText = await res.text()
        onEvent({ node: 'error', status: 'error', log: [`Error: ${errText}`] })
        return
      }
      const reader = res.body!.getReader()
      const dec    = new TextDecoder()
      // Buffer across chunks — SSE events can be split across multiple read() calls
      let buffer = ''
      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        buffer += dec.decode(value, { stream: true })
        // SSE events are delimited by double newline
        const parts = buffer.split('\n\n')
        buffer = parts.pop() ?? ''   // last part may be incomplete — keep buffering
        for (const part of parts) {
          for (const line of part.split('\n')) {
            if (line.startsWith('data: ')) {
              try { onEvent(JSON.parse(line.slice(6))) } catch {}
            }
          }
        }
      }
      // Flush any remaining buffer after stream ends
      for (const line of buffer.split('\n')) {
        if (line.startsWith('data: ')) {
          try { onEvent(JSON.parse(line.slice(6))) } catch {}
        }
      }
    }).catch(() => {})
  return () => controller.abort()
}
