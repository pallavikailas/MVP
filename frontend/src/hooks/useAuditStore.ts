import { create } from 'zustand'

export interface AuditSession {
  modelFile: File | null
  datasetFile: File | null
  datasetSource: string
  datasetUrl: string
  modelType: string
  modelEndpoint: string
  llmApiKey: string
  hfToken: string
  protectedCols: string[]
  targetCol: string

  // Phase 1 — model probe on embedded reference dataset
  modelProbeResults: any | null
  // Phase 2 — dataset-only bias analysis
  datasetProbeResults: any | null
  // Phase 3 — cross-analysis (model × user dataset)
  crossAnalysisResults: any | null
  // Phase 4 — red-team
  redteamResults: any | null

  // Legacy fields kept for compatibility with existing result displays
  cartographyResults: any | null
  constitutionResults: any | null
  proxyResults: any | null

  confirmedBiases: any[]
  stage: 'upload' | 'model_probe' | 'dataset_probe' | 'cross_analysis' | 'review' | 'redteam' | 'done'
  activeSubStage: string | null
  loading: boolean
  error: string | null
}

interface AuditStore extends AuditSession {
  setModelFile: (f: File | null) => void
  setDatasetFile: (f: File | null) => void
  setDatasetSource: (s: string) => void
  setDatasetUrl: (u: string) => void
  setModelType: (t: string) => void
  setModelEndpoint: (e: string) => void
  setLlmApiKey: (k: string) => void
  setHfToken: (t: string) => void
  setProtectedCols: (cols: string[]) => void
  setTargetCol: (col: string) => void
  setModelProbeResults: (r: any) => void
  setDatasetProbeResults: (r: any) => void
  setCrossAnalysisResults: (r: any) => void
  setRedteamResults: (r: any) => void
  // Legacy setters
  setCartographyResults: (r: any) => void
  setConstitutionResults: (r: any) => void
  setProxyResults: (r: any) => void
  setConfirmedBiases: (b: any[]) => void
  setStage: (s: AuditSession['stage']) => void
  setActiveSubStage: (s: string | null) => void
  setLoading: (v: boolean) => void
  setError: (e: string | null) => void
  reset: () => void
}

const initial: AuditSession = {
  modelFile: null,
  datasetFile: null,
  datasetSource: 'upload',
  datasetUrl: '',
  modelType: 'sklearn',
  modelEndpoint: '',
  llmApiKey: '',
  hfToken: '',
  protectedCols: [],
  targetCol: '',
  modelProbeResults: null,
  datasetProbeResults: null,
  crossAnalysisResults: null,
  redteamResults: null,
  cartographyResults: null,
  constitutionResults: null,
  proxyResults: null,
  confirmedBiases: [],
  stage: 'upload',
  activeSubStage: null,
  loading: false,
  error: null,
}

export const useAuditStore = create<AuditStore>((set) => ({
  ...initial,
  setModelFile:            (modelFile)            => set({ modelFile }),
  setDatasetFile:          (datasetFile)          => set({ datasetFile }),
  setDatasetSource:        (datasetSource)        => set({ datasetSource }),
  setDatasetUrl:           (datasetUrl)           => set({ datasetUrl }),
  setModelType:            (modelType)            => set({ modelType }),
  setModelEndpoint:        (modelEndpoint)        => set({ modelEndpoint }),
  setLlmApiKey:            (llmApiKey)            => set({ llmApiKey }),
  setHfToken:              (hfToken)              => set({ hfToken }),
  setProtectedCols:        (protectedCols)        => set({ protectedCols }),
  setTargetCol:            (targetCol)            => set({ targetCol }),
  setModelProbeResults:    (modelProbeResults)    => set({ modelProbeResults }),
  setDatasetProbeResults:  (datasetProbeResults)  => set({ datasetProbeResults }),
  setCrossAnalysisResults: (crossAnalysisResults) => set({ crossAnalysisResults }),
  setRedteamResults:       (redteamResults)       => set({ redteamResults }),
  setCartographyResults:   (cartographyResults)   => set({ cartographyResults }),
  setConstitutionResults:  (constitutionResults)  => set({ constitutionResults }),
  setProxyResults:         (proxyResults)         => set({ proxyResults }),
  setConfirmedBiases:      (confirmedBiases)      => set({ confirmedBiases }),
  setStage:                (stage)                => set({ stage }),
  setActiveSubStage:       (activeSubStage)       => set({ activeSubStage }),
  setLoading:              (loading)              => set({ loading }),
  setError:                (error)                => set({ error }),
  reset:                   ()                     => set(initial),
}))
