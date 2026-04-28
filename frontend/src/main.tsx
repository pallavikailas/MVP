import React from 'react'
import ReactDOM from 'react-dom/client'
import { BrowserRouter, Routes, Route } from 'react-router-dom'
import './index.css'
import LandingPage from './pages/LandingPage'
import AuditPage from './pages/AuditPage'
import ResultsPage from './pages/ResultsPage'
import RedTeamPage from './pages/RedTeamPage'
import Layout from './components/Layout'

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <BrowserRouter basename="/FairLens">
      <Routes>
        <Route path="/" element={<LandingPage />} />
        <Route element={<Layout />}>
          <Route path="/audit" element={<AuditPage />} />
          <Route path="/results" element={<ResultsPage />} />
          <Route path="/redteam" element={<RedTeamPage />} />
        </Route>
      </Routes>
    </BrowserRouter>
  </React.StrictMode>
)
