import { Routes, Route, Navigate } from 'react-router-dom'
import Layout from './components/Layout'
import Installation from './pages/Installation'
import GetStarted from './pages/GetStarted'
import Parameters from './pages/Parameters'
import BoosterParameters from './pages/BoosterParameters'
import TreeMethods from './pages/TreeMethods'
import FAQ from './pages/FAQ'
import GPUSupport from './pages/GPUSupport'

function App() {
  return (
    <Routes>
      <Route path="/" element={<Layout />}>
        <Route index element={<Navigate to="/parameters" replace />} />
        <Route path="installation" element={<Installation />} />
        <Route path="get-started" element={<GetStarted />} />
        <Route path="parameters" element={<Parameters />} />
        <Route path="booster-parameters" element={<BoosterParameters />} />
        <Route path="tree-methods" element={<TreeMethods />} />
        <Route path="faq" element={<FAQ />} />
        <Route path="gpu-support" element={<GPUSupport />} />
      </Route>
    </Routes>
  )
}

export default App
