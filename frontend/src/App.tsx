import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { Toaster } from 'react-hot-toast';
import { Layout } from './components/layout/Layout';
import { Dashboard } from './pages/Dashboard';
import { WorkflowBuilder } from './pages/WorkflowBuilder';
import { Optimizer } from './pages/Optimizer';
import { DataExplorer } from './pages/DataExplorer';
import { Models } from './pages/Models';
import { Analytics } from './pages/Analytics';
import { Settings } from './pages/Settings';
import { FEATURES } from './config/features';

const queryClient = new QueryClient();

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<Layout />}>
            <Route index element={<Dashboard />} />
            {FEATURES.WORKFLOW_BUILDER && <Route path="workflow" element={<WorkflowBuilder />} />}
            {FEATURES.STEP_OPTIMIZER && <Route path="optimizer" element={<Optimizer />} />}
            {FEATURES.DATA_EXPLORER && <Route path="data" element={<DataExplorer />} />}
            {FEATURES.MODELS && <Route path="models" element={<Models />} />}
            {FEATURES.ANALYTICS && <Route path="analytics" element={<Analytics />} />}
            <Route path="settings" element={<Settings />} />
          </Route>
        </Routes>
      </BrowserRouter>
      <Toaster
        position="top-right"
        toastOptions={{
          duration: 4000,
          style: {
            background: '#fff',
            color: '#333',
          },
          success: {
            duration: 3000,
            iconTheme: {
              primary: '#10b981',
              secondary: '#fff',
            },
          },
          error: {
            duration: 5000,
            iconTheme: {
              primary: '#ef4444',
              secondary: '#fff',
            },
          },
        }}
      />
    </QueryClientProvider>
  );
}

export default App;
