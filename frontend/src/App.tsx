import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { Layout } from './components/layout/Layout';
import { Dashboard } from './pages/Dashboard';
import { WorkflowBuilder } from './pages/WorkflowBuilder';
import { DataExplorer } from './pages/DataExplorer';
import { Models } from './pages/Models';
import { Analytics } from './pages/Analytics';
import { Settings } from './pages/Settings';

const queryClient = new QueryClient();

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<Layout />}>
            <Route index element={<Dashboard />} />
            <Route path="workflow" element={<WorkflowBuilder />} />
            <Route path="data" element={<DataExplorer />} />
            <Route path="models" element={<Models />} />
            <Route path="analytics" element={<Analytics />} />
            <Route path="settings" element={<Settings />} />
          </Route>
        </Routes>
      </BrowserRouter>
    </QueryClientProvider>
  );
}

export default App;
