import { Link } from 'react-router-dom';
import { Workflow, Database, BarChart3, Brain } from 'lucide-react';

export function Dashboard() {
  return (
    <div className="p-8">
      <h1 className="text-3xl font-bold mb-8">QuOptuna Next</h1>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
        <div className="bg-card p-6 rounded-lg border border-border">
          <div className="flex items-center justify-between mb-2">
            <h3 className="text-lg font-semibold">Workflows</h3>
            <Workflow className="w-5 h-5 text-blue-500" />
          </div>
          <p className="text-3xl font-bold">0</p>
          <p className="text-sm text-muted-foreground">Total workflows</p>
        </div>

        <div className="bg-card p-6 rounded-lg border border-border">
          <div className="flex items-center justify-between mb-2">
            <h3 className="text-lg font-semibold">Datasets</h3>
            <Database className="w-5 h-5 text-green-500" />
          </div>
          <p className="text-3xl font-bold">0</p>
          <p className="text-sm text-muted-foreground">Uploaded datasets</p>
        </div>

        <div className="bg-card p-6 rounded-lg border border-border">
          <div className="flex items-center justify-between mb-2">
            <h3 className="text-lg font-semibold">Models</h3>
            <Brain className="w-5 h-5 text-purple-500" />
          </div>
          <p className="text-3xl font-bold">26</p>
          <p className="text-sm text-muted-foreground">Available models</p>
        </div>

        <div className="bg-card p-6 rounded-lg border border-border">
          <div className="flex items-center justify-between mb-2">
            <h3 className="text-lg font-semibold">Experiments</h3>
            <BarChart3 className="w-5 h-5 text-orange-500" />
          </div>
          <p className="text-3xl font-bold">0</p>
          <p className="text-sm text-muted-foreground">Completed runs</p>
        </div>
      </div>

      <div className="mb-8">
        <h2 className="text-2xl font-bold mb-4">Quick Actions</h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <Link
            to="/workflow"
            className="bg-blue-600 hover:bg-blue-700 text-white p-6 rounded-lg transition-colors"
          >
            <Workflow className="w-8 h-8 mb-2" />
            <h3 className="text-lg font-semibold mb-1">Create Workflow</h3>
            <p className="text-sm opacity-90">Build a new ML pipeline with drag & drop</p>
          </Link>

          <Link
            to="/data"
            className="bg-green-600 hover:bg-green-700 text-white p-6 rounded-lg transition-colors"
          >
            <Database className="w-8 h-8 mb-2" />
            <h3 className="text-lg font-semibold mb-1">Upload Data</h3>
            <p className="text-sm opacity-90">Import CSV or fetch from UCI repository</p>
          </Link>

          <Link
            to="/analytics"
            className="bg-purple-600 hover:bg-purple-700 text-white p-6 rounded-lg transition-colors"
          >
            <BarChart3 className="w-8 h-8 mb-2" />
            <h3 className="text-lg font-semibold mb-1">View Analytics</h3>
            <p className="text-sm opacity-90">Explore SHAP analysis and insights</p>
          </Link>
        </div>
      </div>

      <div>
        <h2 className="text-2xl font-bold mb-4">Recent Workflows</h2>
        <div className="bg-card p-6 rounded-lg border border-border text-center text-muted-foreground">
          No workflows yet. Create your first workflow to get started!
        </div>
      </div>
    </div>
  );
}
