import { BarChart3 } from 'lucide-react';

export function Analytics() {
  return (
    <div className="p-8">
      <h1 className="text-3xl font-bold mb-8">Analytics & Insights</h1>

      <div className="bg-card p-12 rounded-lg border border-border text-center">
        <BarChart3 className="w-16 h-16 mx-auto mb-4 text-muted-foreground" />
        <h2 className="text-2xl font-semibold mb-2">No Analysis Available</h2>
        <p className="text-muted-foreground mb-6">
          Run an optimization workflow with SHAP analysis to see insights here
        </p>
        <button className="px-6 py-3 bg-primary text-primary-foreground rounded-md hover:bg-primary/90 transition-colors">
          Go to Workflow Builder
        </button>
      </div>
    </div>
  );
}
