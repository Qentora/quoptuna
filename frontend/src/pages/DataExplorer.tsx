import { Upload, Database } from 'lucide-react';

export function DataExplorer() {
  return (
    <div className="p-8">
      <h1 className="text-3xl font-bold mb-8">Data Explorer</h1>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
        <div className="bg-card p-6 rounded-lg border border-border">
          <div className="flex items-center gap-3 mb-4">
            <Upload className="w-6 h-6 text-blue-500" />
            <h2 className="text-xl font-semibold">Upload Dataset</h2>
          </div>
          <p className="text-muted-foreground mb-4">
            Upload a CSV file to use in your quantum ML workflows
          </p>
          <div className="border-2 border-dashed border-border rounded-lg p-12 text-center hover:border-blue-500 transition-colors cursor-pointer">
            <Upload className="w-12 h-12 mx-auto mb-4 text-muted-foreground" />
            <p className="font-medium mb-2">Drop your CSV file here</p>
            <p className="text-sm text-muted-foreground">or click to browse</p>
          </div>
        </div>

        <div className="bg-card p-6 rounded-lg border border-border">
          <div className="flex items-center gap-3 mb-4">
            <Database className="w-6 h-6 text-green-500" />
            <h2 className="text-xl font-semibold">UCI Repository</h2>
          </div>
          <p className="text-muted-foreground mb-4">
            Browse and fetch datasets from the UCI Machine Learning Repository
          </p>
          <input
            type="text"
            placeholder="Search datasets..."
            className="w-full px-4 py-2 border border-border rounded-md mb-4"
          />
          <div className="space-y-2 max-h-64 overflow-y-auto">
            <div className="p-3 bg-secondary rounded-md hover:bg-accent transition-colors cursor-pointer">
              <p className="font-medium">Iris Dataset</p>
              <p className="text-sm text-muted-foreground">150 samples, 4 features</p>
            </div>
            <div className="p-3 bg-secondary rounded-md hover:bg-accent transition-colors cursor-pointer">
              <p className="font-medium">Wine Quality</p>
              <p className="text-sm text-muted-foreground">1599 samples, 11 features</p>
            </div>
            <div className="p-3 bg-secondary rounded-md hover:bg-accent transition-colors cursor-pointer">
              <p className="font-medium">Breast Cancer</p>
              <p className="text-sm text-muted-foreground">569 samples, 30 features</p>
            </div>
          </div>
        </div>
      </div>

      <div>
        <h2 className="text-2xl font-bold mb-4">Uploaded Datasets</h2>
        <div className="bg-card p-6 rounded-lg border border-border">
          <div className="text-center text-muted-foreground py-12">
            No datasets uploaded yet. Upload a CSV or fetch from UCI to get started!
          </div>
        </div>
      </div>
    </div>
  );
}
