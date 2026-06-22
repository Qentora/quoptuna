'use client';

import { type UCIDataset, getSystemInfo, listUCIDatasets } from '@/lib/api';
import { useQuery } from '@tanstack/react-query';
import { BarChart3, Brain, Database, Settings, Zap } from 'lucide-react';
import Link from 'next/link';

export default function HomePage() {
  const {
    data: info,
    isLoading: infoLoading,
    isFetched: infoFetched,
  } = useQuery({
    queryKey: ['system-info'],
    queryFn: () => getSystemInfo().catch(() => null),
  });
  const { data: datasets = [] } = useQuery({
    queryKey: ['uci-datasets'],
    queryFn: () => listUCIDatasets().catch(() => [] as UCIDataset[]),
  });
  const totalModels = info?.total_models ?? 26;
  const online = Boolean(info);

  return (
    <div className="p-8 max-w-7xl mx-auto">
      <div className="flex flex-wrap items-start justify-between gap-4 mb-8">
        <div>
          <h1 className="text-3xl font-bold mb-2">QuOptuna</h1>
          <p className="text-muted-foreground">
            Quantum-enhanced machine learning with automated hyperparameter optimization
          </p>
        </div>
        <StatusPill online={online} loading={infoLoading || !infoFetched} />
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-10">
        <StatCard
          title="Total Models"
          value={totalModels}
          subtitle="Quantum + classical"
          icon={<Brain className="w-5 h-5 text-purple-500" />}
          tint="bg-purple-500/10"
        />
        <StatCard
          title="Quantum Models"
          value={info?.quantum_models ?? 18}
          subtitle="Variational & kernel"
          icon={<Zap className="w-5 h-5 text-blue-500" />}
          tint="bg-blue-500/10"
        />
        <StatCard
          title="Classical Models"
          value={info?.classical_models ?? 8}
          subtitle="Baselines"
          icon={<BarChart3 className="w-5 h-5 text-orange-500" />}
          tint="bg-orange-500/10"
        />
        <StatCard
          title="UCI Datasets"
          value={datasets.length}
          subtitle="Ready to load"
          icon={<Database className="w-5 h-5 text-green-500" />}
          tint="bg-green-500/10"
        />
      </div>

      <div className="mb-10">
        <h2 className="text-2xl font-bold mb-4">Get Started</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <Link
            href="/optimizer"
            className="group flex flex-col bg-primary hover:bg-primary/90 text-primary-foreground p-6 rounded-lg transition-colors"
          >
            <Zap className="w-8 h-8 mb-3" />
            <h3 className="text-lg font-semibold mb-1">Open the Optimizer</h3>
            <p className="text-sm opacity-90">
              Load data, run hyperparameter optimization, explore SHAP analysis and generate an AI
              report.
            </p>
          </Link>
          <Link
            href="/settings"
            className="group flex flex-col bg-card hover:border-primary/40 border border-border p-6 rounded-lg transition-colors"
          >
            <Settings className="w-8 h-8 mb-3 text-primary" />
            <h3 className="text-lg font-semibold mb-1">Configure API Keys</h3>
            <p className="text-sm text-muted-foreground">
              Add your OpenAI, Anthropic or Google keys to enable AI-generated reports. Stored
              encrypted in your browser.
            </p>
          </Link>
        </div>
      </div>

      <div>
        <div className="flex items-center gap-3 mb-4">
          <h2 className="text-2xl font-bold">Available UCI Datasets</h2>
          {datasets.length > 0 && (
            <span className="rounded-full bg-secondary px-2.5 py-0.5 text-sm font-medium text-secondary-foreground">
              {datasets.length}
            </span>
          )}
        </div>
        {datasets.length === 0 ? (
          <div className="bg-card p-8 rounded-lg border border-dashed border-border text-center">
            <Database className="w-8 h-8 mx-auto mb-3 text-muted-foreground" />
            <p className="font-medium">No datasets available</p>
            <p className="text-sm text-muted-foreground mt-1">
              Could not reach the backend. Start it and refresh to see available datasets.
            </p>
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {datasets.map((d) => (
              <div
                key={d.id}
                className="bg-card p-4 rounded-lg border border-border transition-all hover:border-primary/40 hover:shadow-sm"
              >
                <p className="font-semibold">{d.name}</p>
                {d.description && (
                  <p className="text-sm text-muted-foreground mt-1 line-clamp-2">{d.description}</p>
                )}
                <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-muted-foreground mt-3">
                  <span>UCI ID: {d.id}</span>
                  {typeof d.num_instances === 'number' && <span>{d.num_instances} rows</span>}
                  {typeof d.num_features === 'number' && <span>{d.num_features} features</span>}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

function StatusPill({ online, loading }: { online: boolean; loading: boolean }) {
  if (loading) {
    return (
      <span className="inline-flex items-center gap-2 rounded-full border border-border bg-card px-3 py-1.5 text-sm text-muted-foreground">
        <span className="h-2 w-2 rounded-full bg-muted-foreground/50 animate-pulse" />
        Connecting…
      </span>
    );
  }
  return (
    <span
      className={`inline-flex items-center gap-2 rounded-full border px-3 py-1.5 text-sm ${
        online
          ? 'border-green-500/30 bg-green-500/10 text-green-600'
          : 'border-amber-500/30 bg-amber-500/10 text-amber-600'
      }`}
    >
      <span className={`h-2 w-2 rounded-full ${online ? 'bg-green-500' : 'bg-amber-500'}`} />
      {online ? 'Backend online' : 'Backend offline'}
    </span>
  );
}

function StatCard({
  title,
  value,
  subtitle,
  icon,
  tint,
}: {
  title: string;
  value: number;
  subtitle: string;
  icon: React.ReactNode;
  tint: string;
}) {
  return (
    <div className="bg-card p-6 rounded-lg border border-border transition-all hover:border-primary/40 hover:shadow-sm">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-medium text-muted-foreground">{title}</h3>
        <span className={`flex h-9 w-9 items-center justify-center rounded-md ${tint}`}>{icon}</span>
      </div>
      <p className="text-3xl font-bold">{value}</p>
      <p className="text-sm text-muted-foreground mt-1">{subtitle}</p>
    </div>
  );
}
