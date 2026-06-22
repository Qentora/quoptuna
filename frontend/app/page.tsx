'use client';

import { Badge } from '@/components/ui/badge';
import { Card, CardContent } from '@/components/ui/card';
import { PageHeader } from '@/components/ui/page-header';
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
    <div className="mx-auto max-w-6xl p-6 md:p-8">
      <PageHeader
        title="QuOptuna"
        subtitle="Quantum-enhanced machine learning with automated hyperparameter optimization"
        actions={<StatusPill online={online} loading={infoLoading || !infoFetched} />}
        className="mb-8"
      />

      <div className="mb-10 grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <StatCard
          title="Total Models"
          value={totalModels}
          subtitle="Quantum + classical"
          icon={<Brain className="h-5 w-5 text-accent-purple-foreground" />}
        />
        <StatCard
          title="Quantum Models"
          value={info?.quantum_models ?? 18}
          subtitle="Variational & kernel"
          icon={<Zap className="h-5 w-5 text-accent-indigo-foreground" />}
        />
        <StatCard
          title="Classical Models"
          value={info?.classical_models ?? 8}
          subtitle="Baselines"
          icon={<BarChart3 className="h-5 w-5 text-accent-orange-foreground" />}
        />
        <StatCard
          title="UCI Datasets"
          value={datasets.length}
          subtitle="Ready to load"
          icon={<Database className="h-5 w-5 text-accent-emerald-foreground" />}
        />
      </div>

      <div className="mb-10">
        <h2 className="mb-4 text-lg font-semibold tracking-tight">Get Started</h2>
        <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
          <Link
            href="/optimizer"
            className="group rounded-lg bg-primary p-6 text-primary-foreground transition-colors hover:bg-primary-hover"
          >
            <Zap className="mb-3 h-8 w-8" />
            <h3 className="mb-1 text-base font-semibold">Open the Optimizer</h3>
            <p className="text-sm opacity-90">
              Load data, run hyperparameter optimization, explore SHAP analysis and generate an AI
              report.
            </p>
          </Link>
          <Link
            href="/settings"
            className="group rounded-lg border border-border bg-card p-6 transition-colors hover:bg-accent"
          >
            <Settings className="mb-3 h-8 w-8" />
            <h3 className="mb-1 text-base font-semibold">Configure API Keys</h3>
            <p className="text-sm text-muted-foreground">
              Add your OpenAI, Anthropic or Google keys to enable AI-generated reports. Stored
              encrypted in your browser.
            </p>
          </Link>
        </div>
      </div>

      <div>
        <div className="mb-4 flex items-center gap-3">
          <h2 className="text-lg font-semibold tracking-tight">Available UCI Datasets</h2>
          {datasets.length > 0 && <Badge variant="secondary">{datasets.length}</Badge>}
        </div>
        {datasets.length === 0 ? (
          <Card className="border-dashed">
            <CardContent className="p-8 text-center">
              <Database className="mx-auto mb-3 h-8 w-8 text-muted-foreground" />
              <p className="font-medium">No datasets available</p>
              <p className="mt-1 text-sm text-muted-foreground">
                Could not reach the backend. Start it and refresh to see available datasets.
              </p>
            </CardContent>
          </Card>
        ) : (
          <div className="grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-3">
            {datasets.map((d) => (
              <Card key={d.id} className="transition-colors hover:border-foreground/30">
                <CardContent className="p-4">
                  <p className="font-semibold">{d.name}</p>
                  {d.description && (
                    <p className="mt-1 line-clamp-2 text-sm text-muted-foreground">
                      {d.description}
                    </p>
                  )}
                  <div className="mt-3 flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-muted-foreground">
                    <span>UCI ID: {d.id}</span>
                    {typeof d.num_instances === 'number' && <span>{d.num_instances} rows</span>}
                    {typeof d.num_features === 'number' && <span>{d.num_features} features</span>}
                  </div>
                </CardContent>
              </Card>
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
      <Badge variant="secondary" size="md" className="font-medium">
        <span className="h-2 w-2 animate-pulse rounded-full bg-muted-foreground/50" />
        Connecting…
      </Badge>
    );
  }
  return (
    <Badge variant={online ? 'emerald' : 'amber'} size="md" className="font-medium">
      <span
        className={`h-2 w-2 rounded-full ${online ? 'bg-accent-emerald-foreground' : 'bg-accent-amber-foreground'}`}
      />
      {online ? 'Backend online' : 'Backend offline'}
    </Badge>
  );
}

function StatCard({
  title,
  value,
  subtitle,
  icon,
}: {
  title: string;
  value: number;
  subtitle: string;
  icon: React.ReactNode;
}) {
  return (
    <Card className="transition-colors hover:border-foreground/30">
      <CardContent className="p-6">
        <div className="mb-3 flex items-center justify-between">
          <h3 className="text-sm font-medium text-muted-foreground">{title}</h3>
          <span className="flex h-9 w-9 items-center justify-center rounded-md bg-muted">
            {icon}
          </span>
        </div>
        <p className="text-3xl font-bold">{value}</p>
        <p className="mt-1 text-sm text-muted-foreground">{subtitle}</p>
      </CardContent>
    </Card>
  );
}
