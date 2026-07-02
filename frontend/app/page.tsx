'use client';

import { Card, CardContent } from '@/components/ui/card';
import { Metric } from '@/components/ui/metric';
import { PageShell } from '@/components/ui/page-shell';
import {
  type PastRun,
  type UCIDataset,
  getSystemInfo,
  listOptimizations,
  listUCIDatasets,
} from '@/lib/api';
import { useQuery } from '@tanstack/react-query';
import {
  BarChart3,
  Brain,
  CheckCircle2,
  Database,
  History,
  Loader2,
  PauseCircle,
  Settings,
  Zap,
} from 'lucide-react';
import Link from 'next/link';

const MAX_DATASETS = 6;

export default function HomePage() {
  const { data: info } = useQuery({
    queryKey: ['system-info'],
    queryFn: () => getSystemInfo().catch(() => null),
  });
  const { data: datasets = [] } = useQuery({
    queryKey: ['uci-datasets'],
    queryFn: () => listUCIDatasets().catch(() => [] as UCIDataset[]),
  });
  const { data: runs = [] } = useQuery({
    queryKey: ['optimization-runs'],
    queryFn: () => listOptimizations().catch(() => [] as PastRun[]),
  });
  const totalModels = info?.total_models ?? 26;
  const shownDatasets = datasets.slice(0, MAX_DATASETS);
  const remaining = datasets.length - shownDatasets.length;
  const completedStudies = runs.filter((r) => r.status === 'completed').length;
  const inProgressStudies = runs.filter(
    (r) => r.status === 'running' || r.status === 'pending'
  ).length;
  const otherStudies = runs.length - completedStudies - inProgressStudies;

  return (
    <PageShell title="Dashboard" contentClassName="mx-auto max-w-6xl">
      <div className="mb-6 grid grid-cols-2 gap-3 lg:grid-cols-4">
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
          icon={<Zap className="h-5 w-5 text-brand" />}
          tone="brand"
        />
        <StatCard
          title="Classical Models"
          value={info?.classical_models ?? 8}
          subtitle="Baselines"
          icon={<BarChart3 className="h-5 w-5 text-accent-orange-foreground" />}
          tone="amber"
        />
        <StatCard
          title="UCI Datasets"
          value={datasets.length}
          subtitle="Ready to load"
          icon={<Database className="h-5 w-5 text-accent-emerald-foreground" />}
          tone="emerald"
        />
      </div>

      <div className="mb-6">
        <div className="mb-3 flex items-center justify-between">
          <h2 className="text-base font-semibold tracking-tight">Studies</h2>
          <Link
            href="/runs"
            className="flex items-center gap-1 text-sm text-muted-foreground hover:text-foreground hover:underline"
          >
            <History className="h-4 w-4" />
            View all runs
          </Link>
        </div>
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
          <StatCard
            title="Completed Studies"
            value={completedStudies}
            subtitle="Ready to analyze & replay"
            icon={<CheckCircle2 className="h-5 w-5 text-accent-emerald-foreground" />}
            tone="emerald"
          />
          <StatCard
            title="In Progress"
            value={inProgressStudies}
            subtitle="Currently optimizing"
            icon={<Loader2 className="h-5 w-5 text-brand" />}
            tone="brand"
          />
          <StatCard
            title="Interrupted / Failed"
            value={otherStudies}
            subtitle="Can be restarted from Runs"
            icon={<PauseCircle className="h-5 w-5 text-accent-orange-foreground" />}
            tone="amber"
          />
        </div>
      </div>

      <div className="mb-6">
        <h2 className="mb-3 text-base font-semibold tracking-tight">Get Started</h2>
        <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
          <Link
            href="/optimizer"
            className="group rounded-lg bg-gradient-to-br from-brand to-brand/70 p-5 text-brand-foreground transition-shadow hover:shadow-glow-brand"
          >
            <Zap className="mb-2 h-7 w-7" />
            <h3 className="mb-1 text-base font-semibold">Open the Optimizer</h3>
            <p className="text-sm opacity-90">
              Load data, run hyperparameter optimization, explore SHAP analysis and generate an AI
              report.
            </p>
          </Link>
          <Link
            href="/settings"
            className="group rounded-lg border border-border bg-card p-5 transition-colors hover:bg-accent"
          >
            <Settings className="mb-2 h-7 w-7" />
            <h3 className="mb-1 text-base font-semibold">Configure API Keys</h3>
            <p className="text-sm text-muted-foreground">
              Add your OpenAI, Anthropic or Google keys to enable AI-generated reports. Stored
              encrypted in your browser.
            </p>
          </Link>
        </div>
      </div>

      <div>
        <h2 className="mb-3 text-base font-semibold tracking-tight">Available UCI Datasets</h2>
        {datasets.length === 0 ? (
          <Card className="border-dashed">
            <CardContent className="p-6 text-center">
              <Database className="mx-auto mb-3 h-7 w-7 text-muted-foreground" />
              <p className="font-medium">No datasets available</p>
              <p className="mt-1 text-sm text-muted-foreground">
                Could not reach the backend. Start it and refresh to see available datasets.
              </p>
            </CardContent>
          </Card>
        ) : (
          <>
            <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
              {shownDatasets.map((d) => (
                <Card key={d.id} className="transition-colors hover:border-foreground/30">
                  <CardContent className="p-3">
                    <p className="truncate font-medium">{d.name}</p>
                    <div className="mt-2 flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-muted-foreground">
                      <span>UCI ID: {d.id}</span>
                      {typeof d.num_instances === 'number' && <span>{d.num_instances} rows</span>}
                      {typeof d.num_features === 'number' && <span>{d.num_features} features</span>}
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
            {remaining > 0 && (
              <Link
                href="/optimizer"
                className="mt-3 inline-block text-sm text-muted-foreground hover:text-foreground hover:underline"
              >
                +{remaining} more — open the Optimizer to browse all
              </Link>
            )}
          </>
        )}
      </div>
    </PageShell>
  );
}

function StatCard({
  title,
  value,
  subtitle,
  icon,
  tone = 'default',
}: {
  title: string;
  value: number;
  subtitle: string;
  icon: React.ReactNode;
  tone?: 'default' | 'emerald' | 'brand' | 'amber';
}) {
  return (
    <Card className="transition-colors hover:border-foreground/30">
      <CardContent className="p-4">
        <div className="mb-2 flex items-center justify-between">
          <h3 className="text-sm font-medium text-muted-foreground">{title}</h3>
          <span className="flex h-8 w-8 items-center justify-center rounded-md bg-muted">
            {icon}
          </span>
        </div>
        <Metric value={value} tone={tone} className="block text-2xl" />
        <p className="mt-0.5 text-xs text-muted-foreground">{subtitle}</p>
      </CardContent>
    </Card>
  );
}
