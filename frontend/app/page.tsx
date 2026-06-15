import { type SystemInfo, type UCIDataset, getSystemInfo, listUCIDatasets } from '@/lib/api';
import { BarChart3, Brain, Database, Zap } from 'lucide-react';
import Link from 'next/link';

// Re-fetch lightweight dashboard data on each request (SSR).
export const dynamic = 'force-dynamic';

async function loadDashboardData(): Promise<{
  info: SystemInfo | null;
  datasets: UCIDataset[];
}> {
  const [info, datasets] = await Promise.all([
    getSystemInfo().catch(() => null),
    listUCIDatasets().catch(() => [] as UCIDataset[]),
  ]);
  return { info, datasets };
}

export default async function HomePage() {
  const { info, datasets } = await loadDashboardData();
  const totalModels = info?.total_models ?? 26;

  return (
    <div className="p-8">
      <h1 className="text-3xl font-bold mb-2">QuOptuna</h1>
      <p className="text-muted-foreground mb-8">
        Quantum-enhanced machine learning with automated hyperparameter optimization
      </p>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
        <StatCard
          title="Total Models"
          value={totalModels}
          subtitle="Quantum + classical"
          icon={<Brain className="w-5 h-5 text-purple-500" />}
        />
        <StatCard
          title="Quantum Models"
          value={info?.quantum_models ?? 18}
          subtitle="Variational & kernel"
          icon={<Zap className="w-5 h-5 text-blue-500" />}
        />
        <StatCard
          title="Classical Models"
          value={info?.classical_models ?? 8}
          subtitle="Baselines"
          icon={<BarChart3 className="w-5 h-5 text-orange-500" />}
        />
        <StatCard
          title="UCI Datasets"
          value={datasets.length}
          subtitle="Ready to load"
          icon={<Database className="w-5 h-5 text-green-500" />}
        />
      </div>

      <div className="mb-8">
        <h2 className="text-2xl font-bold mb-4">Get Started</h2>
        <Link
          href="/optimizer"
          className="inline-flex flex-col bg-primary hover:bg-primary/90 text-primary-foreground p-6 rounded-lg transition-colors max-w-md"
        >
          <Zap className="w-8 h-8 mb-2" />
          <h3 className="text-lg font-semibold mb-1">Open the Optimizer</h3>
          <p className="text-sm opacity-90">
            Load data, run hyperparameter optimization, explore SHAP analysis and generate an AI
            report.
          </p>
        </Link>
      </div>

      <div>
        <h2 className="text-2xl font-bold mb-4">Available UCI Datasets</h2>
        {datasets.length === 0 ? (
          <div className="bg-card p-6 rounded-lg border border-border text-center text-muted-foreground">
            Could not reach the backend. Start it and refresh to see available datasets.
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {datasets.map((d) => (
              <div key={d.id} className="bg-card p-4 rounded-lg border border-border">
                <p className="font-semibold">{d.name}</p>
                {d.description && (
                  <p className="text-sm text-muted-foreground mt-1">{d.description}</p>
                )}
                <p className="text-xs text-muted-foreground mt-2">UCI ID: {d.id}</p>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
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
    <div className="bg-card p-6 rounded-lg border border-border">
      <div className="flex items-center justify-between mb-2">
        <h3 className="text-lg font-semibold">{title}</h3>
        {icon}
      </div>
      <p className="text-3xl font-bold">{value}</p>
      <p className="text-sm text-muted-foreground">{subtitle}</p>
    </div>
  );
}
