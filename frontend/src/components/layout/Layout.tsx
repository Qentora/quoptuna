import { Link, Outlet, useLocation } from 'react-router-dom';
import { LayoutDashboard, Workflow, Database, BarChart3, Settings, Brain, Zap } from 'lucide-react';
import { FEATURES } from '../../config/features';

const allNavigation = [
  { name: 'Dashboard', href: '/', icon: LayoutDashboard, feature: null },
  { name: 'Optimizer', href: '/optimizer', icon: Zap, feature: 'STEP_OPTIMIZER' as const },
  { name: 'Workflow Builder', href: '/workflow', icon: Workflow, feature: 'WORKFLOW_BUILDER' as const },
  { name: 'Data Explorer', href: '/data', icon: Database, feature: 'DATA_EXPLORER' as const },
  { name: 'Models', href: '/models', icon: Brain, feature: 'MODELS' as const },
  { name: 'Analytics', href: '/analytics', icon: BarChart3, feature: 'ANALYTICS' as const },
  { name: 'Settings', href: '/settings', icon: Settings, feature: null },
];

// Filter navigation based on feature flags
const navigation = allNavigation.filter(item => !item.feature || FEATURES[item.feature]);

export function Layout() {
  const location = useLocation();

  return (
    <div className="min-h-screen flex">
      {/* Sidebar */}
      <div className="w-64 bg-card border-r border-border">
        <div className="p-6">
          <h1 className="text-2xl font-bold text-primary">QuOptuna</h1>
          <p className="text-xs text-muted-foreground">Next Generation</p>
        </div>

        <nav className="px-3">
          {navigation.map((item) => {
            const isActive = location.pathname === item.href;
            const Icon = item.icon;

            return (
              <Link
                key={item.name}
                to={item.href}
                className={`flex items-center gap-3 px-3 py-2 mb-1 rounded-md transition-colors ${
                  isActive
                    ? 'bg-primary text-primary-foreground'
                    : 'text-muted-foreground hover:bg-accent hover:text-accent-foreground'
                }`}
              >
                <Icon className="w-5 h-5" />
                <span className="font-medium">{item.name}</span>
              </Link>
            );
          })}
        </nav>
      </div>

      {/* Main Content */}
      <div className="flex-1 overflow-auto">
        <Outlet />
      </div>
    </div>
  );
}
