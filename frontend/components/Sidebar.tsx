'use client';

import { StatusDot } from '@/components/ui/status-dot';
import { ThemeToggle } from '@/components/ui/theme-toggle';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip';
import { useBackendStatus } from '@/lib/hooks';
import { cn } from '@/lib/utils';
import { History, LayoutDashboard, Settings, Zap } from 'lucide-react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { useState } from 'react';

const navigation = [
  { name: 'Dashboard', href: '/', icon: LayoutDashboard },
  { name: 'Optimizer', href: '/optimizer', icon: Zap },
  { name: 'Runs', href: '/runs', icon: History },
  { name: 'Settings', href: '/settings', icon: Settings },
];

export function Sidebar() {
  const pathname = usePathname();
  const [expanded, setExpanded] = useState(false);
  const { online, loading } = useBackendStatus();

  // Static export uses trailingSlash: true, so usePathname() yields "/optimizer/".
  // Normalize so active matching works on every route, not just "/".
  const normalize = (p: string) => (p !== '/' && p.endsWith('/') ? p.slice(0, -1) : p);
  const currentPath = normalize(pathname);

  return (
    <TooltipProvider delayDuration={200}>
      <div
        onMouseEnter={() => setExpanded(true)}
        onMouseLeave={() => setExpanded(false)}
        onFocusCapture={() => setExpanded(true)}
        onBlurCapture={() => setExpanded(false)}
        className={cn(
          'flex shrink-0 flex-col overflow-hidden border-r border-border bg-card transition-[width] duration-200 ease-in-out',
          expanded ? 'w-64' : 'w-14'
        )}
      >
        <div className="flex items-center gap-2 px-2 py-2">
          <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-brand text-base font-bold text-brand-foreground">
            Q
          </span>
          <div
            className={cn(
              'min-w-0 transition-opacity duration-200',
              expanded ? 'opacity-100' : 'opacity-0'
            )}
          >
            <h1 className="whitespace-nowrap text-lg font-bold leading-tight tracking-tight">
              QuOptuna
            </h1>
            <p className="whitespace-nowrap text-xs text-muted-foreground">Next Generation</p>
          </div>
        </div>

        <nav className="mt-2 flex-1 px-2">
          {navigation.map((item) => {
            const isActive = currentPath === item.href;
            const Icon = item.icon;
            return (
              <Tooltip key={item.name}>
                <TooltipTrigger asChild>
                  <Link
                    href={item.href}
                    className={cn(
                      'mb-1 flex items-center gap-2 rounded-lg text-sm font-medium transition-colors',
                      isActive ? 'text-brand' : 'text-muted-foreground hover:text-foreground'
                    )}
                  >
                    <span
                      className={cn(
                        'flex h-9 w-9 shrink-0 items-center justify-center rounded-lg transition-colors',
                        isActive ? 'bg-brand/15' : 'group-hover:bg-accent/60 hover:bg-accent/60'
                      )}
                    >
                      <Icon size={18} className="shrink-0" />
                    </span>
                    <span
                      className={cn(
                        'whitespace-nowrap transition-opacity duration-200',
                        expanded ? 'opacity-100' : 'opacity-0'
                      )}
                    >
                      {item.name}
                    </span>
                  </Link>
                </TooltipTrigger>
                <TooltipContent side="right">{item.name}</TooltipContent>
              </Tooltip>
            );
          })}
        </nav>

        <div className="flex flex-col gap-1 p-2">
          <div className="flex h-9 items-center gap-2">
            <span className="flex h-9 w-9 shrink-0 items-center justify-center">
              <StatusDot status={loading ? 'idle' : online ? 'online' : 'offline'} />
            </span>
            <span
              className={cn(
                'whitespace-nowrap text-xs text-muted-foreground transition-opacity duration-200',
                expanded ? 'opacity-100' : 'opacity-0'
              )}
            >
              {loading ? 'Connecting…' : online ? 'Backend online' : 'Backend offline'}
            </span>
          </div>
          <ThemeToggle expanded={expanded} />
        </div>
      </div>
    </TooltipProvider>
  );
}
