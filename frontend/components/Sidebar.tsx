'use client';

import { ThemeToggle } from '@/components/ui/theme-toggle';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip';
import { cn } from '@/lib/utils';
import { LayoutDashboard, Settings, Zap } from 'lucide-react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { useState } from 'react';

const navigation = [
  { name: 'Dashboard', href: '/', icon: LayoutDashboard },
  { name: 'Optimizer', href: '/optimizer', icon: Zap },
  { name: 'Settings', href: '/settings', icon: Settings },
];

export function Sidebar() {
  const pathname = usePathname();
  const [expanded, setExpanded] = useState(false);

  return (
    <TooltipProvider delayDuration={200}>
      <div
        onMouseEnter={() => setExpanded(true)}
        onMouseLeave={() => setExpanded(false)}
        onFocusCapture={() => setExpanded(true)}
        onBlurCapture={() => setExpanded(false)}
        className={cn(
          'flex shrink-0 flex-col overflow-hidden border-r border-border bg-card transition-[width] duration-200 ease-in-out',
          expanded ? 'w-64' : 'w-16'
        )}
      >
        <div className="flex items-center gap-3 p-3">
          <span className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-primary text-xl font-bold text-primary-foreground">
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
            const isActive = pathname === item.href;
            const Icon = item.icon;
            return (
              <Tooltip key={item.name}>
                <TooltipTrigger asChild>
                  <Link
                    href={item.href}
                    className={cn(
                      'mb-1 flex h-9 items-center gap-3 rounded-md px-3 text-sm font-medium transition-colors',
                      isActive
                        ? 'bg-accent text-accent-foreground'
                        : 'text-muted-foreground hover:bg-accent hover:text-accent-foreground'
                    )}
                  >
                    <Icon className="h-5 w-5 shrink-0" />
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

        <div className="p-2">
          <ThemeToggle expanded={expanded} />
        </div>
      </div>
    </TooltipProvider>
  );
}
