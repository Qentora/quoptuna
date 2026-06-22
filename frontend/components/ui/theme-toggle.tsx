'use client';

import { cn } from '@/lib/utils';
import { Monitor, Moon, Sun } from 'lucide-react';
import { useTheme } from 'next-themes';
import { useEffect, useState } from 'react';

const OPTIONS = [
  { value: 'light', icon: Sun, label: 'Light' },
  { value: 'dark', icon: Moon, label: 'Dark' },
  { value: 'system', icon: Monitor, label: 'System' },
] as const;

export function ThemeToggle({ expanded }: { expanded: boolean }) {
  const { theme, setTheme } = useTheme();
  const [mounted, setMounted] = useState(false);

  useEffect(() => setMounted(true), []);

  if (!mounted) {
    return <div className="h-9" />;
  }

  if (!expanded) {
    const order = ['light', 'dark', 'system'] as const;
    const current = (theme as (typeof order)[number]) ?? 'system';
    const next = order[(order.indexOf(current) + 1) % order.length];
    const Active = OPTIONS.find((o) => o.value === current)?.icon ?? Monitor;
    return (
      <button
        type="button"
        onClick={() => setTheme(next)}
        aria-label="Toggle theme"
        className="flex h-9 w-9 items-center justify-center rounded-md text-muted-foreground transition-colors hover:bg-accent hover:text-accent-foreground"
      >
        <Active size={18} />
      </button>
    );
  }

  return (
    <div className="flex items-center gap-1 rounded-md border border-border p-1">
      {OPTIONS.map((opt) => {
        const Icon = opt.icon;
        const active = (theme ?? 'system') === opt.value;
        return (
          <button
            key={opt.value}
            type="button"
            onClick={() => setTheme(opt.value)}
            aria-label={opt.label}
            className={cn(
              'flex h-7 flex-1 items-center justify-center rounded transition-colors',
              active
                ? 'bg-accent text-accent-foreground'
                : 'text-muted-foreground hover:text-foreground'
            )}
          >
            <Icon className="h-4 w-4" />
          </button>
        );
      })}
    </div>
  );
}
