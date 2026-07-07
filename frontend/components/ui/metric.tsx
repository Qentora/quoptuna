import { cn } from '@/lib/utils';
import type * as React from 'react';

type MetricTone = 'default' | 'emerald' | 'brand' | 'amber' | 'orange';

const toneClasses: Record<MetricTone, string> = {
  default: 'text-foreground',
  emerald: 'text-accent-emerald-foreground',
  brand: 'text-brand',
  amber: 'text-accent-amber-foreground',
  orange: 'text-accent-orange-foreground',
};

export interface MetricProps extends React.HTMLAttributes<HTMLSpanElement> {
  /** The large number / value to display. */
  value: React.ReactNode;
  tone?: MetricTone;
  /** Apply a subtle text glow in the tone's color. */
  glow?: boolean;
}

export function Metric({
  value,
  tone = 'default',
  glow = false,
  className,
  ...props
}: MetricProps) {
  return (
    <span
      className={cn(
        'font-semibold tabular-nums tracking-tight',
        toneClasses[tone],
        glow && tone !== 'default' && 'text-glow',
        className
      )}
      {...props}
    >
      {value}
    </span>
  );
}
