import { cn } from '@/lib/utils';
import { type VariantProps, cva } from 'class-variance-authority';
import { AlertTriangle, CheckCircle2, Info, type LucideIcon, XCircle } from 'lucide-react';
import type * as React from 'react';

const alertVariants = cva('flex items-start gap-3 rounded-lg border p-4 text-sm', {
  variants: {
    variant: {
      default: 'border-border bg-card text-foreground',
      info: 'border-accent-indigo bg-accent-indigo/40 text-accent-indigo-foreground',
      success: 'border-accent-emerald bg-accent-emerald/40 text-accent-emerald-foreground',
      warning: 'border-accent-amber bg-accent-amber/40 text-accent-amber-foreground',
      destructive: 'border-accent-red bg-accent-red/40 text-accent-red-foreground',
    },
  },
  defaultVariants: { variant: 'default' },
});

const ICONS: Record<string, LucideIcon> = {
  info: Info,
  success: CheckCircle2,
  warning: AlertTriangle,
  destructive: XCircle,
};

export interface AlertProps
  extends React.HTMLAttributes<HTMLDivElement>,
    VariantProps<typeof alertVariants> {
  icon?: boolean;
}

export function Alert({ className, variant, icon = true, children, ...props }: AlertProps) {
  const Icon = variant && variant !== 'default' ? ICONS[variant] : undefined;
  return (
    <div className={cn(alertVariants({ variant }), className)} {...props}>
      {icon && Icon && <Icon className="mt-0.5 h-4 w-4 shrink-0" />}
      <div className="min-w-0 flex-1">{children}</div>
    </div>
  );
}
