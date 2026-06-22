import { cn } from '@/lib/utils';
import { type VariantProps, cva } from 'class-variance-authority';
import type * as React from 'react';

const badgeVariants = cva(
  'inline-flex items-center gap-1.5 rounded-full font-semibold whitespace-nowrap',
  {
    variants: {
      variant: {
        default: 'bg-badge text-badge-foreground',
        secondary: 'bg-muted text-secondary-foreground',
        outline: 'border border-border text-foreground',
        emerald: 'bg-accent-emerald text-accent-emerald-foreground',
        amber: 'bg-accent-amber text-accent-amber-foreground',
        destructive: 'bg-accent-red text-accent-red-foreground',
        quantum: 'bg-accent-purple text-accent-purple-foreground',
        classical: 'bg-accent-orange text-accent-orange-foreground',
      },
      size: {
        sm: 'h-5 px-2 text-xs',
        md: 'h-6 px-2.5 text-sm',
      },
    },
    defaultVariants: {
      variant: 'default',
      size: 'sm',
    },
  }
);

const glowByVariant: Record<string, string> = {
  emerald: 'shadow-glow-emerald',
  amber: 'shadow-glow-amber',
  quantum: 'shadow-glow-quantum',
  classical: 'shadow-glow-classical',
};

export interface BadgeProps
  extends React.HTMLAttributes<HTMLSpanElement>,
    VariantProps<typeof badgeVariants> {
  /** Apply a subtle matching glow shadow for the variant. */
  glow?: boolean;
}

export function Badge({ className, variant, size, glow = false, ...props }: BadgeProps) {
  return (
    <span
      className={cn(
        badgeVariants({ variant, size }),
        glow && glowByVariant[variant ?? 'default'],
        className
      )}
      {...props}
    />
  );
}

export { badgeVariants };
