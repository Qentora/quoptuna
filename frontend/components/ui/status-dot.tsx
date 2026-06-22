import { cn } from '@/lib/utils';
import type * as React from 'react';

export type StatusDotState = 'online' | 'offline' | 'busy' | 'idle';

export interface StatusDotProps extends React.HTMLAttributes<HTMLSpanElement> {
  status: StatusDotState;
  /** Optional text rendered next to the dot. */
  label?: string;
}

export function StatusDot({ status, label, className, ...props }: StatusDotProps) {
  const dot = <span className={cn('status-dot', `status-dot--${status}`)} aria-hidden />;

  if (!label) {
    return (
      <span className={cn('inline-flex', className)} {...props}>
        {dot}
      </span>
    );
  }

  return (
    <span
      className={cn('inline-flex items-center gap-2 text-sm text-muted-foreground', className)}
      {...props}
    >
      {dot}
      <span>{label}</span>
    </span>
  );
}
