import { cn } from '@/lib/utils';
import type * as React from 'react';
import { Label } from './label';

export interface FieldProps extends React.HTMLAttributes<HTMLDivElement> {
  label?: React.ReactNode;
  htmlFor?: string;
  helper?: React.ReactNode;
  error?: React.ReactNode;
}

export function Field({
  label,
  htmlFor,
  helper,
  error,
  className,
  children,
  ...props
}: FieldProps) {
  return (
    <div className={cn('space-y-2', className)} {...props}>
      {label && <Label htmlFor={htmlFor}>{label}</Label>}
      {children}
      {error ? (
        <p className="text-xs text-destructive">{error}</p>
      ) : helper ? (
        <p className="text-xs text-muted-foreground">{helper}</p>
      ) : null}
    </div>
  );
}
