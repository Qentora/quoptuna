import { cn } from '@/lib/utils';
import * as React from 'react';

export const Select = React.forwardRef<
  HTMLSelectElement,
  React.SelectHTMLAttributes<HTMLSelectElement>
>(({ className, ...props }, ref) => (
  <select
    ref={ref}
    className={cn(
      'block h-10 w-full rounded-md border border-input bg-background px-3 text-sm transition-colors hover:border-muted-foreground focus:border-foreground focus:outline-none disabled:pointer-events-none disabled:cursor-not-allowed disabled:bg-muted disabled:text-muted-foreground',
      className
    )}
    {...props}
  />
));
Select.displayName = 'Select';
