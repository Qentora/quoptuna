import { cn } from '@/lib/utils';
import * as React from 'react';

export const Input = React.forwardRef<
  HTMLInputElement,
  React.InputHTMLAttributes<HTMLInputElement>
>(({ className, ...props }, ref) => (
  <input ref={ref} className={cn('primary-input', className)} {...props} />
));
Input.displayName = 'Input';
