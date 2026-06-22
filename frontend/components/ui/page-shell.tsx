import { PageHeader } from '@/components/ui/page-header';
import { cn } from '@/lib/utils';
import type * as React from 'react';

export interface PageShellProps {
  title: string;
  children: React.ReactNode;
  /** Optional max-width / centering for the content region (e.g. "mx-auto max-w-6xl"). */
  contentClassName?: string;
}

/**
 * Consistent page chrome: a full-width header band (title flush-left at the same x on every page)
 * above a content region. The header sits outside `contentClassName`, so per-page content widths
 * never shift the title. The app layout (`app/layout.tsx`) owns the scroll container.
 */
export function PageShell({ title, children, contentClassName }: PageShellProps) {
  return (
    <div className="flex min-h-full flex-col">
      <div className="border-b border-border bg-card p-6">
        <PageHeader title={title} />
      </div>
      <div className="flex-1 p-6">
        <div className={cn(contentClassName)}>{children}</div>
      </div>
    </div>
  );
}
