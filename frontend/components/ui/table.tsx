import { cn } from '@/lib/utils';
import * as React from 'react';

const StickyHeaderContext = React.createContext(false);

export function TableContainer({ className, ...props }: React.HTMLAttributes<HTMLDivElement>) {
  return (
    <div className={cn('overflow-hidden rounded-lg border border-border', className)} {...props} />
  );
}

interface TableProps extends React.HTMLAttributes<HTMLTableElement> {
  /**
   * When set, the table uses border-separate so each `<th>` can carry a sticky,
   * opaque background that stays pinned (and visible) while the body scrolls.
   * Wrap the table in a scroll container with `overflow-y-auto`.
   */
  stickyHeader?: boolean;
}

export const Table = React.forwardRef<HTMLTableElement, TableProps>(
  ({ className, stickyHeader = false, ...props }, ref) => (
    <StickyHeaderContext.Provider value={stickyHeader}>
      <table
        ref={ref}
        className={cn(
          'w-full text-sm',
          stickyHeader && 'border-separate border-spacing-0',
          className
        )}
        {...props}
      />
    </StickyHeaderContext.Provider>
  )
);
Table.displayName = 'Table';

export function TableHead({ className, ...props }: React.HTMLAttributes<HTMLTableSectionElement>) {
  const stickyHeader = React.useContext(StickyHeaderContext);
  return (
    <thead
      className={cn(
        'text-muted-foreground',
        // With sticky headers the border + bg live on the <th> cells instead.
        !stickyHeader && 'border-b border-border bg-muted',
        className
      )}
      {...props}
    />
  );
}

export function TableBody({ className, ...props }: React.HTMLAttributes<HTMLTableSectionElement>) {
  return <tbody className={cn('divide-y divide-border', className)} {...props} />;
}

export function Th({ className, ...props }: React.ThHTMLAttributes<HTMLTableCellElement>) {
  const stickyHeader = React.useContext(StickyHeaderContext);
  return (
    <th
      className={cn(
        'px-3 py-2 text-left font-medium',
        stickyHeader && 'sticky top-0 z-10 border-b border-border bg-muted',
        className
      )}
      {...props}
    />
  );
}

export function Td({ className, ...props }: React.TdHTMLAttributes<HTMLTableCellElement>) {
  return <td className={cn('px-3 py-2 text-foreground', className)} {...props} />;
}
