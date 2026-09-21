'use client';

import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import type { AnalysisRevisionSummary } from '@/lib/api';
import { Eye } from 'lucide-react';

/** ISO timestamp -> locale string, falling back to the raw value. */
export function formatTimestamp(value: string): string {
  const parsed = new Date(value);
  return Number.isNaN(parsed.getTime()) ? value : parsed.toLocaleString();
}

/**
 * Marks which analysis revision a result was grounded in.
 *
 * This is the point of the history: two reports about the same trial can
 * legitimately disagree because they read different evidence — for example one
 * written before the fairness audit was computed and one after.
 */
export function RevisionBadge({ revision, current }: { revision: number; current: number | null }) {
  const isCurrent = revision === current;
  return (
    <Badge
      variant={isCurrent ? 'secondary' : 'outline'}
      title={
        isCurrent
          ? 'Grounded in the analysis revision currently loaded'
          : 'Grounded in an earlier analysis revision, so its evidence differed'
      }
    >
      rev {revision}
    </Badge>
  );
}

/**
 * Every completed analysis of this snapshot, newest first.
 *
 * Re-running an analysis with unchanged settings is deterministic, so the
 * numbers usually come back identical — the timestamp and revision here are
 * what distinguish a fresh run from the one already on screen.
 *
 * Figures for older revisions are pruned past `ANALYSIS_HISTORY_LIMIT`; those
 * rows survive as metadata and cannot be reopened, which the table says.
 */
export function AnalysisHistoryTable({
  revisions,
  currentRevision,
  viewingRevision,
  busyRevision,
  onView,
}: {
  revisions: AnalysisRevisionSummary[];
  currentRevision: number | null;
  viewingRevision: number | null;
  busyRevision: number | null;
  onView: (revision: number) => void;
}) {
  const latest = revisions[0]?.revision ?? null;
  return (
    <div className="space-y-3">
      {revisions.length > 1 && (
        <p className="text-sm text-muted-foreground">
          This trial has been analysed {revisions.length} times. Identical numbers across revisions
          are expected — the same trial re-analysed at the same settings is deterministic.
        </p>
      )}
      <div className="overflow-x-auto rounded-md border border-border">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Completed</TableHead>
              <TableHead>Trial</TableHead>
              <TableHead>Model</TableHead>
              <TableHead>Evidence</TableHead>
              <TableHead>Status</TableHead>
              <TableHead className="text-right">Actions</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {revisions.map((item) => (
              <TableRow
                key={item.id}
                className={item.revision === viewingRevision ? 'bg-muted/50' : ''}
              >
                <TableCell className="whitespace-nowrap font-mono text-xs">
                  {formatTimestamp(item.created_at)}
                </TableCell>
                <TableCell className="whitespace-nowrap tabular-nums">
                  {item.analysed_trial ?? '—'}
                </TableCell>
                <TableCell className="whitespace-nowrap font-mono text-xs">
                  {item.analysed_model_type ?? '—'}
                </TableCell>
                <TableCell>
                  <RevisionBadge revision={item.revision} current={currentRevision} />
                </TableCell>
                <TableCell>
                  {item.revision === latest ? (
                    <Badge variant="emerald">latest</Badge>
                  ) : item.artifacts_pruned ? (
                    <span
                      className="text-muted-foreground text-xs"
                      title="Figures for this revision were pruned; only its metadata is kept"
                    >
                      figures pruned
                    </span>
                  ) : (
                    <span className="text-muted-foreground text-xs">superseded</span>
                  )}
                </TableCell>
                <TableCell className="text-right whitespace-nowrap">
                  {item.artifacts_pruned ? (
                    <span className="text-muted-foreground text-xs">—</span>
                  ) : (
                    <Button
                      type="button"
                      variant="ghost"
                      size="sm"
                      disabled={busyRevision !== null || item.revision === viewingRevision}
                      onClick={() => onView(item.revision)}
                    >
                      <Eye className="h-4 w-4" />
                      {item.revision === viewingRevision ? 'Viewing' : 'View'}
                    </Button>
                  )}
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </div>
    </div>
  );
}
