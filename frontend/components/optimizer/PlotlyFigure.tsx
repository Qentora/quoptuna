'use client';

import { Button } from '@/components/ui/button';
import * as Dialog from '@radix-ui/react-dialog';
import { Loader2, Maximize2, X } from 'lucide-react';
import { useEffect, useRef, useState } from 'react';
import type { PlotlyFigureJSON } from './types';

// plotly.js is heavy (~1MB gz); load it once, on demand, in the browser only.
let plotlyPromise: Promise<any> | null = null;
function loadPlotly(): Promise<any> {
  if (!plotlyPromise) {
    plotlyPromise = import('plotly.js-dist-min').then((mod) => mod.default ?? mod);
  }
  return plotlyPromise;
}

function PlotlyChart({ figure, className }: { figure: PlotlyFigureJSON; className?: string }) {
  const ref = useRef<HTMLDivElement>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    const el = ref.current;
    if (!el) return;

    loadPlotly().then((Plotly) => {
      if (cancelled || !ref.current) return;
      const layout = {
        ...figure.layout,
        autosize: true,
        margin: { t: 48, r: 24, b: 48, l: 56, ...(figure.layout?.margin ?? {}) },
        paper_bgcolor: 'rgba(0,0,0,0)',
      };
      Plotly.newPlot(ref.current, figure.data, layout, {
        responsive: true,
        displaylogo: false,
      }).then(() => setLoading(false));
    });

    return () => {
      cancelled = true;
      if (el) {
        loadPlotly().then((Plotly) => Plotly.purge(el));
      }
    };
  }, [figure]);

  return (
    <div className={className}>
      {loading && (
        <div className="flex h-full min-h-[200px] items-center justify-center text-muted-foreground">
          <Loader2 className="h-5 w-5 animate-spin" />
        </div>
      )}
      <div ref={ref} className={loading ? 'h-0 overflow-hidden' : 'h-full w-full'} />
    </div>
  );
}

export function PlotlyFigure({
  title,
  figure,
}: {
  title: string;
  figure: PlotlyFigureJSON;
}) {
  const [fullscreen, setFullscreen] = useState(false);

  return (
    <div className="flex flex-col rounded-lg border border-border bg-card">
      <div className="flex items-center justify-between gap-2 border-b border-border bg-muted px-4 py-3">
        <h4 className="text-sm font-semibold">{title}</h4>
        <Button
          type="button"
          variant="ghost"
          size="sm"
          onClick={() => setFullscreen(true)}
          aria-label={`Expand ${title}`}
        >
          <Maximize2 className="h-4 w-4" />
        </Button>
      </div>
      <PlotlyChart figure={figure} className="h-[360px] p-2" />

      <Dialog.Root open={fullscreen} onOpenChange={setFullscreen}>
        <Dialog.Portal>
          <Dialog.Overlay className="fixed inset-0 z-50 bg-black/60" />
          <Dialog.Content className="fixed inset-4 z-50 flex flex-col rounded-lg border border-border bg-card shadow-xl md:inset-10">
            <div className="flex items-center justify-between border-b border-border bg-muted px-4 py-3">
              <Dialog.Title className="text-sm font-semibold">{title}</Dialog.Title>
              <Dialog.Close asChild>
                <Button type="button" variant="ghost" size="sm" aria-label="Close">
                  <X className="h-4 w-4" />
                </Button>
              </Dialog.Close>
            </div>
            {fullscreen && <PlotlyChart figure={figure} className="min-h-0 flex-1 p-2" />}
          </Dialog.Content>
        </Dialog.Portal>
      </Dialog.Root>
    </div>
  );
}

export function PlotSkeleton({ title }: { title: string }) {
  return (
    <div className="flex flex-col rounded-lg border border-border bg-card">
      <div className="border-b border-border bg-muted px-4 py-3">
        <h4 className="text-sm font-semibold">{title}</h4>
      </div>
      <div className="h-[360px] animate-pulse bg-muted/40 p-2" />
    </div>
  );
}
