'use client';

import { Button } from '@/components/ui/button';
import { Card, CardAction, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from '@/components/ui/dialog';
import { Skeleton } from '@/components/ui/skeleton';
import { cn } from '@/lib/utils';
import { Loader2, Maximize2 } from 'lucide-react';
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
        <div className="flex h-full min-h-[240px] items-center justify-center text-muted-foreground">
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
  className,
}: {
  title: string;
  figure: PlotlyFigureJSON;
  className?: string;
}) {
  const [fullscreen, setFullscreen] = useState(false);

  return (
    <Card className={className}>
      <CardHeader>
        <CardTitle>{title}</CardTitle>
        <CardAction>
          <Dialog open={fullscreen} onOpenChange={setFullscreen}>
            <DialogTrigger asChild>
              <Button type="button" variant="ghost" size="sm" aria-label={`Expand ${title}`}>
                <Maximize2 className="h-4 w-4" />
              </Button>
            </DialogTrigger>
            <DialogContent className="inset-4 top-4 left-4 flex h-auto w-auto max-w-none translate-x-0 translate-y-0 flex-col gap-2 md:inset-10 md:top-10 md:left-10">
              <DialogHeader>
                <DialogTitle>{title}</DialogTitle>
              </DialogHeader>
              {fullscreen && <PlotlyChart figure={figure} className="min-h-0 flex-1" />}
            </DialogContent>
          </Dialog>
        </CardAction>
      </CardHeader>
      <CardContent>
        <PlotlyChart figure={figure} className="aspect-[16/9] min-h-[280px] w-full" />
      </CardContent>
    </Card>
  );
}

export function PlotSkeleton({ title, className }: { title: string; className?: string }) {
  return (
    <Card className={cn(className)}>
      <CardHeader>
        <CardTitle>{title}</CardTitle>
      </CardHeader>
      <CardContent>
        <Skeleton className="aspect-[16/9] min-h-[280px] w-full" />
      </CardContent>
    </Card>
  );
}
