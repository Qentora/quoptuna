'use client';

import dynamic from 'next/dynamic';

// The wizard is fully client-rendered (charts/canvas, file inputs, polling).
const Wizard = dynamic(() => import('@/components/optimizer/Wizard').then((m) => m.Wizard), {
  ssr: false,
});

export default function OptimizerPage() {
  return <Wizard />;
}
