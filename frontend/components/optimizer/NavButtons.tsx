'use client';

import { Alert } from '@/components/ui/alert';

export function ErrorBanner({ message }: { message: string | null }) {
  if (!message) return null;
  return <Alert variant="destructive">{message}</Alert>;
}
