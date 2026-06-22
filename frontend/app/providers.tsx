'use client';

import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { ThemeProvider } from 'next-themes';
import { useState } from 'react';
import { Toaster } from 'react-hot-toast';

export function Providers({ children }: { children: React.ReactNode }) {
  const [queryClient] = useState(
    () =>
      new QueryClient({
        defaultOptions: {
          queries: {
            staleTime: 30_000,
            refetchOnWindowFocus: false,
          },
        },
      })
  );

  return (
    <ThemeProvider attribute="class" defaultTheme="system" enableSystem disableTransitionOnChange>
      <QueryClientProvider client={queryClient}>
        {children}
        <Toaster
          position="top-right"
          toastOptions={{
            duration: 4000,
            style: {
              background: 'hsl(var(--popover))',
              color: 'hsl(var(--popover-foreground))',
              border: '1px solid hsl(var(--border))',
            },
            success: {
              duration: 3000,
              iconTheme: {
                primary: 'hsl(var(--accent-emerald-foreground))',
                secondary: 'hsl(var(--popover))',
              },
            },
            error: {
              duration: 5000,
              iconTheme: { primary: 'hsl(var(--destructive))', secondary: 'hsl(var(--popover))' },
            },
          }}
        />
      </QueryClientProvider>
    </ThemeProvider>
  );
}
