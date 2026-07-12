'use client';

import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';
import { loginUrl } from '@/lib/api';
import { useUser } from '@/lib/hooks';
import { LogIn } from 'lucide-react';

/**
 * Blocks the app behind a login screen when the backend has Auth0 configured
 * and there is no session. Renders children untouched while the profile is
 * loading (avoids a flash) and when auth is disabled.
 */
export function AuthGate({ children }: { children: React.ReactNode }) {
  const { user, authEnabled, loading } = useUser();

  // Render nothing until the session state is known — avoids flashing the
  // dashboard (and a burst of 401s) before the login screen appears.
  if (loading) {
    return null;
  }

  if (!authEnabled || user) {
    return <>{children}</>;
  }

  return (
    <div className="flex h-svh w-full items-center justify-center p-4">
      <Card className="w-full max-w-sm">
        <CardContent className="flex flex-col items-center gap-6 p-8 text-center">
          <span className="flex size-12 items-center justify-center rounded-xl bg-brand text-2xl font-bold text-brand-foreground">
            Q
          </span>
          <div>
            <h1 className="text-lg font-bold tracking-tight">Welcome to QuOptuna</h1>
            <p className="mt-1 text-sm text-muted-foreground">
              Sign in to run and analyze optimizations.
            </p>
          </div>
          <div className="flex w-full flex-col gap-2">
            <Button asChild>
              <a href={loginUrl()}>
                <LogIn className="size-4" />
                Log in
              </a>
            </Button>
            <Button variant="outline" asChild>
              <a href={loginUrl('signup')}>Sign up</a>
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
