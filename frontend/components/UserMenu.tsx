'use client';

import { Button } from '@/components/ui/button';
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip';
import { loginUrl, logoutUrl } from '@/lib/api';
import { useUser } from '@/lib/hooks';
import { LogIn, LogOut } from 'lucide-react';

/**
 * Sidebar-footer auth control. Hidden entirely when the backend has no Auth0
 * configuration; otherwise shows a login button or the signed-in user with a
 * logout action. Login/logout are full-page navigations driven by the backend.
 */
export function UserMenu({ expanded }: { expanded: boolean }) {
  const { user, authEnabled, loading } = useUser();

  if (loading || !authEnabled) return null;

  if (!user) {
    return (
      <Button
        variant="ghost"
        size="sm"
        className="h-8 justify-start gap-2 px-2 text-muted-foreground"
        asChild
      >
        <a href={loginUrl()}>
          <LogIn className="size-4 shrink-0" />
          {expanded && <span className="truncate text-xs">Log in</span>}
        </a>
      </Button>
    );
  }

  const label = user.name || user.nickname || user.email || 'Account';
  const initial = (label[0] || '?').toUpperCase();

  return (
    <div className="flex h-8 items-center gap-2 px-2">
      {user.picture ? (
        // eslint-disable-next-line @next/next/no-img-element
        <img
          src={user.picture}
          alt=""
          className="size-5 shrink-0 rounded-full"
          referrerPolicy="no-referrer"
        />
      ) : (
        <span className="flex size-5 shrink-0 items-center justify-center rounded-full bg-brand text-[10px] font-bold text-brand-foreground">
          {initial}
        </span>
      )}
      {expanded && (
        <span
          className="min-w-0 flex-1 truncate text-xs text-muted-foreground"
          title={String(user.email ?? label)}
        >
          {label}
        </span>
      )}
      <Tooltip>
        <TooltipTrigger asChild>
          <Button
            variant="ghost"
            size="icon"
            className="size-6 shrink-0 text-muted-foreground"
            asChild
          >
            <a href={logoutUrl()} aria-label="Log out">
              <LogOut className="size-3.5" />
            </a>
          </Button>
        </TooltipTrigger>
        <TooltipContent side="right">Log out</TooltipContent>
      </Tooltip>
    </div>
  );
}
