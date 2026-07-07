'use client';

import {
  SidebarContent,
  SidebarFooter,
  SidebarHeader,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
  SidebarRail,
  Sidebar as SidebarRoot,
  useSidebar,
} from '@/components/ui/sidebar';
import { StatusDot } from '@/components/ui/status-dot';
import { ThemeToggle } from '@/components/ui/theme-toggle';
import { useBackendStatus } from '@/lib/hooks';
import { History, LayoutDashboard, Settings, Zap } from 'lucide-react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';

const navigation = [
  { name: 'Dashboard', href: '/', icon: LayoutDashboard },
  { name: 'Optimizer', href: '/optimizer', icon: Zap },
  { name: 'Runs', href: '/runs', icon: History },
  { name: 'Settings', href: '/settings', icon: Settings },
];

export function Sidebar() {
  const pathname = usePathname();
  const { state } = useSidebar();
  const { online, loading } = useBackendStatus();
  const expanded = state === 'expanded';

  // Static export uses trailingSlash: true, so usePathname() yields "/optimizer/".
  // Normalize so active matching works on every route, not just "/".
  const normalize = (p: string) => (p !== '/' && p.endsWith('/') ? p.slice(0, -1) : p);
  const currentPath = normalize(pathname);

  return (
    <SidebarRoot collapsible="icon">
      <SidebarHeader>
        <div className="flex items-center gap-2">
          <span className="flex size-8 shrink-0 items-center justify-center rounded-lg bg-brand text-base font-bold text-brand-foreground">
            Q
          </span>
          {expanded && (
            <div className="min-w-0">
              <h1 className="truncate text-base font-bold leading-tight tracking-tight">
                QuOptuna
              </h1>
              <p className="truncate text-xs text-muted-foreground">Next Generation</p>
            </div>
          )}
        </div>
      </SidebarHeader>

      <SidebarContent>
        <SidebarMenu className="px-2">
          {navigation.map((item) => (
            <SidebarMenuItem key={item.name}>
              <SidebarMenuButton asChild isActive={currentPath === item.href} tooltip={item.name}>
                <Link href={item.href}>
                  <item.icon />
                  <span>{item.name}</span>
                </Link>
              </SidebarMenuButton>
            </SidebarMenuItem>
          ))}
        </SidebarMenu>
      </SidebarContent>

      <SidebarFooter>
        <div className="flex h-8 items-center gap-2 px-2">
          <StatusDot status={loading ? 'idle' : online ? 'online' : 'offline'} />
          {expanded && (
            <span className="truncate text-xs text-muted-foreground">
              {loading ? 'Connecting…' : online ? 'Backend online' : 'Backend offline'}
            </span>
          )}
        </div>
        <ThemeToggle expanded={expanded} />
      </SidebarFooter>

      <SidebarRail />
    </SidebarRoot>
  );
}
