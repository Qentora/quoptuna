'use client';

import { useMutation, useQuery } from '@tanstack/react-query';
import {
  type ReportRequest,
  type SHAPRequest,
  generateReport,
  generateSHAP,
  getAuthProfile,
  getDatasetPreview,
  getHealth,
  getMetrics,
  getSystemInfo,
  listUCIDatasets,
  loadUCIDataset,
  startOptimization,
  uploadDataset,
} from './api';

export function useSystemInfo() {
  return useQuery({ queryKey: ['system-info'], queryFn: getSystemInfo });
}

/**
 * Lightweight backend reachability for status dots. Swallows errors so an
 * offline backend resolves to `online: false` rather than throwing.
 */
export function useBackendStatus() {
  // Uses the unauthenticated health endpoint so the dot reflects reachability
  // even when the user is logged out.
  const { data, isLoading, isFetched } = useQuery({
    queryKey: ['backend-health'],
    queryFn: () => getHealth().catch(() => null),
  });
  return {
    online: Boolean(data),
    loading: isLoading || !isFetched,
  } as const;
}

/**
 * Current Auth0 user. `user` is null when logged out; `authEnabled` is false
 * when the backend has no Auth0 configuration (auth UI hides itself).
 */
export function useUser() {
  const { data, isLoading } = useQuery({
    queryKey: ['auth-profile'],
    queryFn: () => getAuthProfile().catch(() => null),
  });
  return {
    user: data?.user ?? null,
    authEnabled: data?.auth_enabled ?? false,
    loading: isLoading,
  } as const;
}

export function useUCIDatasets() {
  return useQuery({ queryKey: ['uci-datasets'], queryFn: listUCIDatasets });
}

export function useDatasetPreview(datasetId: string | null) {
  return useQuery({
    queryKey: ['dataset-preview', datasetId],
    queryFn: () => getDatasetPreview(datasetId as string),
    enabled: Boolean(datasetId),
  });
}

export function useUploadDataset() {
  return useMutation({ mutationFn: (file: File) => uploadDataset(file) });
}

export function useLoadUCIDataset() {
  return useMutation({ mutationFn: (id: number) => loadUCIDataset(id) });
}

export function useStartOptimization() {
  return useMutation({ mutationFn: startOptimization });
}

export function useGenerateSHAP() {
  return useMutation({ mutationFn: (body: SHAPRequest) => generateSHAP(body) });
}

export function useMetrics() {
  return useMutation({
    mutationFn: ({
      optimizationId,
      trialNumber,
    }: { optimizationId: string; trialNumber?: number }) => getMetrics(optimizationId, trialNumber),
  });
}

export function useGenerateReport() {
  return useMutation({ mutationFn: (body: ReportRequest) => generateReport(body) });
}
