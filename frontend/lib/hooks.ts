'use client';

import { useMutation, useQuery } from '@tanstack/react-query';
import {
  type ReportRequest,
  type SHAPRequest,
  generateReport,
  generateSHAP,
  getDatasetPreview,
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
