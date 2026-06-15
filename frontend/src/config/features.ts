/**
 * Feature flags configuration
 *
 * Enable or disable features across the application
 */

export const FEATURES = {
  // Workflow builder (node-based visual workflow)
  WORKFLOW_BUILDER: false,

  // Step-based optimizer (streamlit-like wizard)
  STEP_OPTIMIZER: true,

  // Data explorer page
  DATA_EXPLORER: true,

  // Models page
  MODELS: true,

  // Analytics page
  ANALYTICS: true,
} as const;

export type FeatureName = keyof typeof FEATURES;

/**
 * Check if a feature is enabled
 */
export function isFeatureEnabled(feature: FeatureName): boolean {
  return FEATURES[feature];
}
