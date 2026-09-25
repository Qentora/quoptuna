import type { TargetBalanceProfile } from './api';

export function targetBalancePresentation(profile: TargetBalanceProfile) {
  switch (profile.label) {
    case 'balanced':
      return { label: 'Balanced', variant: 'emerald' as const };
    case 'moderate_imbalance':
      return { label: 'Moderate imbalance', variant: 'amber' as const };
    case 'imbalanced':
      return { label: 'Imbalanced', variant: 'destructive' as const };
    case 'multiclass':
      return { label: `Multiclass · ${profile.classes.length} classes`, variant: 'secondary' as const };
  }
}

export function targetBalancePercentages(profile: TargetBalanceProfile): string {
  const total = profile.classes.reduce((sum, item) => sum + item.count, 0);
  return profile.classes.map(({ count }) => Math.round((count / total) * 100)).join(' / ');
}

export function targetBalanceHelp(profile: TargetBalanceProfile): string {
  const { label } = targetBalancePresentation(profile);
  return `${label}: ${targetBalancePercentages(profile)}. Based on the selected target's observed class distribution; this is not a fairness score.`;
}
