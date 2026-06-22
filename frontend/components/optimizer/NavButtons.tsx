'use client';

import { Alert } from '@/components/ui/alert';
import { Button } from '@/components/ui/button';

export function NavButtons({
  onBack,
  onNext,
  nextLabel = 'Next Step',
  nextDisabled = false,
  backDisabled = false,
  hideNext = false,
}: {
  onBack?: () => void;
  onNext?: () => void;
  nextLabel?: string;
  nextDisabled?: boolean;
  backDisabled?: boolean;
  hideNext?: boolean;
}) {
  return (
    <div className="flex justify-between pt-4">
      {onBack ? (
        <Button type="button" variant="outline" onClick={onBack} disabled={backDisabled}>
          Previous
        </Button>
      ) : (
        <span />
      )}
      {!hideNext && (
        <Button type="button" onClick={onNext} disabled={nextDisabled}>
          {nextLabel}
        </Button>
      )}
    </div>
  );
}

export function ErrorBanner({ message }: { message: string | null }) {
  if (!message) return null;
  return <Alert variant="destructive">{message}</Alert>;
}
