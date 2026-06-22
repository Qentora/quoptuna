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
    <div className="sticky bottom-0 -mx-8 -mb-8 mt-2 flex justify-between border-t border-border bg-card/95 px-8 py-4 backdrop-blur supports-[backdrop-filter]:bg-card/80">
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
