'use client';

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
        <button
          type="button"
          onClick={onBack}
          disabled={backDisabled}
          className="px-6 py-2 border border-gray-300 text-gray-700 rounded-md hover:bg-gray-50 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
        >
          Previous
        </button>
      ) : (
        <span />
      )}
      {!hideNext && (
        <button
          type="button"
          onClick={onNext}
          disabled={nextDisabled}
          className="px-6 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition-colors disabled:bg-gray-300 disabled:cursor-not-allowed"
        >
          {nextLabel}
        </button>
      )}
    </div>
  );
}

export function ErrorBanner({ message }: { message: string | null }) {
  if (!message) return null;
  return (
    <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-md">
      {message}
    </div>
  );
}
