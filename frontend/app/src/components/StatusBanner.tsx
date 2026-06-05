import { useEffect } from "react";

export type StatusBannerType = "success" | "error" | "info" | "warning";

export type StatusBannerState = {
  message: string;
  type: StatusBannerType;
};

type StatusBannerProps = {
  status: StatusBannerState | null;
  onDismiss: () => void;
};

const AUTO_DISMISS_MS = 4000;

export default function StatusBanner({ status, onDismiss }: StatusBannerProps) {
  useEffect(() => {
    if (!status) {
      return;
    }
    if (status.type === "error" || status.type === "warning") {
      return;
    }
    const timer = window.setTimeout(() => {
      onDismiss();
    }, AUTO_DISMISS_MS);
    return () => window.clearTimeout(timer);
  }, [status, onDismiss]);

  if (!status) {
    return null;
  }

  return (
    <div className={`status-banner status-banner--${status.type}`} role="status">
      <span>{status.message}</span>
      <button type="button" onClick={onDismiss} aria-label="Dismiss status message">
        ×
      </button>
    </div>
  );
}
