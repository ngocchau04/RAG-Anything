import { useRef, useState } from "react";

import { uploadDocument } from "../api/documents";
import type { StatusBannerState } from "./StatusBanner";

type FileUploaderProps = {
  onUploaded: () => void | Promise<void>;
  setStatus: (status: StatusBannerState | null) => void;
  globalActionRunning: boolean;
  setGlobalActionRunning: (value: boolean) => void;
};

export default function FileUploader({
  onUploaded,
  setStatus,
  globalActionRunning,
  setGlobalActionRunning,
}: FileUploaderProps) {
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement | null>(null);

  async function handleUpload() {
    if (!file) {
      setError("Please choose a file first.");
      return;
    }
    setLoading(true);
    setError(null);
    setGlobalActionRunning(true);
    setStatus({ type: "info", message: "Uploading document..." });
    try {
      const payload = await uploadDocument(file);
      if (!payload.ok) {
        throw new Error(payload.error ?? "Upload failed.");
      }
      setStatus({
        type: "success",
        message: payload.message ?? "Document uploaded successfully.",
      });
      await onUploaded();
      setFile(null);
      if (inputRef.current) {
        inputRef.current.value = "";
      }
    } catch (err) {
      const message = err instanceof Error ? err.message : "Upload failed.";
      setError(message);
      setStatus({ type: "error", message });
    } finally {
      setLoading(false);
      setGlobalActionRunning(false);
    }
  }

  return (
    <section className="panel">
      <h2>Upload</h2>
      <input
        ref={inputRef}
        type="file"
        onChange={(event) => setFile(event.target.files?.[0] ?? null)}
      />
      {file ? <p>Selected file: {file.name}</p> : <p>No file selected.</p>}
      <div className="panel-actions">
        <button
          type="button"
          onClick={() => void handleUpload()}
          disabled={loading || globalActionRunning || !file}
        >
          {loading ? "Uploading..." : "Upload"}
        </button>
      </div>
      {error ? <p className="panel-error">{error}</p> : null}
      <p>Current production flow remains in Gradio.</p>
    </section>
  );
}
