import { useCallback, useEffect, useState } from "react";

import {
  deleteDocument,
  getDocuments,
  indexDocument,
  reprocessDocument,
  type DocumentItem,
  type DocumentsResponse,
} from "../api/documents";
import type { StatusBannerState } from "./StatusBanner";

type DocumentListProps = {
  refreshKey?: number;
  onRefresh: () => void;
  setStatus: (status: StatusBannerState | null) => void;
  globalActionRunning: boolean;
  setGlobalActionRunning: (value: boolean) => void;
};

export default function DocumentList({
  refreshKey = 0,
  onRefresh,
  setStatus,
  globalActionRunning,
  setGlobalActionRunning,
}: DocumentListProps) {
  const [documents, setDocuments] = useState<DocumentItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [activeAction, setActiveAction] = useState<{
    docId: string;
    type: "index" | "reprocess" | "delete";
  } | null>(null);

  const loadDocuments = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const payload: DocumentsResponse = await getDocuments();
      setDocuments(payload.documents ?? []);
    } catch (err) {
      const message =
        err instanceof Error ? err.message : "Failed to load documents.";
      setError(message);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void loadDocuments();
  }, [loadDocuments, refreshKey]);

  async function runDocumentAction(
    doc: DocumentItem,
    type: "index" | "reprocess" | "delete"
  ) {
    setActiveAction({ docId: doc.doc_id, type });
    setGlobalActionRunning(true);
    setError(null);
    const actionLabel =
      type === "index"
        ? "Indexing document... This may take a while."
        : type === "reprocess"
          ? "Reprocessing document... This may take a while."
          : "Deleting document...";
    setStatus({ type: "info", message: actionLabel });
    try {
      if (type === "index") {
        await indexDocument(doc.doc_id);
      } else if (type === "reprocess") {
        await reprocessDocument(doc.doc_id);
      } else {
        await deleteDocument(doc.doc_id);
      }
      await loadDocuments();
      onRefresh();
      setStatus({
        type: "success",
        message:
          type === "index"
            ? `Document indexed successfully: ${doc.filename}`
            : type === "reprocess"
              ? `Document reprocessed successfully: ${doc.filename}`
              : `Document deleted successfully: ${doc.filename}`,
      });
    } catch (err) {
      await loadDocuments();
      const message =
        err instanceof Error
          ? err.message
          : type === "index"
            ? "Index failed."
            : type === "reprocess"
              ? "Reprocess failed."
              : "Delete failed.";
      setError(message);
      setStatus({ type: "error", message });
    } finally {
      setActiveAction(null);
      setGlobalActionRunning(false);
    }
  }

  function getStatusVariant(doc: DocumentItem): "success" | "error" | "info" | "warning" {
    if (doc.status === "failed") {
      return "error";
    }
    if (doc.needs_reprocess) {
      return "warning";
    }
    if (doc.status === "indexed") {
      return "success";
    }
    return "info";
  }

  function getStatusText(doc: DocumentItem): string {
    if (doc.needs_reprocess) {
      return "needs reprocess";
    }
    return doc.status;
  }

  return (
    <section className="panel">
      <h2>Documents</h2>
      {loading ? <p>Loading documents...</p> : null}
      {error ? <p className="panel-error">{error}</p> : null}
      {!loading && !documents.length ? (
        <p>No indexed documents found.</p>
      ) : null}
      {documents.map((doc) => (
        <article className="document-item" key={doc.doc_id}>
          <div className="document-item__header">
            <strong>{doc.filename}</strong>
            <div className="panel-actions">
              <button
                type="button"
                onClick={() => void runDocumentAction(doc, "index")}
                disabled={
                  activeAction?.docId === doc.doc_id ||
                  globalActionRunning ||
                  (doc.status === "indexed" && !doc.needs_reprocess)
                }
              >
                {activeAction?.docId === doc.doc_id && activeAction.type === "index"
                  ? "Indexing..."
                  : "Index"}
              </button>
              <button
                type="button"
                onClick={() => void runDocumentAction(doc, "reprocess")}
                disabled={
                  activeAction?.docId === doc.doc_id ||
                  globalActionRunning ||
                  doc.status === "uploaded"
                }
              >
                {activeAction?.docId === doc.doc_id && activeAction.type === "reprocess"
                  ? "Reprocessing..."
                  : "Reprocess"}
              </button>
              <button
                type="button"
                onClick={() => void runDocumentAction(doc, "delete")}
                disabled={activeAction?.docId === doc.doc_id || globalActionRunning}
              >
                {activeAction?.docId === doc.doc_id && activeAction.type === "delete"
                  ? "Deleting..."
                  : "Delete"}
              </button>
            </div>
          </div>
          <div className="document-item__meta">
            <span className={`status-badge status-badge--${getStatusVariant(doc)}`}>
              {getStatusText(doc)}
            </span>
            <span className="status-badge status-badge--neutral">{doc.file_type}</span>
          </div>
          <p>Type: {doc.file_type}</p>
          {doc.needs_reprocess ? <p>Needs reprocess: yes</p> : null}
          {doc.error_message ? (
            <p className="panel-error">Error: {doc.error_message}</p>
          ) : null}
        </article>
      ))}
    </section>
  );
}
