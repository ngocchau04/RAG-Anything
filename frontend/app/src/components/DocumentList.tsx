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
  onDocumentsChanged: (documents: DocumentItem[]) => void;
};

export default function DocumentList({
  refreshKey = 0,
  onRefresh,
  setStatus,
  globalActionRunning,
  setGlobalActionRunning,
  onDocumentsChanged,
}: DocumentListProps) {
  const [documents, setDocuments] = useState<DocumentItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [activeAction, setActiveAction] = useState<{
    docId: string;
    type: "index" | "reprocess" | "delete";
  } | null>(null);

  const loadDocuments = useCallback(async () => {
    // Reload the persisted registry so corpus chat and report UI stay aligned
    // after uploads, indexing, deletes, and backend restarts.
    setLoading(true);
    setError(null);
    try {
      const payload: DocumentsResponse = await getDocuments();
      const nextDocuments = payload.documents ?? [];
      setDocuments(nextDocuments);
      onDocumentsChanged(nextDocuments);
    } catch (err) {
      const message =
        err instanceof Error ? err.message : "Failed to load documents.";
      setError(message);
      onDocumentsChanged([]);
    } finally {
      setLoading(false);
    }
  }, [onDocumentsChanged]);

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

  function getChatAvailabilityReason(doc: DocumentItem): string {
    if (doc.needs_reprocess) {
      return "Reprocess this document before corpus chat can use it.";
    }
    if (doc.status === "failed") {
      return "Fix the indexing error before corpus chat can use it.";
    }
    if (doc.status !== "indexed") {
      return "Index this document before corpus chat can use it.";
    }
    return "Available for corpus chat.";
  }

  function summarizeError(doc: DocumentItem): string | null {
    if (!doc.error_message) {
      return null;
    }
    const compact = doc.error_message.trim().replace(/\s+/g, " ");
    return compact.length > 120 ? `${compact.slice(0, 117)}...` : compact;
  }

  return (
    <section className="panel panel--fill">
      <h2>Documents</h2>
      <p className="panel-subtitle">
        Indexed files stay available for workspace-level chat after restart.
      </p>
      {loading ? <p>Loading documents...</p> : null}
      {error ? <p className="panel-error">{error}</p> : null}
      {!loading && !documents.length ? (
        <p>No documents yet. Upload a file to get started.</p>
      ) : null}
      <div className="document-list">
        {documents.map((doc) => {
          const availabilityReason = getChatAvailabilityReason(doc);
          const errorSummary = summarizeError(doc);

          return (
            <article className="document-item" key={doc.doc_id}>
              <div className="document-item__header">
                <div className="document-item__title-wrap">
                  <strong className="document-item__title" title={doc.filename}>
                    {doc.filename}
                  </strong>
                  <div className="document-item__meta">
                    <span
                      className={`status-badge status-badge--${getStatusVariant(doc)}`}
                    >
                      {getStatusText(doc)}
                    </span>
                    <span className="status-badge status-badge--neutral">
                      {doc.file_type}
                    </span>
                    {doc.available_for_chat ? (
                      <span className="status-badge status-badge--success">
                        corpus chat ready
                      </span>
                    ) : null}
                  </div>
                </div>
              </div>
              <p className="document-item__hint">{availabilityReason}</p>
              {errorSummary ? (
                <p className="document-item__summary document-item__summary--error">
                  {errorSummary}
                </p>
              ) : null}
              <div className="document-item__actions">
                <div className="panel-actions panel-actions--secondary">
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
                    {activeAction?.docId === doc.doc_id &&
                    activeAction.type === "reprocess"
                      ? "Reprocessing..."
                      : "Reprocess"}
                  </button>
                  <button
                    type="button"
                    onClick={() => void runDocumentAction(doc, "delete")}
                    disabled={
                      activeAction?.docId === doc.doc_id || globalActionRunning
                    }
                  >
                    {activeAction?.docId === doc.doc_id &&
                    activeAction.type === "delete"
                      ? "Deleting..."
                      : "Delete"}
                  </button>
                </div>
              </div>
            </article>
          );
        })}
      </div>
    </section>
  );
}
