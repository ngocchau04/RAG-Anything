import { useEffect, useMemo, useState } from "react";

import type { DocumentItem } from "../api/documents";
import {
  generateAgentReport,
  toDownloadUrl,
  type ReportResponse,
} from "../api/reports";
import type { StatusBannerState } from "./StatusBanner";

type ReportPanelProps = {
  documents: DocumentItem[];
  setStatus: (status: StatusBannerState | null) => void;
};

type RunningAction = "agent" | null;

export default function ReportPanel({ documents, setStatus }: ReportPanelProps) {
  const [runningAction, setRunningAction] = useState<RunningAction>(null);
  const [reportRequest, setReportRequest] = useState(
    "Generate a technical report about this paper, including objective, method, dataset, formulas, results, limitations, and key findings."
  );
  const [downloadInfo, setDownloadInfo] = useState<ReportResponse | null>(null);
  const [inlineError, setInlineError] = useState<string | null>(null);

  useEffect(() => {
    if (!documents.length) {
      setDownloadInfo(null);
      setInlineError(null);
    }
  }, [documents]);

  const indexedDocuments = useMemo(
    () =>
      documents.filter(
        (document) => document.status === "indexed" && !document.needs_reprocess
      ),
    [documents]
  );

  const reportBlockedReason = useMemo(() => {
    if (!documents.length) {
      return "Index at least one document to enable structured agent reports.";
    }
    if (!indexedDocuments.length) {
      return "At least one indexed document is required before generating a report.";
    }
    return null;
  }, [documents.length, indexedDocuments.length]);

  const openReportUrl = useMemo(
    () => toDownloadUrl(downloadInfo?.open_url ?? downloadInfo?.download_url),
    [downloadInfo]
  );
  const downloadReportUrl = useMemo(
    () => toDownloadUrl(downloadInfo?.download_url),
    [downloadInfo]
  );

  async function runReportAction() {
    if (reportBlockedReason || !reportRequest.trim()) {
      return;
    }
    setRunningAction("agent");
    setInlineError(null);
    setStatus({ type: "info", message: "Generating structured agent report..." });
    try {
      const response = await generateAgentReport({
        request: reportRequest,
        structured: true,
      });
      setDownloadInfo(response);
      setStatus({
        type: "success",
        message:
          response.message ?? "Structured agent report generated successfully.",
      });
    } catch (error) {
      const message =
        error instanceof Error ? error.message : "Structured agent report request failed.";
      setInlineError(message);
      setStatus({ type: "error", message });
    } finally {
      setRunningAction(null);
    }
  }

  const reportDisabledReason =
    reportBlockedReason || (!reportRequest.trim() ? "Enter a report request first." : null);

  const detectedSources = downloadInfo?.metadata?.candidate_sources
    ?.map(
      (source) =>
        `${source.filename} (score: ${source.score}; reason: ${source.reason})`
    )
    .join(", ");
  const excludedSources = downloadInfo?.metadata?.excluded_candidates
    ?.map(
      (source) =>
        `${source.filename} (${source.exclusion_reason ?? source.reason})`
    )
    .join(", ");

  return (
    <section className="panel panel--compact panel--secondary report-panel">
      <h2>Structured Agent Report</h2>
      <p className="panel-subtitle">
        Generate a multi-section PDF report. The backend auto-detects the best indexed source document, infers the report type from your request, validates sources, and exports a formatted PDF.
      </p>
      {inlineError ? <p className="panel-error">{inlineError}</p> : null}
      <div className="report-panel__agent">
        <label className="report-panel__label" htmlFor="report-request">
          Report request
        </label>
        <textarea
          id="report-request"
          value={reportRequest}
          onChange={(event) => setReportRequest(event.target.value)}
          rows={4}
          disabled={Boolean(runningAction)}
        />
        {reportDisabledReason ? (
          <p className="report-panel__hint">{reportDisabledReason}</p>
        ) : null}
        <button
          type="button"
          disabled={Boolean(runningAction || reportDisabledReason)}
          onClick={() => void runReportAction()}
        >
          {runningAction === "agent"
            ? "Generating..."
            : "Generate structured agent report"}
        </button>
      </div>
      {downloadInfo?.metadata ? (
        <div className="report-panel__hint">
          <strong>Detected source:</strong>{" "}
          {downloadInfo.metadata.source_filename ?? "No source selected"}
          <br />
          <strong>Inferred report type:</strong>{" "}
          {downloadInfo.metadata.inferred_report_type ?? "custom"}
          <br />
          <strong>Source validation:</strong>{" "}
          {downloadInfo.metadata.source_validation ?? "unknown"}
          {detectedSources ? (
            <>
              <br />
              <strong>Candidate sources:</strong> {detectedSources}
            </>
          ) : null}
          {excludedSources ? (
            <>
              <br />
              <strong>Excluded sources:</strong> {excludedSources}
            </>
          ) : null}
          {downloadInfo.metadata.sections_generated?.length ? (
            <>
              <br />
              <strong>Generated sections:</strong>{" "}
              {downloadInfo.metadata.sections_generated.join(", ")}
            </>
          ) : null}
        </div>
      ) : null}
      {downloadInfo?.filename ? (
        <div className="report-panel__download">
          <strong>Document report:</strong> <span>{downloadInfo.filename}</span>
          {openReportUrl ? (
            <a href={openReportUrl} target="_blank" rel="noreferrer">
              Open PDF
            </a>
          ) : null}
          {downloadReportUrl ? <a href={downloadReportUrl}>Download PDF</a> : null}
        </div>
      ) : null}
    </section>
  );
}
