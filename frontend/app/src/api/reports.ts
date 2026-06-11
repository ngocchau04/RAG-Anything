import { API_BASE_URL, apiPost } from "./client";
import type { ChatMessage } from "./chat";

export type ReportResponse = {
  ok: boolean;
  message?: string;
  filename?: string;
  file_path?: string;
  open_url?: string;
  download_url?: string;
  error?: string;
  metadata?: {
    report_mode?: string;
    source_selection_mode?: string;
    source_doc_id?: string;
    source_filename?: string;
    inferred_report_type?: string;
    available_sources?: string[];
    candidate_sources?: Array<{
      doc_id: string;
      filename: string;
      score: number;
      reason: string;
    }>;
    excluded_candidates?: Array<{
      doc_id: string;
      filename: string;
      file_type: string;
      score: number;
      reason: string;
      exclusion_reason?: string | null;
      contaminated_for_request?: boolean;
    }>;
    sections_generated?: string[];
    source_validation?: string;
  };
};

export function exportCurrentAnswer(payload: {
  question: string;
  answer: string;
  source_file?: string;
}) {
  return apiPost<ReportResponse>("/reports/current-answer", payload);
}

export function exportChatHistory(payload: {
  messages: ChatMessage[];
  selected_doc_id?: string;
  source_file?: string;
}) {
  return apiPost<ReportResponse>("/reports/chat-history", payload);
}

export function generateAgentReport(payload: {
  report_request?: string;
  request?: string;
  structured?: boolean;
}) {
  return apiPost<ReportResponse>("/reports/agent", payload);
}

export function toDownloadUrl(path: string | undefined): string | null {
  if (!path) {
    return null;
  }
  if (path.startsWith("http://") || path.startsWith("https://")) {
    return path;
  }
  return `${API_BASE_URL}${path}`;
}
