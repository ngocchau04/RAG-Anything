import { apiPost } from "./client";

export function exportCurrentAnswer(payload: {
  question: string;
  answer: string;
  source_file?: string;
}) {
  return apiPost("/reports/current-answer", payload);
}

export function exportChatHistory(payload: {
  messages: Array<{ role: string; content: string }>;
  selected_doc_label?: string;
}) {
  return apiPost("/reports/chat-history", payload);
}

export function generateAgentReport(payload: {
  report_request: string;
  selected_doc_label?: string;
}) {
  return apiPost("/reports/agent", payload);
}
