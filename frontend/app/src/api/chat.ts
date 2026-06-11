import { apiPost } from "./client";

export type ChatMessage = {
  role: "user" | "assistant";
  content: string;
  sourceFilename?: string;
  sourceDocId?: string;
  sources?: ChatSourceRef[];
};

export type ChatDocumentRef = {
  doc_id: string;
  filename: string;
  status?: string;
};

export type ChatSourceRef = {
  doc_id: string;
  filename: string;
  score?: number;
  reason?: string;
};

export type ChatResponse = {
  ok: boolean;
  answer: string;
  document?: ChatDocumentRef | null;
  sources?: ChatSourceRef[];
  metadata?: {
    mode?: string;
    indexed_document_count?: number;
    selected_doc_id?: string;
    selected_filename?: string;
    working_dir?: string;
    source_files?: string[];
    referenced_files?: string[];
    query_mode?: string;
    [key: string]: unknown;
  };
  history?: ChatMessage[];
  error?: string;
};

export type SendChatPayload = {
  message: string;
  selectedDocId?: string | null;
  history?: ChatMessage[];
  mode?: "corpus" | "selected_document";
};

export function sendChat(payload: SendChatPayload): Promise<ChatResponse> {
  return apiPost(payload.mode === "corpus" ? "/chat/corpus" : "/chat", {
    message: payload.message,
    selected_doc_id: payload.selectedDocId,
    selected_document_id: payload.selectedDocId,
    history: payload.history ?? [],
    mode: payload.mode ?? "selected_document",
    require_selected_document: Boolean(payload.selectedDocId),
  });
}
