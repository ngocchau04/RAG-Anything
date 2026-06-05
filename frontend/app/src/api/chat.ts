import { apiPost } from "./client";

export function sendChat(question: string, selectedDocId?: string) {
  return apiPost("/chat", {
    question,
    selected_doc_id: selectedDocId,
  });
}
