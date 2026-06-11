import { apiDelete, apiGet, apiPost, apiPostForm } from "./client";

export type DocumentItem = {
  doc_id: string;
  filename: string;
  status: string;
  file_type: string;
  stored_file_rel?: string;
  working_dir_rel?: string;
  created_at?: string | null;
  updated_at?: string | null;
  error_message?: string | null;
  needs_reprocess?: boolean;
  available_for_chat?: boolean;
};

export type DocumentsResponse = {
  documents: DocumentItem[];
};

export type DeleteDocumentResponse = {
  ok: boolean;
  deleted_doc_id: string;
  message?: string;
  error?: string;
};

export type UploadDocumentResponse = {
  ok: boolean;
  document?: DocumentItem;
  message?: string;
  error?: string;
};

export type IndexDocumentResponse = {
  ok: boolean;
  document?: DocumentItem | null;
  message?: string;
  error?: string;
};

export function getDocuments(): Promise<DocumentsResponse> {
  return apiGet("/documents");
}

export async function uploadDocument(file: File): Promise<UploadDocumentResponse> {
  const formData = new FormData();
  formData.append("file", file);
  return apiPostForm("/documents/upload", formData);
}

export function indexDocument(docId: string): Promise<IndexDocumentResponse> {
  return apiPost(`/documents/${docId}/index`, {});
}

export function reprocessDocument(docId: string): Promise<IndexDocumentResponse> {
  return apiPost(`/documents/${docId}/reprocess`, {});
}

export function deleteDocument(docId: string): Promise<DeleteDocumentResponse> {
  return apiDelete(`/documents/${docId}`);
}
