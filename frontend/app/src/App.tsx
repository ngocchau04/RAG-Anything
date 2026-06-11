import { useCallback, useState } from "react";

import type { ChatMessage } from "./api/chat";
import type { DocumentItem } from "./api/documents";
import type { ReportResponse } from "./api/reports";
import ChatPanel from "./components/ChatPanel";
import DocumentList from "./components/DocumentList";
import FileUploader from "./components/FileUploader";
import ReportPanel from "./components/ReportPanel";
import StatusBanner, {
  type StatusBannerState,
} from "./components/StatusBanner";

export default function App() {
  const [refreshKey, setRefreshKey] = useState(0);
  const [status, setStatus] = useState<StatusBannerState | null>(null);
  const [globalActionRunning, setGlobalActionRunning] = useState(false);
  const [documents, setDocuments] = useState<DocumentItem[]>([]);
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([]);
  const [lastQuestion, setLastQuestion] = useState("");
  const [currentAnswer, setCurrentAnswer] = useState("");
  const [chatExportResult, setChatExportResult] = useState<ReportResponse | null>(null);

  const triggerRefresh = useCallback(() => {
    setRefreshKey((current) => current + 1);
  }, []);

  const indexedDocuments = documents.filter(
    (document) => document.available_for_chat || (document.status === "indexed" && !document.needs_reprocess)
  );

  return (
    <main className="app-shell">
      <header className="app-header">
        <h1>RAG-Anything Web UI</h1>
        <p>Upload and index many files, then chat across all indexed files in one workspace conversation.</p>
      </header>
      <StatusBanner status={status} onDismiss={() => setStatus(null)} />
      <section className="app-layout">
        <aside className="app-sidebar">
          <FileUploader
            onUploaded={triggerRefresh}
            setStatus={setStatus}
            globalActionRunning={globalActionRunning}
            setGlobalActionRunning={setGlobalActionRunning}
          />
          <DocumentList
            refreshKey={refreshKey}
            setStatus={setStatus}
            globalActionRunning={globalActionRunning}
            setGlobalActionRunning={setGlobalActionRunning}
            onRefresh={triggerRefresh}
            onDocumentsChanged={setDocuments}
          />
        </aside>
        <section className="app-main">
          <ChatPanel
            indexedDocuments={indexedDocuments}
            messages={chatMessages}
            setMessages={setChatMessages}
            setLastQuestion={setLastQuestion}
            setCurrentAnswer={setCurrentAnswer}
            chatExportResult={chatExportResult}
            setChatExportResult={setChatExportResult}
            setStatus={setStatus}
          />
          <ReportPanel documents={indexedDocuments} setStatus={setStatus} />
        </section>
      </section>
    </main>
  );
}
