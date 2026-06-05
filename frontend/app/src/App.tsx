import { useCallback, useState } from "react";

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

  const triggerRefresh = useCallback(() => {
    setRefreshKey((current) => current + 1);
  }, []);

  return (
    <main className="app-shell">
      <header className="app-header">
        <h1>RAG-Anything Future Frontend</h1>
        <p>React/Vite scaffold is ready. Gradio remains the active UI for now.</p>
      </header>
      <StatusBanner status={status} onDismiss={() => setStatus(null)} />
      <section className="app-grid">
        <DocumentList
          refreshKey={refreshKey}
          setStatus={setStatus}
          globalActionRunning={globalActionRunning}
          setGlobalActionRunning={setGlobalActionRunning}
          onRefresh={triggerRefresh}
        />
        <FileUploader
          onUploaded={triggerRefresh}
          setStatus={setStatus}
          globalActionRunning={globalActionRunning}
          setGlobalActionRunning={setGlobalActionRunning}
        />
        <ChatPanel />
        <ReportPanel />
      </section>
    </main>
  );
}
