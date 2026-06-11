import { useMemo, useState } from "react";

import { sendChat, type ChatMessage } from "../api/chat";
import type { DocumentItem } from "../api/documents";
import {
  exportChatHistory,
  exportCurrentAnswer,
  toDownloadUrl,
  type ReportResponse,
} from "../api/reports";
import type { StatusBannerState } from "./StatusBanner";

type ChatPanelProps = {
  indexedDocuments: DocumentItem[];
  messages: ChatMessage[];
  setMessages: (messages: ChatMessage[]) => void;
  setLastQuestion: (value: string) => void;
  setCurrentAnswer: (value: string) => void;
  chatExportResult: ReportResponse | null;
  setChatExportResult: (value: ReportResponse | null) => void;
  setStatus: (status: StatusBannerState | null) => void;
};

export default function ChatPanel({
  indexedDocuments,
  messages,
  setMessages,
  setLastQuestion,
  setCurrentAnswer,
  chatExportResult,
  setChatExportResult,
  setStatus,
}: ChatPanelProps) {
  const [draft, setDraft] = useState("");
  const [loading, setLoading] = useState(false);
  const [inlineError, setInlineError] = useState<string | null>(null);
  const [exportingAnswer, setExportingAnswer] = useState(false);
  const [exportingHistory, setExportingHistory] = useState(false);

  const indexedDocumentCount = indexedDocuments.length;

  const chatBlockedReason = useMemo(() => {
    if (!indexedDocumentCount) {
      return "Upload and index at least one file to start chatting.";
    }
    return null;
  }, [indexedDocumentCount]);

  const latestAssistantMessage = useMemo(() => {
    for (let index = messages.length - 1; index >= 0; index -= 1) {
      if (messages[index]?.role === "assistant") {
        return messages[index];
      }
    }
    return null;
  }, [messages]);

  const canSend = useMemo(() => {
    return Boolean(draft.trim() && !loading && !chatBlockedReason);
  }, [chatBlockedReason, draft, loading]);

  const sendDisabledReason = useMemo(() => {
    if (loading) {
      return "Please wait for the current answer to finish.";
    }
    if (chatBlockedReason) {
      return chatBlockedReason;
    }
    if (!draft.trim()) {
      return "Type a question to enable Send.";
    }
    return null;
  }, [chatBlockedReason, draft, loading]);

  async function handleSend() {
    const question = draft.trim();
    if (!question || chatBlockedReason) {
      return;
    }

    const previousMessages = messages;
    const nextMessages = [
      ...previousMessages,
      { role: "user" as const, content: question },
    ];
    setMessages(nextMessages);
    setLastQuestion(question);
    setDraft("");
    setInlineError(null);
    setLoading(true);
    setStatus({ type: "info", message: "Generating answer across indexed files..." });

    try {
      const response = await sendChat({
        message: question,
        history: previousMessages,
        mode: "corpus",
      });
      const answer = (response.answer || "").trim() || "No answer was returned.";
      const sources = response.sources ?? [];
      setCurrentAnswer(answer);
      setMessages([
        ...nextMessages,
        {
          role: "assistant",
          content: answer,
          sourceFilename: sources[0]?.filename,
          sourceDocId: sources[0]?.doc_id,
          sources,
        },
      ]);
      const sourceLabel =
        sources.length > 0
          ? `Answer generated from ${sources.map((source) => source.filename).join(", ")}.`
          : "Answer generated from indexed workspace files.";
      setStatus({ type: "success", message: sourceLabel });
    } catch (error) {
      const message =
        error instanceof Error ? error.message : "Chat request failed unexpectedly.";
      const lowered = message.toLowerCase();
      const isNoSource =
        lowered.includes("no relevant indexed file found") ||
        lowered.includes("could not find relevant information");
      if (isNoSource) {
        // No-source is a valid workspace result. Keep the user message and show
        // the backend explanation in the transcript instead of a fake crash.
        setCurrentAnswer(message);
        setMessages([
          ...nextMessages,
          {
            role: "assistant",
            content: message,
            sources: [],
          },
        ]);
        setStatus({ type: "warning", message });
      } else {
        setMessages(previousMessages);
        setCurrentAnswer("");
        setInlineError(message);
        setStatus({ type: "error", message });
      }
    } finally {
      setLoading(false);
    }
  }

  async function handleExportCurrentAnswer() {
    if (!latestAssistantMessage) {
      return;
    }
    setExportingAnswer(true);
    setInlineError(null);
    setStatus({ type: "info", message: "Exporting current answer to PDF..." });
    try {
      const sourceLabel =
        latestAssistantMessage.sources?.map((source) => source.filename).join(", ") ||
        latestAssistantMessage.sourceFilename ||
        "workspace corpus";
      const response = await exportCurrentAnswer({
        question: messages
          .slice()
          .reverse()
          .find((message) => message.role === "user")?.content ?? "",
        answer: latestAssistantMessage.content,
        source_file: sourceLabel,
      });
      setChatExportResult(response);
      setStatus({
        type: "success",
        message: response.message ?? "Current answer exported successfully.",
      });
    } catch (error) {
      const message =
        error instanceof Error ? error.message : "Current answer export failed.";
      setInlineError(message);
      setStatus({ type: "error", message });
    } finally {
      setExportingAnswer(false);
    }
  }

  async function handleExportChatHistory() {
    if (!messages.length) {
      return;
    }
    setExportingHistory(true);
    setInlineError(null);
    setStatus({ type: "info", message: "Exporting chat history to PDF..." });
    try {
      const response = await exportChatHistory({
        messages,
        source_file: "workspace corpus chat",
      });
      setChatExportResult(response);
      setStatus({
        type: "success",
        message: response.message ?? "Chat history exported successfully.",
      });
    } catch (error) {
      const message =
        error instanceof Error ? error.message : "Chat history export failed.";
      setInlineError(message);
      setStatus({ type: "error", message });
    } finally {
      setExportingHistory(false);
    }
  }

  const openChatPdfUrl = toDownloadUrl(
    chatExportResult?.open_url ?? chatExportResult?.download_url
  );
  const downloadChatPdfUrl = toDownloadUrl(chatExportResult?.download_url);

  return (
    <section className="panel chat-panel">
      <div className="chat-panel__header">
        <div>
          <h2>Chat with all indexed files</h2>
          <p className="chat-panel__selection">
            Indexed files available: <strong>{indexedDocumentCount}</strong>
          </p>
        </div>
      </div>
      {chatBlockedReason ? (
        <div className="chat-panel__empty-state">
          <p>{chatBlockedReason}</p>
        </div>
      ) : null}
      {inlineError ? <p className="panel-error">{inlineError}</p> : null}
      <div className="chat-panel__history" aria-live="polite">
        {!messages.length ? (
          <p className="chat-panel__empty">
            {chatBlockedReason
              ? "Once at least one file is indexed, messages will appear here."
              : "Ask a question and the backend will retrieve across the indexed workspace files."}
          </p>
        ) : null}
        {messages.map((message, index) => (
          <article
            key={`${message.role}-${index}`}
            className={`chat-message chat-message--${message.role}`}
          >
            <strong>{message.role === "user" ? "You" : "Assistant"}</strong>
            {message.role === "assistant" && message.sources?.length ? (
              <div className="chat-message__source">
                <span>Sources:</span>
                <ul>
                  {message.sources.map((source) => (
                    <li key={`${source.doc_id}-${source.filename}`}>
                      {source.filename}
                    </li>
                  ))}
                </ul>
              </div>
            ) : null}
            <p>{message.content}</p>
            {message.role === "assistant" && index === messages.length - 1 ? (
              <div className="chat-message__actions">
                <button
                  type="button"
                  disabled={exportingAnswer}
                  onClick={() => void handleExportCurrentAnswer()}
                >
                  {exportingAnswer ? "Exporting..." : "Export current answer"}
                </button>
              </div>
            ) : null}
          </article>
        ))}
      </div>
      <div className="chat-panel__composer">
        <textarea
          value={draft}
          onChange={(event) => setDraft(event.target.value)}
          placeholder="Ask something about any indexed file..."
          rows={6}
          disabled={loading || Boolean(chatBlockedReason)}
        />
        {sendDisabledReason ? (
          <p className="chat-panel__composer-hint">{sendDisabledReason}</p>
        ) : null}
        <div className="chat-panel__actions">
          <button type="button" onClick={() => void handleSend()} disabled={!canSend}>
            {loading ? "Sending..." : "Send"}
          </button>
          <button
            type="button"
            disabled={Boolean(exportingHistory || !messages.length)}
            onClick={() => void handleExportChatHistory()}
          >
            {exportingHistory ? "Exporting..." : "Export chat history"}
          </button>
        </div>
        {chatExportResult?.filename ? (
          <div className="chat-panel__export-result">
            <strong>Chat export:</strong> <span>{chatExportResult.filename}</span>
            {openChatPdfUrl ? (
              <a href={openChatPdfUrl} target="_blank" rel="noreferrer">
                Open PDF
              </a>
            ) : null}
            {downloadChatPdfUrl ? (
              <a href={downloadChatPdfUrl}>Download PDF</a>
            ) : null}
          </div>
        ) : null}
      </div>
    </section>
  );
}
