export const API_BASE_URL =
import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

type ErrorPayload = {
  error?: unknown;
  message?: unknown;
  detail?: unknown;
  details?: unknown;
};

function formatCandidateList(
  value: unknown,
  label: string,
  detailKey: "reason" | "exclusion_reason" = "reason"
): string | null {
  if (!Array.isArray(value) || !value.length) {
    return null;
  }
  const parts = value
    .map((item) => {
      if (!item || typeof item !== "object") {
        return null;
      }
      const candidate = item as Record<string, unknown>;
      const filename = normalizeErrorValue(candidate.filename) ?? "unknown";
      const detail = normalizeErrorValue(candidate[detailKey]);
      return detail ? `${filename} (${detail})` : filename;
    })
    .filter((item): item is string => Boolean(item));
  return parts.length ? `${label}: ${parts.join(", ")}` : null;
}

function normalizeErrorValue(value: unknown): string | null {
  if (typeof value === "string" && value.trim()) {
    return value.trim();
  }
  if (Array.isArray(value)) {
    const parts = value
      .map((item) => normalizeErrorValue(item))
      .filter((item): item is string => Boolean(item));
    return parts.length ? parts.join(", ") : null;
  }
  if (value && typeof value === "object") {
    const candidate = value as Record<string, unknown>;
    if ("candidate_sources" in candidate || "excluded_candidates" in candidate) {
      const parts: string[] = [];
      const baseMessage =
        normalizeErrorValue(candidate.error) ?? normalizeErrorValue(candidate.message);
      if (baseMessage) {
        parts.push(baseMessage);
      }
      const candidateSources = formatCandidateList(
        candidate.candidate_sources,
        "Candidate sources"
      );
      if (candidateSources) {
        parts.push(candidateSources);
      }
      const excludedCandidates = formatCandidateList(
        candidate.excluded_candidates,
        "Excluded candidates",
        "exclusion_reason"
      );
      if (excludedCandidates) {
        parts.push(excludedCandidates);
      }
      return parts.length ? parts.join(" | ") : null;
    }
    // Surface Ollama diagnostics in a readable format so the UI does not hide
    // the requested model / available model mismatch behind raw JSON.
    if (
      "requested_model" in candidate ||
      "available_models" in candidate ||
      "suggestion" in candidate
    ) {
      const parts: string[] = [];
      if (normalizeErrorValue(candidate.requested_model)) {
        parts.push(`Requested model: ${normalizeErrorValue(candidate.requested_model)}`);
      }
      if (normalizeErrorValue(candidate.resolved_model)) {
        parts.push(`Resolved model: ${normalizeErrorValue(candidate.resolved_model)}`);
      }
      if (Array.isArray(candidate.available_models)) {
        const available = candidate.available_models
          .map((item) => normalizeErrorValue(item))
          .filter((item): item is string => Boolean(item));
        parts.push(
          available.length
            ? `Available models: ${available.join(", ")}`
            : "Available models: none reported"
        );
      }
      if (normalizeErrorValue(candidate.ollama_host)) {
        parts.push(`Ollama host: ${normalizeErrorValue(candidate.ollama_host)}`);
      }
      if (normalizeErrorValue(candidate.suggestion)) {
        parts.push(`Suggestion: ${normalizeErrorValue(candidate.suggestion)}`);
      }
      return parts.join(" | ");
    }
    try {
      return JSON.stringify(value);
    } catch {
      return null;
    }
  }
  return null;
}

async function parseErrorResponse(response: Response, fallback: string): Promise<string> {
  try {
    const payload = (await response.json()) as ErrorPayload;
    return (
      normalizeErrorValue(payload.error) ??
      normalizeErrorValue(payload.message) ??
      normalizeErrorValue(payload.details) ??
      normalizeErrorValue(payload.detail) ??
      fallback
    );
  } catch {
    return fallback;
  }
}

function toConnectionError(error: unknown): Error {
  if (error instanceof Error && error.name === "TypeError") {
    return new Error(
      "Could not connect to backend. Please make sure FastAPI is running on http://localhost:8000."
    );
  }
  return error instanceof Error ? error : new Error("Unexpected frontend error.");
}

export async function apiGet<T>(path: string): Promise<T> {
  try {
    const response = await fetch(`${API_BASE_URL}${path}`);
    if (!response.ok) {
      throw new Error(
        await parseErrorResponse(response, `GET ${path} failed: ${response.status}`)
      );
    }
    return response.json() as Promise<T>;
  } catch (error) {
    throw toConnectionError(error);
  }
}

export async function apiPost<T>(path: string, payload?: unknown): Promise<T> {
  try {
    const response = await fetch(`${API_BASE_URL}${path}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: payload === undefined ? undefined : JSON.stringify(payload),
    });
    if (!response.ok) {
      throw new Error(
        await parseErrorResponse(response, `POST ${path} failed: ${response.status}`)
      );
    }
    return response.json() as Promise<T>;
  } catch (error) {
    throw toConnectionError(error);
  }
}

export async function apiDelete<T>(path: string): Promise<T> {
  try {
    const response = await fetch(`${API_BASE_URL}${path}`, { method: "DELETE" });
    if (!response.ok) {
      throw new Error(
        await parseErrorResponse(response, `DELETE ${path} failed: ${response.status}`)
      );
    }
    return response.json() as Promise<T>;
  } catch (error) {
    throw toConnectionError(error);
  }
}

export async function apiPostForm<T>(path: string, formData: FormData): Promise<T> {
  try {
    const response = await fetch(`${API_BASE_URL}${path}`, {
      method: "POST",
      body: formData,
    });
    if (!response.ok) {
      throw new Error(
        await parseErrorResponse(response, `POST ${path} failed: ${response.status}`)
      );
    }
    return response.json() as Promise<T>;
  } catch (error) {
    throw toConnectionError(error);
  }
}
