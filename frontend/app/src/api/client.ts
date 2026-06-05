export const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

type ErrorPayload = {
  error?: unknown;
  message?: unknown;
  detail?: unknown;
};

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
