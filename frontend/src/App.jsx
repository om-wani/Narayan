import { useEffect, useMemo, useRef, useState } from "react"
import "./App.css"

const API = "/api"

function makeId() {
  if (typeof crypto !== "undefined" && crypto.randomUUID) return crypto.randomUUID()
  return `${Date.now()}-${Math.random().toString(16).slice(2)}`
}

function shortId(id) {
  return id ? `${id.slice(0, 8)}…` : ""
}

function formatCost(cost) {
  if (typeof cost !== "number") return "0.0000"
  return cost.toFixed(4)
}

async function readErrorMessage(response) {
  const text = await response.text()
  if (!text) return `Request failed (${response.status})`
  try {
    const parsed = JSON.parse(text)
    return parsed.detail || parsed.message || text
  } catch {
    return text
  }
}

async function uploadDocument(file) {
  const formData = new FormData()
  formData.append("file", file)
  const response = await fetch(`${API}/ingest/upload`, {
    method: "POST",
    body: formData,
  })
  if (!response.ok) throw new Error(await readErrorMessage(response))
  return response.json()
}

async function fetchDocuments() {
  const response = await fetch(`${API}/ingest/`)
  if (!response.ok) throw new Error(await readErrorMessage(response))
  const data = await response.json()
  return data.documents || []
}

async function deleteDocument(docId) {
  const response = await fetch(`${API}/ingest/${encodeURIComponent(docId)}`, {
    method: "DELETE",
  })
  if (!response.ok) throw new Error(await readErrorMessage(response))
}

async function* readSse(responseBody) {
  if (!responseBody) throw new Error("Stream body missing")
  const reader = responseBody.getReader()
  const decoder = new TextDecoder()
  let buffer = ""

  while (true) {
    const { done, value } = await reader.read()
    if (value) buffer += decoder.decode(value, { stream: !done })

    let separatorIndex = buffer.indexOf("\n\n")
    while (separatorIndex !== -1) {
      const rawEvent = buffer.slice(0, separatorIndex).trim()
      buffer = buffer.slice(separatorIndex + 2)
      if (rawEvent) yield parseSseEvent(rawEvent)
      separatorIndex = buffer.indexOf("\n\n")
    }

    if (done) break
  }

  const tail = buffer.trim()
  if (tail) yield parseSseEvent(tail)
}

function parseSseEvent(rawEvent) {
  const lines = rawEvent.split(/\r?\n/)
  let event = "message"
  const dataLines = []

  for (const line of lines) {
    if (line.startsWith("event:")) {
      event = line.slice(6).trim()
    } else if (line.startsWith("data:")) {
      dataLines.push(line.slice(5).trimStart())
    }
  }

  return { event, data: dataLines.join("\n") }
}

function MessageBubble({ message }) {
  const isUser = message.role === "user"
  return (
    <article className={`message ${isUser ? "user" : "assistant"}`}>
      <div className="message-head">
        <span className="message-role">{isUser ? "You" : "Narayan"}</span>
        {message.pending ? <span className="message-pulse">streaming</span> : null}
      </div>
      <div className="message-body">
        {message.content || (message.pending ? " " : "No response")}
      </div>
      {message.role === "assistant" && message.stats ? (
        <div className="message-stats">
          <span>{message.stats.model || "model"}</span>
          <span>{message.stats.totalTokens || 0} tokens</span>
          <span>${formatCost(message.stats.estimatedCostUsd || 0)} est.</span>
          <span>{message.stats.latencyMs || 0} ms</span>
        </div>
      ) : null}
      {message.role === "assistant" && message.sources?.length ? (
        <div className="sources">
          <div className="section-label">Sources</div>
          <div className="source-grid">
            {message.sources.map((source) => (
              <details key={source.source_id} className="source-card">
                <summary>
                  <span>{source.filename || "Document"}</span>
                  <span>Page {source.page || 0}</span>
                  <span>{Math.round((source.score || 0) * 100)}%</span>
                </summary>
                <p>{source.text}</p>
              </details>
            ))}
          </div>
        </div>
      ) : null}
    </article>
  )
}

export default function App() {
  const [documents, setDocuments] = useState([])
  const [selectedDocIds, setSelectedDocIds] = useState([])
  const [messages, setMessages] = useState([
    {
      id: "welcome",
      role: "assistant",
      content:
        "Drop PDFs, pick docs, then ask a question. Answers stream in real time with citations.",
    },
  ])
  const [question, setQuestion] = useState("")
  const [loading, setLoading] = useState(false)
  const [uploading, setUploading] = useState(false)
  const [error, setError] = useState("")
  const [sessionId] = useState(() => makeId())
  const fileInputRef = useRef(null)
  const endRef = useRef(null)

  const selectedDocsLabel = useMemo(() => {
    if (!selectedDocIds.length) return "All docs"
    if (selectedDocIds.length === 1) return "1 selected doc"
    return `${selectedDocIds.length} selected docs`
  }, [selectedDocIds.length])

  useEffect(() => {
    refreshDocuments().catch((err) => setError(err.message))
  }, [])

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth", block: "end" })
  }, [messages, loading])

  async function refreshDocuments() {
    const next = await fetchDocuments()
    setDocuments(next)
  }

  function toggleDoc(docId) {
    setSelectedDocIds((current) =>
      current.includes(docId)
        ? current.filter((item) => item !== docId)
        : [...current, docId],
    )
  }

  async function handleUpload(event) {
    const files = Array.from(event.target.files || []).filter((file) =>
      file.name.toLowerCase().endsWith(".pdf"),
    )
    event.target.value = ""
    if (!files.length) return

    setUploading(true)
    setError("")
    try {
      for (const file of files) {
        await uploadDocument(file)
      }
      await refreshDocuments()
    } catch (err) {
      setError(err.message)
    } finally {
      setUploading(false)
    }
  }

  async function handleDelete(docId) {
    setError("")
    try {
      await deleteDocument(docId)
      setSelectedDocIds((current) => current.filter((item) => item !== docId))
      await refreshDocuments()
    } catch (err) {
      setError(err.message)
    }
  }

  async function handleSubmit(event) {
    event.preventDefault()
    const text = question.trim()
    if (!text || loading) return

    const userMessage = { id: makeId(), role: "user", content: text }
    const assistantId = makeId()
    setQuestion("")
    setError("")
    setLoading(true)
    const startedAt = performance.now()
    setMessages((current) => [
      ...current,
      userMessage,
      { id: assistantId, role: "assistant", content: "", pending: true },
    ])

    try {
      const response = await fetch(`${API}/chat/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question: text,
          session_id: sessionId,
          doc_ids: selectedDocIds.length ? selectedDocIds : null,
        }),
      })

      if (!response.ok) throw new Error(await readErrorMessage(response))

      let streamedAnswer = ""
      for await (const event of readSse(response.body)) {
        if (event.event === "error") {
          const payload = JSON.parse(event.data)
          throw new Error(payload.message || "Stream failed")
        }

        const payload = JSON.parse(event.data)
        if (payload.type === "delta") {
          streamedAnswer += payload.text || ""
          setMessages((current) =>
            current.map((message) =>
              message.id === assistantId
                ? { ...message, content: streamedAnswer }
                : message,
            ),
          )
        }

        if (payload.type === "done") {
          streamedAnswer = payload.answer || streamedAnswer
          setMessages((current) =>
            current.map((message) =>
              message.id === assistantId
                ? {
                    ...message,
                    content: streamedAnswer,
                    pending: false,
                    sources: payload.sources || [],
                    stats: {
                      model: payload.model,
                      promptTokens: payload.prompt_tokens,
                      completionTokens: payload.completion_tokens,
                      totalTokens: payload.total_tokens,
                      estimatedCostUsd: payload.estimated_cost_usd,
                      latencyMs: Math.round(performance.now() - startedAt),
                    },
                  }
                : message,
            ),
          )
        }
      }
    } catch (err) {
      setError(err.message)
      setMessages((current) =>
        current.map((message) =>
          message.id === assistantId
            ? {
                ...message,
                content: `Error: ${err.message}`,
                pending: false,
              }
            : message,
        ),
      )
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand">
          <div className="brand-mark">N</div>
          <div>
            <div className="brand-title">Narayan Azure RAG</div>
            <div className="brand-subtitle">
              SSE chat, vector search, citations, memory, cost tracking
            </div>
          </div>
        </div>
        <div className="topbar-meta">
          <span className="pill">{documents.length} docs</span>
          <span className="pill">{selectedDocsLabel}</span>
          <span className="pill">session {shortId(sessionId)}</span>
        </div>
      </header>

      <main className="layout">
        <aside className="sidebar">
          <section className="card accent">
            <div className="card-head">
              <div>
                <h2>Corpus</h2>
                <p>Upload PDFs, pick scope, or search across whole collection.</p>
              </div>
              <button
                className="ghost-btn"
                type="button"
                onClick={() => fileInputRef.current?.click()}
                disabled={uploading}
              >
                {uploading ? "Uploading…" : "Upload"}
              </button>
              <input
                ref={fileInputRef}
                type="file"
                accept=".pdf,application/pdf"
                multiple
                hidden
                onChange={handleUpload}
              />
            </div>
            <div className="note">
              Azure AI Search + Azure OpenAI paths live in <code>backend_azure</code>.
            </div>
          </section>

          <section className="card">
            <div className="section-head">
              <h3>Documents</h3>
              <button
                type="button"
                className="text-btn"
                onClick={() => refreshDocuments().catch((err) => setError(err.message))}
              >
                Refresh
              </button>
            </div>

            <div className="doc-list">
              {documents.length === 0 ? (
                <div className="empty-state">
                  No docs yet. Upload PDF to start.
                </div>
              ) : (
                documents.map((doc) => (
                  <div key={doc.doc_id} className="doc-row">
                    <label className="doc-label">
                      <input
                        type="checkbox"
                        checked={selectedDocIds.includes(doc.doc_id)}
                        onChange={() => toggleDoc(doc.doc_id)}
                      />
                      <span>
                        <strong>{doc.filename}</strong>
                        <small>
                          {doc.page_count || 0} pages · {doc.chunk_count || 0} chunks
                        </small>
                      </span>
                    </label>
                    <button
                      type="button"
                      className="danger-btn"
                      onClick={() => handleDelete(doc.doc_id)}
                    >
                      Delete
                    </button>
                  </div>
                ))
              )}
            </div>
          </section>

          <section className="card">
            <h3>Azure checklist</h3>
            <ul className="checklist">
              <li>Azure student subscription</li>
              <li>Foundry project</li>
              <li>Chat deployment</li>
              <li>Embedding deployment</li>
              <li>AI Search index</li>
              <li>Budget alert</li>
            </ul>
          </section>
        </aside>

        <section className="chat-panel">
          <div className="chat-card">
            <div className="section-head chat-head">
              <div>
                <h3>Chat</h3>
                <p>Ask about uploaded docs. SSE stream updates live.</p>
              </div>
              <div className="status-row">
                <span className={`status ${loading ? "live" : "idle"}`}>
                  {loading ? "streaming" : "ready"}
                </span>
              </div>
            </div>

            {error ? <div className="error-banner">{error}</div> : null}

            <div className="messages">
              {messages.map((message) => (
                <MessageBubble key={message.id} message={message} />
              ))}
              <div ref={endRef} />
            </div>

            <form className="composer" onSubmit={handleSubmit}>
              <label className="composer-label" htmlFor="question">
                Question
              </label>
              <textarea
                id="question"
                value={question}
                placeholder="Ask a question about the uploaded papers..."
                rows={4}
                onChange={(event) => setQuestion(event.target.value)}
                onKeyDown={(event) => {
                  if (event.key === "Enter" && !event.shiftKey) {
                    event.preventDefault()
                    handleSubmit(event)
                  }
                }}
              />
              <div className="composer-footer">
                <div className="composer-hint">
                  Press <kbd>Enter</kbd> to send, <kbd>Shift</kbd> + <kbd>Enter</kbd> for newline.
                </div>
                <button type="submit" className="send-btn" disabled={loading}>
                  {loading ? "Working…" : "Send"}
                </button>
              </div>
            </form>
          </div>
        </section>
      </main>
    </div>
  )
}
