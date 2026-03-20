import { useState, useRef, useEffect } from "react"

const API = "/api"

async function uploadDoc(file) {
  const fd = new FormData()
  fd.append("file", file)
  const r = await fetch(`${API}/documents/upload`, { method: "POST", body: fd })
  if (!r.ok) throw new Error((await r.json()).detail || "Upload failed")
  return r.json()
}

async function fetchDocs() {
  const r = await fetch(`${API}/documents/`)
  return (await r.json()).documents
}

async function deleteDoc(id) {
  await fetch(`${API}/documents/${id}`, { method: "DELETE" })
}

async function askQuestion(question, docIds) {
  const r = await fetch(`${API}/query/`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      question,
      doc_ids: docIds?.length ? docIds : null,
    }),
  })
  if (!r.ok) throw new Error((await r.json()).detail || "Query failed")
  return r.json()
}

function SourceCard({ src, index }) {
  const [open, setOpen] = useState(false)
  return (
    <div style={{
      border: "1px solid #e2e8f0", borderRadius: 8,
      overflow: "hidden", fontSize: 13, marginBottom: 6,
    }}>
      <div
        onClick={() => setOpen(v => !v)}
        style={{
          display: "flex", alignItems: "center", gap: 10,
          padding: "8px 12px", background: "#f8fafc",
          cursor: "pointer", userSelect: "none",
        }}
      >
        <span style={{
          background: "#dbeafe", color: "#1e40af",
          borderRadius: 4, padding: "1px 7px",
          fontFamily: "monospace", fontSize: 12, fontWeight: 600,
        }}>
          [{index}]
        </span>
        <span style={{ flex: 1, color: "#374151", fontWeight: 500 }}>
          {src.filename}
        </span>
        <span style={{ color: "#9ca3af" }}>Page {src.page}</span>
        <span style={{
          background: "#dcfce7", color: "#166534",
          borderRadius: 4, padding: "1px 6px", fontSize: 11,
        }}>
          {Math.round(src.score * 100)}% match
        </span>
        <span style={{ color: "#9ca3af" }}>{open ? "▲" : "▼"}</span>
      </div>
      {open && (
        <div style={{
          padding: "10px 14px", background: "#fff",
          color: "#6b7280", lineHeight: 1.6,
          borderTop: "1px solid #e2e8f0",
        }}>
          {src.text}
        </div>
      )}
    </div>
  )
}

function Answer({ data }) {
  if (!data) return null
  const parts = data.answer.split(/(\[Source \d+\])/g)
  return (
    <div style={{ marginTop: 24 }}>
      <div style={{
        background: "#f0fdf4", border: "1px solid #bbf7d0",
        borderRadius: 12, padding: "18px 20px",
        lineHeight: 1.8, color: "#111827", fontSize: 14,
      }}>
        {parts.map((part, i) =>
          /\[Source \d+\]/.test(part)
            ? <span key={i} style={{
                color: "#1d4ed8", fontFamily: "monospace",
                fontSize: 12, background: "#dbeafe",
                borderRadius: 3, padding: "1px 5px",
              }}>{part}</span>
            : <span key={i}>{part}</span>
        )}
        <div style={{
          marginTop: 12, paddingTop: 10,
          borderTop: "1px solid #dcfce7",
          fontSize: 12, color: "#6b7280",
          display: "flex", gap: 16,
        }}>
          <span>model: {data.model}</span>
          {data.tokens_used > 0 && <span>tokens: {data.tokens_used}</span>}
        </div>
      </div>

      {data.sources.length > 0 && (
        <div style={{ marginTop: 14 }}>
          <div style={{
            fontSize: 11, color: "#9ca3af",
            marginBottom: 8, letterSpacing: "0.06em",
            fontWeight: 600,
          }}>
            SOURCES ({data.sources.length}) — click to expand
          </div>
          {data.sources.map((s, i) => (
            <SourceCard key={i} src={s} index={i + 1} />
          ))}
        </div>
      )}
    </div>
  )
}

export default function App() {
  const [docs, setDocs] = useState([])
  const [selectedDocs, setSelectedDocs] = useState([])
  const [question, setQuestion] = useState("")
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [uploading, setUploading] = useState(false)
  const [error, setError] = useState(null)
  const fileRef = useRef()

  useEffect(() => { fetchDocs().then(setDocs).catch(() => {}) }, [])

  const toggleDoc = (id) =>
    setSelectedDocs(p => p.includes(id) ? p.filter(x => x !== id) : [...p, id])

  const handleUpload = async (files) => {
    setUploading(true)
    setError(null)
    for (const f of files) {
      try {
        await uploadDoc(f)
      } catch (e) {
        setError(e.message)
      }
    }
    setDocs(await fetchDocs())
    setUploading(false)
  }

  const handleAsk = async () => {
    if (!question.trim() || loading) return
    setLoading(true)
    setError(null)
    setResult(null)
    try {
      const r = await askQuestion(question, selectedDocs)
      setResult(r)
    } catch (e) {
      setError(e.message)
    }
    setLoading(false)
  }

  return (
    <div style={{
      minHeight: "100vh", background: "#f9fafb",
      fontFamily: "system-ui, sans-serif",
    }}>
      {/* Header */}
      <div style={{
        background: "#fff", borderBottom: "1px solid #e5e7eb",
        padding: "14px 32px", display: "flex",
        alignItems: "center", gap: 12,
      }}>
      <div style={{
        width: 32, height: 32, borderRadius: 8,
        background: "#1d4ed8", display: "flex",
        alignItems: "center", justifyContent: "center",
        color: "#fff", fontWeight: 700, fontSize: 16,
      }}>R</div>
      <div>
        <div style={{ fontWeight: 600, fontSize: 15 }}>
          Medical Research RAG
        </div>
        <div style={{ fontSize: 12, color: "#6b7280" }}>
          Ask questions about your papers, get cited answers
        </div>
      </div>
      <div style={{ flex: 1 }} />
        <div style={{ fontSize: 13, color: "#9ca3af" }}>
          {docs.length} paper{docs.length !== 1 ? "s" : ""} indexed
        </div>
      </div>

      <div style={{ display: "flex", height: "calc(100vh - 65px)" }}>
        {/* Sidebar */}
        <div style={{
          width: 280, background: "#fff",
          borderRight: "1px solid #e5e7eb",
          padding: 16, display: "flex",
          flexDirection: "column", gap: 10,
          overflowY: "auto",
        }}>
          <div
            onClick={() => fileRef.current.click()}
            onDrop={e => {
              e.preventDefault()
              const files = [...e.dataTransfer.files].filter(f => f.name.endsWith(".pdf"))
              if (files.length) handleUpload(files)
            }}
            onDragOver={e => e.preventDefault()}
            style={{
              border: "2px dashed #e2e8f0", borderRadius: 10,
              padding: "20px 12px", textAlign: "center",
              cursor: "pointer", color: "#9ca3af", fontSize: 13,
              transition: "border-color .15s, color .15s",
            }}
            onMouseEnter={e => {
              e.currentTarget.style.borderColor = "#1d4ed8"
              e.currentTarget.style.color = "#1d4ed8"
            }}
            onMouseLeave={e => {
              e.currentTarget.style.borderColor = "#e2e8f0"
              e.currentTarget.style.color = "#9ca3af"
            }}
          >
            {uploading ? (
              <span style={{ color: "#1d4ed8" }}>Indexing…</span>
            ) : (
              <>
                <div style={{ fontSize: 24, marginBottom: 4 }}>+</div>
                <div style={{ fontWeight: 500 }}>Upload PDF</div>
                <div style={{ fontSize: 11, marginTop: 2 }}>
                  or drag and drop
                </div>
              </>
            )}
          </div>
          <input
            ref={fileRef} type="file" accept=".pdf"
            multiple style={{ display: "none" }}
            onChange={e => handleUpload([...e.target.files])}
          />

          {docs.length > 0 && (
            <>
              <div style={{
                fontSize: 10, color: "#9ca3af",
                letterSpacing: "0.08em", fontWeight: 600,
                marginTop: 6,
              }}>
                PAPERS — click to filter
              </div>
              {docs.map(doc => (
                <div
                  key={doc.doc_id}
                  onClick={() => toggleDoc(doc.doc_id)}
                  style={{
                    display: "flex", alignItems: "center", gap: 8,
                    padding: "8px 10px", borderRadius: 8, cursor: "pointer",
                    border: `1px solid ${selectedDocs.includes(doc.doc_id) ? "#1d4ed8" : "#e5e7eb"}`,
                    background: selectedDocs.includes(doc.doc_id) ? "#eff6ff" : "#fff",
                    fontSize: 13, transition: "all .12s",
                  }}
                >
                  <span style={{
                    color: selectedDocs.includes(doc.doc_id) ? "#1d4ed8" : "#d1d5db",
                    fontSize: 10,
                  }}>
                    {selectedDocs.includes(doc.doc_id) ? "●" : "○"}
                  </span>
                  <span style={{
                    flex: 1, overflow: "hidden",
                    textOverflow: "ellipsis", whiteSpace: "nowrap",
                    color: "#374151",
                  }}>
                    {doc.filename}
                  </span>
                  <span style={{ fontSize: 11, color: "#9ca3af" }}>
                    {doc.chunk_count}c
                  </span>
                  <button
                    onClick={e => {
                      e.stopPropagation()
                      deleteDoc(doc.doc_id).then(() =>
                        setDocs(d => d.filter(x => x.doc_id !== doc.doc_id))
                      )
                      setSelectedDocs(s => s.filter(x => x !== doc.doc_id))
                    }}
                    style={{
                      background: "none", border: "none",
                      color: "#d1d5db", cursor: "pointer",
                      fontSize: 16, lineHeight: 1, padding: "0 2px",
                    }}
                    title="Remove"
                  >×</button>
                </div>
              ))}
              {selectedDocs.length > 0 && (
                <button
                  onClick={() => setSelectedDocs([])}
                  style={{
                    background: "none", border: "none",
                    color: "#6b7280", cursor: "pointer",
                    fontSize: 12, textAlign: "left", padding: 0,
                  }}
                >
                  clear filter ({selectedDocs.length} selected)
                </button>
              )}
            </>
          )}
        </div>

        {/* Main */}
        <div style={{
          flex: 1, padding: "28px 40px",
          overflowY: "auto", maxWidth: "calc(100vw - 280px)",
        }}>
          {/* Input */}
          <div style={{ position: "relative" }}>
            <textarea
              value={question}
              onChange={e => setQuestion(e.target.value)}
              onKeyDown={e => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault()
                  handleAsk()
                }
              }}
              placeholder={
                docs.length === 0
                  ? "Upload a PDF to get started…"
                  : "Ask a question about your papers… (Enter to send)"
              }
              disabled={docs.length === 0 || loading}
              rows={3}
              style={{
                width: "100%", padding: "14px 56px 14px 16px",
                border: "1px solid #e2e8f0", borderRadius: 12,
                fontSize: 14, fontFamily: "inherit", resize: "none",
                outline: "none", background: "#fff",
                boxSizing: "border-box", lineHeight: 1.6,
                color: "#111827",
              }}
            />
            <button
              onClick={handleAsk}
              disabled={!question.trim() || loading || docs.length === 0}
              style={{
                position: "absolute", right: 10, bottom: 10,
                background: loading ? "#e5e7eb" : "#1d4ed8",
                border: "none", borderRadius: 8,
                padding: "7px 16px", color: "#fff",
                cursor: loading ? "default" : "pointer",
                fontSize: 13, fontWeight: 500,
                transition: "background .15s",
              }}
            >
              {loading ? "thinking…" : "Ask"}
            </button>
          </div>

          {selectedDocs.length > 0 && (
            <div style={{ marginTop: 8, fontSize: 12, color: "#1d4ed8" }}>
              Searching {selectedDocs.length} selected paper{selectedDocs.length !== 1 ? "s" : ""}
            </div>
          )}

          {error && (
            <div style={{
              marginTop: 14, padding: "10px 14px",
              background: "#fef2f2", border: "1px solid #fecaca",
              borderRadius: 8, color: "#dc2626", fontSize: 13,
            }}>
              {error}
            </div>
          )}

          {loading && (
            <div style={{
              marginTop: 32, color: "#9ca3af",
              fontSize: 13, textAlign: "center",
            }}>
              Retrieving chunks · generating answer…
            </div>
          )}

          <Answer data={result} />

          {docs.length === 0 && (
            <div style={{
              marginTop: 80, textAlign: "center",
              color: "#d1d5db", lineHeight: 2,
            }}>
              <div style={{ fontSize: 40, marginBottom: 8 }}>📄</div>
              <div style={{ fontSize: 15, fontWeight: 500 }}>
                Upload a research paper to get started
              </div>
              <div style={{ fontSize: 13 }}>
                Supports any medical PDF from PubMed Central
              </div>
            </div>
          )}
        </div>
      </div>

      <style>{`
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { margin: 0; }
        textarea::placeholder { color: #9ca3af; }
        ::-webkit-scrollbar { width: 4px; }
        ::-webkit-scrollbar-thumb {
          background: #e2e8f0; border-radius: 2px;
        }
      `}</style>
    </div>
  )
}