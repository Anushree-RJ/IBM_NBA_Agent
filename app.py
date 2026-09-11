"""
AccrediAI – NBA Accreditation Assistant
Backend: Production-grade Flask application.

Technology Stack
----------------
  IBM Granite (ibm/granite-4-h-small) via IBM watsonx.ai  →  LLM
  FAISS + sentence-transformers/all-MiniLM-L6-v2           →  Vector store + embeddings
  LangChain                                                 →  Document loading / chunking
  Flask                                                     →  Web framework

Endpoints
---------
  GET  /                             → dashboard
  GET  /api/stats                    → dashboard statistics
  GET  /api/documents                → list documents (optional ?category=X)
  GET  /api/documents/categories     → category list + counts
  DELETE /api/documents/<id>         → remove document + rebuild index
  POST /api/upload                   → upload + ingest document
  POST /api/chat                     → general RAG Q&A
  POST /api/sar-assist               → SAR section preparation
  POST /api/copo-map                 → CO-PO mapping assistant
  GET  /api/health                   → service health check
"""

# ---------------------------------------------------------------------------
# Environment — load .env before anything else
# ---------------------------------------------------------------------------
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import json
import logging
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

from flask import Flask, Response, jsonify, render_template, request
from werkzeug.utils import secure_filename

# IBM watsonx AI SDK
from ibm_watsonx_ai import APIClient, Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.metanames import GenChatParamsMetaNames as ChatParams

# Modular RAG pipeline
from rag_pipeline import FAISSVectorStore, RAGPipeline

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Flask application
# ---------------------------------------------------------------------------
app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False   # preserve insertion order in JSON

# ---------------------------------------------------------------------------
# Directory paths
# ---------------------------------------------------------------------------
BASE_DIR          = Path(__file__).parent
UPLOAD_FOLDER     = BASE_DIR / "uploads"
KB_FOLDER         = BASE_DIR / "knowledge_base"
FAISS_INDEX_PATH  = KB_FOLDER / "faiss_index"
DOC_METADATA_PATH = KB_FOLDER / "doc_metadata.json"
STATS_PATH        = KB_FOLDER / "stats.json"

ALLOWED_EXTENSIONS   = {"pdf", "txt", "docx", "xlsx"}
MAX_CONTENT_LENGTH   = 50 * 1024 * 1024   # 50 MB

app.config["UPLOAD_FOLDER"]    = str(UPLOAD_FOLDER)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

# ---------------------------------------------------------------------------
# Document categories
# ---------------------------------------------------------------------------
DOCUMENT_CATEGORIES = [
    "NBA Guidelines",
    "Accreditation Criteria",
    "SAR Document",
    "SAR Format / Template",
    "CO-PO Mapping",
    "Previous Accreditation",
    "Other",
]

# ---------------------------------------------------------------------------
# IBM watsonx credentials  (always from env — never hardcoded)
# ---------------------------------------------------------------------------
WATSONX_URL        = os.environ.get("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")
WATSONX_API_KEY    = os.environ.get("WATSONX_API_KEY", "")
WATSONX_PROJECT_ID = os.environ.get("WATSONX_PROJECT_ID", "")
GRANITE_MODEL_ID   = os.environ.get("GRANITE_MODEL_ID", "ibm/granite-4-h-small")

# ---------------------------------------------------------------------------
# RAG pipeline  (FAISS backend — swap FAISSVectorStore for any VectorStoreBase)
# ---------------------------------------------------------------------------
_vector_store = FAISSVectorStore(index_path=FAISS_INDEX_PATH)
rag           = RAGPipeline(vector_store=_vector_store)

# ---------------------------------------------------------------------------
# Security headers (applied to every response)
# ---------------------------------------------------------------------------
@app.after_request
def add_security_headers(response: Response) -> Response:
    response.headers["X-Content-Type-Options"]  = "nosniff"
    response.headers["X-Frame-Options"]         = "DENY"
    response.headers["Referrer-Policy"]         = "strict-origin-when-cross-origin"
    return response

# ---------------------------------------------------------------------------
# Agent / model prompts
# ---------------------------------------------------------------------------
_AGENT_CORE = """\
You are AccrediAI, a specialised AI assistant that supports faculty members \
with NBA (National Board of Accreditation, India) accreditation processes.

## Absolute Rules
1. Answer ONLY using the text provided in the [Context] section below.
2. Never invent, extrapolate, or assume NBA criteria, policies, statistics, \
   thresholds, or institutional data.
3. If the context does not contain enough information, respond with exactly:
   "I couldn't find this information in the uploaded NBA documents. \
Please upload the relevant accreditation document to receive an accurate answer."
4. Cite sources inline as: (Source: <filename>, Page <N>).
5. Respond in clear, professional English suitable for an academic institution.\
"""

AGENT_INSTRUCTIONS = _AGENT_CORE + """

## Scope
- NBA accreditation criteria, eligibility, evaluation parameters.
- SAR section preparation based solely on uploaded documents.
- CO-PO mapping guidance based solely on uploaded documentation."""

SAR_INSTRUCTIONS = _AGENT_CORE + """

## Mode: SAR Preparation
Structure your response into EXACTLY these five numbered sections:

1. **Relevant Criterion** – NBA clause or criterion that applies (from context only).
2. **Requirements Identified** – Specific requirements stated in uploaded documents.
3. **Suggested Structure** – How the SAR section should be organised.
4. **Information Required from Institution** – Data the faculty must supply \
(do NOT invent any figures or institutional data).
5. **Source Documents** – Every document and page used.

End your response with this exact disclaimer on a new line:
⚠️ Disclaimer: AI-generated content based solely on uploaded documents. \
Review and verify all content before official SAR submission."""

COPO_INSTRUCTIONS = _AGENT_CORE + """

## Mode: CO-PO Mapping
Structure your response into EXACTLY these six numbered sections:

1. **Understanding** – Restate the COs and POs as supplied.
2. **Relevant Guidance** – Mapping methodology from uploaded NBA documents.
3. **Suggested Mapping Table** – CO × PO table with levels (H/M/L or 3/2/1/0) \
and one-line reasoning per non-zero cell.
4. **Reasoning** – Justify correlations citing uploaded document context.
5. **Faculty Review Notes** – Aspects the faculty must validate independently.
6. **Source Documents** – Every document and page used.

End your response with this exact disclaimer on a new line:
⚠️ Disclaimer: AI-suggested mapping based on uploaded documents only. \
Not an officially approved NBA mapping. Faculty must review and approve."""

# ---------------------------------------------------------------------------
# IBM Granite inference  (single entry-point for all three modes)
# ---------------------------------------------------------------------------

def _call_granite(
    system_prompt: str,
    context: str,
    user_query: str,
    max_tokens: int = 1500,
) -> str:
    """
    Compose a grounded prompt and call IBM Granite via the chat endpoint.
    URL: POST /ml/v1/text/chat?version=2023-05-29
    Returns the assistant reply string.
    """
    if not WATSONX_API_KEY or not WATSONX_PROJECT_ID:
        return (
            "⚠️ IBM watsonx credentials are not configured. "
            "Set WATSONX_API_KEY and WATSONX_PROJECT_ID in your .env file "
            "and restart the server."
        )

    ctx_block = context.strip() or "[No relevant context found in the uploaded documents.]"

    # System message: agent instructions + retrieved context
    system_content = (
        f"{system_prompt}\n\n"
        f"## Context (from uploaded NBA documents)\n\n{ctx_block}"
    )

    try:
        credentials = Credentials(url=WATSONX_URL, api_key=WATSONX_API_KEY)
        client = APIClient(credentials=credentials, project_id=WATSONX_PROJECT_ID)
        model  = ModelInference(
            model_id=GRANITE_MODEL_ID,
            api_client=client,
            project_id=WATSONX_PROJECT_ID,
            params={
                ChatParams.MAX_TOKENS:   max_tokens,
                ChatParams.TEMPERATURE:  0.1,
                ChatParams.TOP_P:        0.9,
            },
        )
        # Use the chat interface (POST /ml/v1/text/chat)
        messages = [
            {"role": "system",    "content": system_content},
            {"role": "user",      "content": user_query},
        ]
        response = model.chat(messages=messages)
        # Extract text from IBM chat response structure
        choices = response.get("choices", [])
        if choices:
            return choices[0].get("message", {}).get("content", "").strip()
        return "I couldn't generate a response. Please try again."

    except Exception as exc:
        logger.exception("IBM Granite call failed: %s", exc)
        # Fallback: try generate_text with a combined prompt
        try:
            credentials2 = Credentials(url=WATSONX_URL, api_key=WATSONX_API_KEY)
            client2 = APIClient(credentials=credentials2, project_id=WATSONX_PROJECT_ID)
            model2  = ModelInference(
                model_id=GRANITE_MODEL_ID,
                api_client=client2,
                project_id=WATSONX_PROJECT_ID,
                params={
                    GenParams.MAX_NEW_TOKENS:    max_tokens,
                    GenParams.TEMPERATURE:       0.1,
                    GenParams.TOP_P:             0.9,
                    GenParams.REPETITION_PENALTY:1.1,
                },
            )
            prompt = (
                f"{system_content}\n\n"
                f"## Request\n{user_query}\n\n"
                f"## Response"
            )
            return model2.generate_text(prompt=prompt).strip()
        except Exception as exc2:
            logger.exception("IBM Granite fallback also failed: %s", exc2)
            return (
                f"❌ Error communicating with IBM Granite: {exc2}. "
                "Please check your API key, project ID, and network connectivity."
            )

# ---------------------------------------------------------------------------
# RAG helpers
# ---------------------------------------------------------------------------

def _retrieve(query: str, top_k: int = 6) -> tuple[str, list[dict]]:
    """Return (context_str, sources_list).  Empty values if store not ready."""
    if not _vector_store.is_ready():
        return "", []
    return rag.retrieve(query, top_k=top_k)


def _no_docs_msg() -> str:
    return (
        "No documents have been uploaded yet. "
        "Please upload NBA accreditation documents in the Knowledge Base section first."
    )

# ---------------------------------------------------------------------------
# Metadata helpers
# ---------------------------------------------------------------------------

def load_doc_metadata() -> list[dict]:
    if DOC_METADATA_PATH.exists():
        try:
            with open(DOC_METADATA_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.error("Could not read doc metadata: %s", e)
    return []


def save_doc_metadata(metadata: list[dict]) -> None:
    KB_FOLDER.mkdir(parents=True, exist_ok=True)
    with open(DOC_METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

def load_stats() -> dict:
    if STATS_PATH.exists():
        try:
            with open(STATS_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    return {"total_queries": 0, "sar_queries": 0, "copo_queries": 0, "recent_queries": []}


def save_stats(stats: dict) -> None:
    KB_FOLDER.mkdir(parents=True, exist_ok=True)
    with open(STATS_PATH, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)


def record_query(question: str, query_type: str = "chat") -> None:
    """Increment query counter and store the latest 10 queries."""
    stats = load_stats()
    stats["total_queries"] = stats.get("total_queries", 0) + 1
    if query_type == "sar":
        stats["sar_queries"] = stats.get("sar_queries", 0) + 1
    elif query_type == "copo":
        stats["copo_queries"] = stats.get("copo_queries", 0) + 1

    recent = stats.get("recent_queries", [])
    recent.insert(0, {
        "question": question[:200],
        "type":     query_type,
        "time":     datetime.now(timezone.utc).isoformat(),
    })
    stats["recent_queries"] = recent[:10]   # keep only latest 10
    save_stats(stats)

# ---------------------------------------------------------------------------
# Route — page
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")

# ---------------------------------------------------------------------------
# Route — dashboard statistics
# ---------------------------------------------------------------------------

@app.route("/api/stats", methods=["GET"])
def get_stats():
    """Dashboard endpoint: document counts, query stats, KB status."""
    docs   = load_doc_metadata()
    stats  = load_stats()
    # All persisted docs are indexed (status field may be absent in older records)
    indexed = sum(1 for d in docs if d.get("status", "indexed") != "failed")
    return jsonify({
        "total_documents":      len(docs),
        "indexed_documents":    indexed,
        "total_queries":        stats.get("total_queries", 0),
        "sar_queries":          stats.get("sar_queries", 0),
        "copo_queries":         stats.get("copo_queries", 0),
        "knowledge_base_ready": _vector_store.is_ready(),
        "model":                GRANITE_MODEL_ID,
        "embed_model":          "sentence-transformers/all-MiniLM-L6-v2",
        "vector_store":         "FAISS",
        "recent_queries":       stats.get("recent_queries", []),
        "categories":           DOCUMENT_CATEGORIES,
    })

# ---------------------------------------------------------------------------
# Routes — document management
# ---------------------------------------------------------------------------

@app.route("/api/upload", methods=["POST"])
def upload_document():
    """Upload + ingest a document.  Accepts optional form field `category`."""
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "No file selected"}), 400

    if not allowed_file(file.filename):
        return jsonify({
            "error": (
                f"Unsupported file type. "
                f"Allowed types: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
            )
        }), 400

    category = request.form.get("category", "Other").strip()
    if category not in DOCUMENT_CATEGORIES:
        category = "Other"

    filename     = secure_filename(file.filename)
    if not filename:
        return jsonify({"error": "Invalid filename"}), 400

    file_ext     = filename.rsplit(".", 1)[-1].upper() if "." in filename else "FILE"
    unique_path  = str(UPLOAD_FOLDER / f"{uuid.uuid4().hex}_{filename}")
    UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

    file.save(unique_path)
    logger.info("Saved upload: %s", unique_path)

    try:
        doc_meta                = rag.ingest(unique_path, filename)
        doc_meta["category"]    = category
        doc_meta["file_type"]   = file_ext
        doc_meta["status"]      = "indexed"

        all_meta = load_doc_metadata()
        all_meta.append(doc_meta)
        save_doc_metadata(all_meta)

        logger.info("Indexed: %s  chunks=%d", filename, doc_meta.get("chunks", 0))
        return jsonify({
            "message":  f"'{filename}' uploaded and indexed successfully.",
            "document": doc_meta,
        })

    except Exception as exc:
        logger.exception("Ingestion error for %s: %s", filename, exc)
        if os.path.exists(unique_path):
            os.remove(unique_path)

        # Give a clear, actionable message for the common scanned-PDF case
        err_str = str(exc)
        if "no text chunks" in err_str or "no text" in err_str.lower():
            return jsonify({
                "error": (
                    f"'{filename}' could not be indexed because no text was found in the document. "
                    "This usually means the PDF is a scanned image. "
                    "Please use Adobe Acrobat (Tools → Enhance Scans → Make Searchable) "
                    "or any online OCR tool (e.g. ilovepdf.com) to convert it to a "
                    "text-searchable PDF, then upload again."
                )
            }), 422

        return jsonify({"error": f"Failed to process document: {exc}"}), 500


@app.route("/api/documents", methods=["GET"])
def list_documents():
    """List all documents.  Optional query param: ?category=<name>."""
    docs     = load_doc_metadata()
    category = request.args.get("category", "").strip()
    if category:
        docs = [d for d in docs if d.get("category") == category]
    return jsonify(docs)


@app.route("/api/documents/categories", methods=["GET"])
def list_categories():
    """Return categories with per-category document counts."""
    docs   = load_doc_metadata()
    counts = {c: 0 for c in DOCUMENT_CATEGORIES}
    for d in docs:
        cat = d.get("category", "Other")
        counts[cat] = counts.get(cat, 0) + 1
    return jsonify({"categories": DOCUMENT_CATEGORIES, "counts": counts})


@app.route("/api/documents/<doc_id>", methods=["DELETE"])
def delete_document(doc_id: str):
    """Delete a document by ID and rebuild the vector index."""
    all_meta = load_doc_metadata()
    target   = next((d for d in all_meta if d["id"] == doc_id), None)
    if not target:
        return jsonify({"error": "Document not found"}), 404

    all_meta = [d for d in all_meta if d["id"] != doc_id]
    save_doc_metadata(all_meta)

    fp = target.get("file_path", "")
    if fp and os.path.exists(fp):
        os.remove(fp)

    rag.delete_document(all_meta)
    logger.info("Deleted document: %s", target["filename"])
    return jsonify({"message": f"Document '{target['filename']}' removed successfully."})

# ---------------------------------------------------------------------------
# Route — general chat
# ---------------------------------------------------------------------------

@app.route("/api/chat", methods=["POST"])
def chat():
    """General RAG Q&A.  POST { question, top_k? }"""
    data     = request.get_json(silent=True) or {}
    question = data.get("question", "").strip()
    if not question:
        return jsonify({"error": "No question provided"}), 400
    if len(question) > 2000:
        return jsonify({"error": "Question exceeds 2000 characters"}), 400

    top_k = min(int(data.get("top_k", 5)), 10)   # cap at 10
    logger.info("Chat query: %.80s", question)
    record_query(question, "chat")

    if not _vector_store.is_ready():
        return jsonify({
            "question": question,
            "answer":   _no_docs_msg(),
            "sources":  [],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

    try:
        context, sources = _retrieve(question, top_k=top_k)
        if not context:
            answer = (
                "I couldn't find this information in the uploaded NBA documents. "
                "Please upload the relevant accreditation document to receive an accurate answer."
            )
        else:
            answer = _call_granite(AGENT_INSTRUCTIONS, context, question)

        return jsonify({
            "question":  question,
            "answer":    answer,
            "sources":   sources,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
    except Exception as exc:
        logger.exception("Chat error: %s", exc)
        return jsonify({"error": f"Internal error: {exc}"}), 500

# ---------------------------------------------------------------------------
# Route — SAR assistance
# ---------------------------------------------------------------------------

@app.route("/api/sar-assist", methods=["POST"])
def sar_assist():
    """SAR Preparation.  POST { topic, top_k? }"""
    data  = request.get_json(silent=True) or {}
    topic = data.get("topic", "").strip()
    if not topic:
        return jsonify({"error": "No topic provided"}), 400
    if len(topic) > 1000:
        return jsonify({"error": "Topic exceeds 1000 characters"}), 400

    top_k = min(int(data.get("top_k", 6)), 10)
    logger.info("SAR assist: %.80s", topic)
    record_query(topic, "sar")

    if not _vector_store.is_ready():
        return jsonify({
            "topic":     topic,
            "response":  _no_docs_msg(),
            "sources":   [],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

    try:
        retrieval_query  = f"SAR {topic} NBA accreditation criteria requirements"
        context, sources = _retrieve(retrieval_query, top_k=top_k)
        user_query       = (
            f"Help me prepare the SAR section related to: {topic}\n\n"
            "Please structure your response using the five required sections."
        )
        response = _call_granite(SAR_INSTRUCTIONS, context, user_query, max_tokens=1800)
        return jsonify({
            "topic":     topic,
            "response":  response,
            "sources":   sources,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
    except Exception as exc:
        logger.exception("SAR assist error: %s", exc)
        return jsonify({"error": f"Internal error: {exc}"}), 500

# ---------------------------------------------------------------------------
# Route — CO-PO mapping
# ---------------------------------------------------------------------------

@app.route("/api/copo-map", methods=["POST"])
def copo_map():
    """CO-PO Mapping.  POST { course_outcomes[], program_outcomes[], top_k? }"""
    data = request.get_json(silent=True) or {}
    cos  = data.get("course_outcomes", [])
    pos  = data.get("program_outcomes", [])

    if not cos or not pos:
        return jsonify({"error": "Both course_outcomes and program_outcomes are required"}), 400
    if not isinstance(cos, list) or not isinstance(pos, list):
        return jsonify({"error": "course_outcomes and program_outcomes must be arrays"}), 400

    cos   = [str(c).strip() for c in cos[:12] if str(c).strip()]
    pos   = [str(p).strip() for p in pos[:12] if str(p).strip()]
    top_k = min(int(data.get("top_k", 6)), 10)

    if not cos or not pos:
        return jsonify({"error": "No valid COs or POs after sanitisation"}), 400

    logger.info("CO-PO map: %d COs × %d POs", len(cos), len(pos))
    record_query(f"CO-PO mapping ({len(cos)} COs × {len(pos)} POs)", "copo")

    if not _vector_store.is_ready():
        return jsonify({
            "response":  _no_docs_msg(),
            "sources":   [],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

    try:
        retrieval_query  = "CO-PO mapping methodology correlation levels NBA programme outcomes"
        context, sources = _retrieve(retrieval_query, top_k=top_k)
        co_block         = "\n".join(f"  {c}" for c in cos)
        po_block         = "\n".join(f"  {p}" for p in pos)
        user_query       = (
            f"Course Outcomes (COs):\n{co_block}\n\n"
            f"Programme Outcomes (POs):\n{po_block}\n\n"
            "Generate a CO-PO mapping using the six required sections."
        )
        response = _call_granite(COPO_INSTRUCTIONS, context, user_query, max_tokens=2000)
        return jsonify({
            "course_outcomes":  cos,
            "program_outcomes": pos,
            "response":         response,
            "sources":          sources,
            "timestamp":        datetime.now(timezone.utc).isoformat(),
        })
    except Exception as exc:
        logger.exception("CO-PO map error: %s", exc)
        return jsonify({"error": f"Internal error: {exc}"}), 500

# ---------------------------------------------------------------------------
# Route — health check
# ---------------------------------------------------------------------------

@app.route("/api/health", methods=["GET"])
def health():
    doc_count = len(load_doc_metadata())
    return jsonify({
        "status":               "ok",
        "knowledge_base_ready": _vector_store.is_ready(),
        "document_count":       doc_count,
        "model":                GRANITE_MODEL_ID,
        "timestamp":            datetime.now(timezone.utc).isoformat(),
    })

# ---------------------------------------------------------------------------
# Error handlers
# ---------------------------------------------------------------------------

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify({"error": "Method not allowed"}), 405

@app.errorhandler(413)
def request_entity_too_large(e):
    return jsonify({"error": "File too large. Maximum allowed size is 50 MB."}), 413

@app.errorhandler(500)
def internal_error(e):
    logger.exception("Unhandled 500: %s", e)
    return jsonify({"error": "An unexpected error occurred. Please try again."}), 500

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
    KB_FOLDER.mkdir(parents=True, exist_ok=True)
    debug_mode = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    app.run(debug=debug_mode, host="0.0.0.0", port=5000)
