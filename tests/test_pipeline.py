"""Automated end-to-end verification tests for CampusGPT."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app import create_app
from database import db
from modules.document_processor import (
    process_document,
    process_pdf,
    recursive_split_text,
    sanitize_text,
)
from modules.llm_service import GeminiService, LLMServiceError
from modules.rag_engine import RAGEngine


def _build_minimal_pdf(pages_text: list[str], dest_path: Path) -> Path:
    """Create a valid multi-page PDF without external dependencies."""

    def _pdf_escape(s: str) -> str:
        return s.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")

    objects: list[bytes] = []
    objects.append(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n")
    objects.append(b"")
    font_obj_num = 3
    objects.append(
        b"3 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n"
    )

    page_obj_nums: list[int] = []
    next_obj_num = 4
    for text in pages_text:
        content_obj_num = next_obj_num
        page_obj_num = next_obj_num + 1
        next_obj_num += 2
        page_obj_nums.append(page_obj_num)

        stream_str = f"BT\n/F1 12 Tf\n72 720 Td\n({_pdf_escape(text)}) Tj\nET\n"
        stream_bytes = stream_str.encode("latin-1", errors="replace")
        objects.append(
            f"{content_obj_num} 0 obj\n<< /Length {len(stream_bytes)} >>\nstream\n".encode(
                "ascii"
            )
            + stream_bytes
            + b"\nendstream\nendobj\n"
        )
        objects.append(
            (
                f"{page_obj_num} 0 obj\n"
                f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                f"/Resources << /Font << /F1 {font_obj_num} 0 R >> >> "
                f"/Contents {content_obj_num} 0 R >>\nendobj\n"
            ).encode("ascii")
        )

    kids = " ".join(f"{num} 0 R" for num in page_obj_nums)
    objects[1] = (
        f"2 0 obj\n<< /Type /Pages /Kids [{kids}] /Count {len(page_obj_nums)} >>\nendobj\n"
    ).encode("ascii")

    header = b"%PDF-1.4\n"
    body = bytearray()
    offsets: list[int] = [0]
    current_offset = len(header)
    for obj in objects:
        offsets.append(current_offset)
        body.extend(obj)
        current_offset += len(obj)

    xref_offset = len(header) + len(body)
    xref_lines = [f"xref\n0 {len(objects) + 1}\n", "0000000000 65535 f \n"]
    for off in offsets[1:]:
        xref_lines.append(f"{off:010d} 00000 n \n")

    trailer = (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref_offset}\n%%EOF\n"
    )
    dest_path.write_bytes(
        header + bytes(body) + "".join(xref_lines).encode("ascii") + trailer.encode("ascii")
    )
    return dest_path


def test_database_lifecycle_and_metrics(tmp_path):
    db_path = str(tmp_path / "test.db")
    db.init_db(db_path)

    user_id = db.create_user(db_path, "Student1", "student1@univ.edu", "hash123", "Student One")
    assert user_id > 0

    fetched = db.get_user_by_identifier(db_path, "student1@univ.edu")
    assert fetched is not None
    assert fetched["username"] == "student1"

    doc_id = db.create_document(
        db_path,
        user_id=user_id,
        filename="abc_notes.pdf",
        original_filename="notes.pdf",
        file_path="/tmp/abc_notes.pdf",
        file_size=2048,
        page_count=3,
        chunk_count=6,
        course_name="Operating Systems",
        file_type="pdf",
    )
    docs = db.get_user_documents(db_path, user_id)
    assert len(docs) == 1
    assert docs[0]["course_name"] == "Operating Systems"

    assert db.update_document_course(db_path, doc_id, user_id, "OS-301")
    assert db.update_document_summary(db_path, doc_id, user_id, "## Summary")
    updated = db.get_document_by_id(db_path, doc_id, user_id)
    assert updated["summary"] == "## Summary"
    assert updated["course_name"] == "OS-301"

    art_id = db.create_study_artifact(
        db_path,
        user_id=user_id,
        document_id=doc_id,
        artifact_type="quiz",
        title="OS Quiz",
        raw_markdown="## Quiz",
        content_json="[]",
        total_questions=5,
        engine_used="gemini-3.8-flash",
    )
    assert db.update_quiz_score(db_path, art_id, user_id, 4, 5) is True
    artifacts = db.get_study_artifacts(db_path, user_id)
    assert len(artifacts) == 1
    assert artifacts[0]["score"] == 4

    db.log_query_metric(db_path, user_id, doc_id, "chat", "gemini-3.8-flash", 120, 0.85, 4)
    metrics = db.get_user_metrics_summary(db_path, user_id)
    assert metrics["total_queries"] == 1
    assert metrics["avg_quiz_pct"] == 80.0

    msg_id = db.add_chat_message(db_path, user_id, doc_id, "user", "What is RAG?")
    assert msg_id > 0
    assert len(db.get_chat_history(db_path, user_id, document_id=doc_id)) == 1
    assert db.clear_chat_history(db_path, user_id, document_id=doc_id) == 1
    assert db.delete_study_artifact(db_path, art_id, user_id) is True
    assert db.delete_document(db_path, doc_id, user_id) is True


def test_sanitization_and_multiformat_chunking(tmp_path):
    dirty = "Quantum\x00 Mechanics \ud800and\udfff Linear Algebra\x07 α+β"
    clean = sanitize_text(dirty)
    assert "\x00" not in clean
    assert "\ud800" not in clean
    assert "\udfff" not in clean
    assert "Quantum" in clean and "α+β" in clean
    clean.encode("utf-8")

    long_text = "\n\n".join(
        [f"Paragraph {i}: Neural networks learn layered representations via backpropagation." for i in range(15)]
    )
    chunks = recursive_split_text(long_text, chunk_size=220, chunk_overlap=40)
    assert len(chunks) > 3 and all(len(c) <= 220 for c in chunks)

    # PDF
    pdf_path = _build_minimal_pdf(
        [
            "Page one covers Binary Search Trees and AVL rotations in data structures.",
            "Page two covers Dijkstra shortest path algorithm and priority queues.",
        ],
        tmp_path / "algorithms.pdf",
    )
    pdf_res = process_pdf(str(pdf_path), chunk_size=500, chunk_overlap=50)
    assert pdf_res["page_count"] == 2
    assert len(pdf_res["chunks"]) == 2

    # Scanned PDF OCR callback fallback
    scanned_pdf = _build_minimal_pdf([""], tmp_path / "scanned.pdf")
    ocr_res = process_document(
        str(scanned_pdf),
        ocr_callback=lambda _p: [{"page": 1, "text": "Scanned lecture note about Turing machines."}],
    )
    assert "Turing machines" in ocr_res["chunks"][0]["text"]

    # DOCX
    from docx import Document as DocxDocument

    docx_path = tmp_path / "lecture.docx"
    d = DocxDocument()
    d.add_paragraph("Database normalization eliminates update anomalies using 1NF, 2NF, 3NF, and BCNF.")
    d.save(str(docx_path))
    docx_res = process_document(str(docx_path))
    assert docx_res["file_type"] == "docx"
    assert "BCNF" in docx_res["chunks"][0]["text"]

    # PPTX
    from pptx import Presentation

    pptx_path = tmp_path / "slides.pptx"
    prs = Presentation()
    slide = prs.slides.add_slide(prs.slide_layouts[1])
    slide.shapes.title.text = "Virtual Memory & Paging"
    slide.placeholders[1].text = "TLB caches page table translations for fast address lookup."
    prs.save(str(pptx_path))
    pptx_res = process_document(str(pptx_path))
    assert pptx_res["file_type"] == "pptx"
    assert "TLB" in pptx_res["chunks"][0]["text"]

    # TXT
    txt_path = tmp_path / "notes.txt"
    txt_path.write_text("Bayes theorem computes posterior probability from prior and likelihood.", encoding="utf-8")
    txt_res = process_document(str(txt_path))
    assert txt_res["file_type"] == "txt"
    assert "Bayes" in txt_res["chunks"][0]["text"]

    bad_pdf = tmp_path / "corrupt.pdf"
    bad_pdf.write_bytes(b"not a real pdf")
    with pytest.raises(ValueError):
        process_pdf(str(bad_pdf))


def test_hybrid_faiss_bm25_rag_build_search_delete(tmp_path):
    engine = RAGEngine(
        vector_store_folder=str(tmp_path / "vectors"),
        model_name="all-MiniLM-L6-v2",
        model_cache_folder=str(PROJECT_ROOT / "data" / "model_cache"),
    )

    chunks = [
        {"chunk_index": 0, "page": 1, "text": "Photosynthesis converts sunlight into chemical energy in chloroplasts."},
        {"chunk_index": 1, "page": 2, "text": "TCP uses a three-way handshake (SYN, SYN-ACK, ACK) and RFC 793 defines it."},
        {"chunk_index": 2, "page": 3, "text": "Eigenvalues satisfy the characteristic equation det(A - lambda I) = 0."},
    ]
    count = engine.build_index(doc_id=101, chunks=chunks, doc_name="cs_bio_math.pdf")
    assert count == 3

    results = engine.search("RFC 793 three-way handshake", doc_ids=[101], top_k=2)
    assert len(results) == 2
    assert results[0]["page"] == 2
    assert "dense_score" in results[0] and "bm25_score" in results[0] and "rrf_score" in results[0]

    sampled = engine.get_document_chunks(doc_id=101, max_chunks=2)
    assert len(sampled) == 2

    engine.delete_index(doc_id=101)
    with pytest.raises(FileNotFoundError):
        engine.load_index(101)


def test_llm_service_streaming_quiz_and_fallback():
    unconfigured = GeminiService(api_key="your_api_key_here")
    assert unconfigured.is_configured() is False
    with pytest.raises(LLMServiceError) as exc_info:
        unconfigured.answer_question("What is RAG?", [{"page": 1, "text": "RAG info", "doc_name": "a.pdf"}])
    assert exc_info.value.code == "MISSING_API_KEY"

    svc = GeminiService(api_key="fake-test-key", model_name="gemini-3.8-flash")
    mock_client = MagicMock()
    mock_client.models.list.return_value = []
    mock_client.models.generate_content.return_value = SimpleNamespace(
        text=json.dumps(
            [
                {
                    "question": "What does RAG stand for?",
                    "options": [
                        "Retrieval-Augmented Generation",
                        "Random Access Graph",
                        "Recursive Auto Gradient",
                        "Relational API Gateway",
                    ],
                    "correct_index": 0,
                    "explanation": "RAG combines retrieval with generation.",
                    "page": 1,
                }
            ]
        )
    )
    mock_client.models.generate_content_stream.return_value = [
        SimpleNamespace(text="RAG retrieves "),
        SimpleNamespace(text="grounded context (Page 1)."),
    ]
    svc._client = mock_client

    chunks = [{"page": 1, "text": "RAG combines retrieval with generation.", "doc_name": "rag.pdf"}]
    questions, md = svc.generate_quiz_structured("rag.pdf", chunks, num_questions=1)
    assert len(questions) == 1
    assert questions[0]["correct_index"] == 0
    assert "Answer Key" in md

    streamed = "".join(svc.stream_answer_question("What is RAG?", chunks))
    assert "RAG retrieves grounded context" in streamed

    # Verify automatic fallback to local extractive RAG when ALL cloud models return 503
    offline_svc = GeminiService(
        api_key="fake-test-key",
        model_name="gemini-3.8-flash",
        fallback_models=("gemini-3.5-flash-lite",),
        retry_delay=0.0,
    )
    offline_client = MagicMock()
    offline_client.models.list.return_value = []
    offline_client.models.generate_content.side_effect = RuntimeError("503 UNAVAILABLE: high demand")
    offline_client.models.generate_content_stream.side_effect = RuntimeError("503 UNAVAILABLE: high demand")
    offline_svc._client = offline_client

    assert "retrieval" in offline_svc.answer_question("What does RAG combine?", chunks).lower()
    assert offline_svc.last_model_used == "local-extractive-rag"
    qs, _ = offline_svc.generate_quiz_structured("rag.pdf", chunks, num_questions=1)
    assert "Study Summary" in offline_svc.summarize_document("rag.pdf", chunks)
    assert "Viva Voce" in offline_svc.generate_viva("rag.pdf", chunks, num_questions=1)


def test_flask_routes_end_to_end(tmp_path):
    app = create_app(
        {
            "TESTING": True,
            "SECRET_KEY": "test-secret",
            "DATABASE_PATH": str(tmp_path / "app.db"),
            "UPLOAD_FOLDER": str(tmp_path / "uploads"),
            "VECTOR_STORE_FOLDER": str(tmp_path / "vectors"),
            "MODEL_CACHE_FOLDER": str(PROJECT_ROOT / "data" / "model_cache"),
            "GEMINI_API_KEY": "test-key",
        }
    )

    mock_client = MagicMock()
    mock_client.models.list.return_value = []
    mock_client.models.generate_content.return_value = SimpleNamespace(
        text="### Mocked Academic Response\nGrounded answer citing (Page 1)."
    )
    mock_client.models.generate_content_stream.return_value = [
        SimpleNamespace(text="Streamed answer "),
        SimpleNamespace(text="citing (Page 1)."),
    ]
    app.llm_service._client = mock_client

    client = app.test_client()
    assert client.get("/").status_code == 302
    assert client.get("/dashboard").status_code == 302
    assert client.get("/api/documents").status_code == 401
    assert client.get("/api/health").status_code == 200

    resp = client.post(
        "/register",
        data={
            "full_name": "Navraj Test",
            "username": "navraj_test",
            "email": "navraj@university.edu",
            "password": "password123",
        },
        follow_redirects=False,
    )
    assert resp.status_code == 302
    assert client.get("/dashboard").status_code == 200

    # Reject unsupported file extension
    bad_upload = client.post(
        "/api/documents/upload",
        data={"file": (open(__file__, "rb"), "script.py")},
        content_type="multipart/form-data",
    )
    assert bad_upload.status_code == 400

    # Upload PDF with course tag
    pdf_path = _build_minimal_pdf(
        [
            "Operating systems manage CPU scheduling, virtual memory paging, and deadlocks.",
            "Banker's algorithm checks safe sequences to avoid resource deadlocks.",
        ],
        tmp_path / "os_notes.pdf",
    )
    with open(pdf_path, "rb") as fh:
        upload_resp = client.post(
            "/api/documents/upload",
            data={"file": (fh, "os_notes.pdf"), "course_name": "Operating Systems"},
            content_type="multipart/form-data",
        )
    assert upload_resp.status_code == 201
    doc = upload_resp.get_json()["document"]
    doc_id = doc["id"]
    assert doc["course_name"] == "Operating Systems"

    # Update course tag + view PDF inline
    patch_resp = client.patch(f"/api/documents/{doc_id}", json={"course_name": "CS-302"})
    assert patch_resp.status_code == 200
    assert patch_resp.get_json()["document"]["course_name"] == "CS-302"

    view_resp = client.get(f"/api/documents/{doc_id}/view")
    assert view_resp.status_code == 200
    assert "application/pdf" in view_resp.content_type

    # Standard chat + SSE streaming chat
    chat_resp = client.post(
        "/api/chat",
        json={"question": "How are deadlocks avoided?", "document_ids": [doc_id]},
    )
    assert chat_resp.status_code == 200
    assert "Mocked Academic Response" in chat_resp.get_json()["answer"]

    stream_resp = client.post(
        "/api/chat/stream",
        json={"question": "How are deadlocks avoided?", "document_ids": [doc_id]},
    )
    assert stream_resp.status_code == 200
    stream_body = stream_resp.get_data(as_text=True)
    assert "Streamed answer" in stream_body

    # Summarize, Quiz (local mode for structured MCQs), Score save, Viva
    assert client.post("/api/summarize", json={"document_id": doc_id}).status_code == 200
    quiz_resp = client.post(
        "/api/quiz",
        json={"document_id": doc_id, "num_questions": 2, "mode": "local"},
    )
    assert quiz_resp.status_code == 200
    quiz_data = quiz_resp.get_json()
    assert len(quiz_data["questions"]) >= 1
    assert client.post(
        f"/api/quiz/{quiz_data['artifact_id']}/score",
        json={"score": 1, "total_questions": len(quiz_data["questions"])},
    ).status_code == 200

    assert client.post("/api/viva", json={"document_id": doc_id, "num_questions": 2}).status_code == 200

    # Artifacts & Metrics endpoints
    arts = client.get("/api/artifacts").get_json()["artifacts"]
    assert len(arts) >= 3
    metrics = client.get("/api/metrics").get_json()
    assert metrics["total_queries"] >= 4
    assert metrics["quiz_attempts"] == 1

    # Delete document
    assert client.delete(f"/api/documents/{doc_id}").status_code == 200
    assert client.get("/api/documents").get_json()["documents"] == []
