"""Main Flask application and REST/SSE API routes for CampusGPT."""

import io
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any, Optional

from flask import (
    Flask,
    Response,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    send_file,
    session,
    stream_with_context,
    url_for,
)
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.utils import secure_filename

from config import Config
from database.db import (
    add_chat_message,
    clear_chat_history,
    create_document,
    create_study_artifact,
    create_user,
    delete_document,
    delete_study_artifact,
    get_chat_history,
    get_document_by_id,
    get_study_artifacts,
    get_user_by_identifier,
    get_user_documents,
    get_user_metrics_summary,
    init_db,
    log_query_metric,
    update_document_course,
    update_document_summary,
    update_quiz_score,
)
from modules.auth import (
    hash_password,
    login_required,
    validate_registration_input,
    verify_password,
)
from modules.document_processor import process_document, sanitize_text
from modules.llm_service import GeminiService, LLMServiceError
from modules.rag_engine import RAGEngine


def _allowed_file(filename: str, allowed_extensions: set[str]) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in allowed_extensions


def _dedupe_sources(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[int, int]] = set()
    sources: list[dict[str, Any]] = []
    for chunk in chunks:
        doc_id = int(chunk.get("doc_id", 0))
        page = int(chunk.get("page", 1))
        key = (doc_id, page)
        if key not in seen:
            seen.add(key)
            sources.append(
                {
                    "doc_id": doc_id,
                    "doc_name": chunk.get("doc_name", "Document"),
                    "page": page,
                    "score": chunk.get("score", 0.0),
                }
            )
    return sources


def _prepare_chat_context(
    app: Flask, user_id: int, payload: dict[str, Any]
) -> tuple[Optional[dict[str, Any]], Optional[tuple[Any, int]]]:
    """Validate chat request payload, verify ownership, and retrieve Hybrid RAG chunks."""
    question = sanitize_text(payload.get("question", ""))
    raw_doc_ids = payload.get("document_ids") or []
    primary_doc_id = payload.get("primary_document_id")
    mode = str(payload.get("mode", "auto")).lower()
    if mode not in {"auto", "gemini", "local"}:
        mode = "auto"

    if not question:
        return None, (jsonify({"error": "Question cannot be empty."}), 400)
    if not isinstance(raw_doc_ids, list) or not raw_doc_ids:
        return None, (jsonify({"error": "Please select at least one document."}), 400)

    db_path = app.config["DATABASE_PATH"]
    valid_doc_ids: list[int] = []
    for raw_id in raw_doc_ids:
        try:
            did = int(raw_id)
        except (TypeError, ValueError):
            continue
        if get_document_by_id(db_path, did, user_id):
            valid_doc_ids.append(did)

    if not valid_doc_ids:
        return None, (jsonify({"error": "No valid documents selected."}), 404)

    history_doc_id = (
        int(primary_doc_id)
        if primary_doc_id and int(primary_doc_id) in valid_doc_ids
        else valid_doc_ids[0]
    )

    retrieved = app.rag_engine.search(
        query=question,
        doc_ids=valid_doc_ids,
        top_k=app.config["TOP_K_RESULTS"],
    )

    q_lower = question.lower()
    if any(
        w in q_lower
        for w in ("title", "topic", "subject", "about", "overview", "heading")
    ):
        try:
            _, first_doc_chunks = app.rag_engine.load_index(history_doc_id)
            if first_doc_chunks:
                intro_chunk = dict(first_doc_chunks[0])
                intro_chunk.setdefault("score", 0.5)
                already_present = any(
                    int(c.get("doc_id", 0)) == int(intro_chunk.get("doc_id", 0))
                    and int(c.get("chunk_index", -1))
                    == int(intro_chunk.get("chunk_index", -1))
                    for c in retrieved
                )
                if not already_present:
                    retrieved = [intro_chunk] + retrieved
        except FileNotFoundError:
            pass

    if not retrieved:
        return None, (
            jsonify(
                {"error": "No indexed content was found for the selected documents."}
            ),
            404,
        )

    recent_history = get_chat_history(
        db_path, user_id, document_id=history_doc_id, limit=10
    )
    return (
        {
            "question": question,
            "history_doc_id": history_doc_id,
            "retrieved": retrieved,
            "recent_history": recent_history,
            "mode": mode,
        },
        None,
    )


def create_app(test_config: Optional[dict[str, Any]] = None) -> Flask:
    """Application factory for CampusGPT."""
    app = Flask(
        __name__,
        template_folder=str(Config.BASE_DIR / "templates"),
        static_folder=str(Config.BASE_DIR / "static"),
    )
    app.config.from_object(Config)
    if test_config:
        app.config.update(test_config)

    for key in ("UPLOAD_FOLDER", "VECTOR_STORE_FOLDER", "MODEL_CACHE_FOLDER"):
        Path(app.config[key]).mkdir(parents=True, exist_ok=True)
    Path(app.config["DATABASE_PATH"]).parent.mkdir(parents=True, exist_ok=True)

    init_db(app.config["DATABASE_PATH"])

    app.rag_engine = RAGEngine(
        vector_store_folder=app.config["VECTOR_STORE_FOLDER"],
        model_name=app.config["EMBEDDING_MODEL_NAME"],
        model_cache_folder=app.config["MODEL_CACHE_FOLDER"],
    )
    app.llm_service = GeminiService(
        api_key=app.config.get("GEMINI_API_KEY", ""),
        model_name=app.config.get("GEMINI_MODEL", "gemini-3.8-flash"),
    )

    # ---------------- Page Routes ----------------

    @app.route("/")
    def index():
        if session.get("user_id"):
            return redirect(url_for("dashboard"))
        return redirect(url_for("login"))

    @app.route("/register", methods=["GET", "POST"])
    def register():
        if session.get("user_id"):
            return redirect(url_for("dashboard"))

        if request.method == "POST":
            full_name = request.form.get("full_name", "").strip()
            username = request.form.get("username", "").strip()
            email = request.form.get("email", "").strip()
            password = request.form.get("password", "")

            errors = validate_registration_input(full_name, username, email, password)
            db_path = app.config["DATABASE_PATH"]

            if not errors:
                if get_user_by_identifier(db_path, username):
                    errors.append("That username is already registered.")
                if get_user_by_identifier(db_path, email):
                    errors.append("That email address is already registered.")

            if errors:
                for err in errors:
                    flash(err, "danger")
                return (
                    render_template(
                        "register.html",
                        full_name=full_name,
                        username=username,
                        email=email,
                    ),
                    400,
                )

            pw_hash = hash_password(password)
            user_id = create_user(db_path, username, email, pw_hash, full_name)
            session["user_id"] = user_id
            session["username"] = username.lower()
            session["full_name"] = full_name
            flash("Account created! Welcome to CampusGPT.", "success")
            return redirect(url_for("dashboard"))

        return render_template("register.html")

    @app.route("/login", methods=["GET", "POST"])
    def login():
        if session.get("user_id"):
            return redirect(url_for("dashboard"))

        if request.method == "POST":
            identifier = request.form.get("identifier", "").strip()
            password = request.form.get("password", "")

            if not identifier or not password:
                flash("Please enter both username/email and password.", "danger")
                return render_template("login.html", identifier=identifier), 400

            user = get_user_by_identifier(app.config["DATABASE_PATH"], identifier)
            if not user or not verify_password(user["password_hash"], password):
                flash("Invalid credentials. Please try again.", "danger")
                return render_template("login.html", identifier=identifier), 401

            session["user_id"] = user["id"]
            session["username"] = user["username"]
            session["full_name"] = user["full_name"]
            return redirect(url_for("dashboard"))

        return render_template("login.html")

    @app.route("/logout")
    def logout():
        session.clear()
        flash("You have been signed out.", "info")
        return redirect(url_for("login"))

    @app.route("/dashboard")
    @login_required
    def dashboard():
        return render_template("dashboard.html")

    # ---------------- API Routes ----------------

    @app.route("/api/health")
    def api_health():
        configured = app.llm_service.is_configured()
        return jsonify(
            {
                "status": "ok",
                "gemini_configured": configured,
                "model": app.config.get("GEMINI_MODEL", "gemini-3.8-flash"),
                "supported_formats": sorted(app.config["ALLOWED_EXTENSIONS"]),
                "message": (
                    f"{app.config.get('GEMINI_MODEL', 'gemini-3.8-flash')} is ready."
                    if configured
                    else "GEMINI_API_KEY is missing or set to a placeholder in .env (Local RAG mode still works)."
                ),
            }
        )

    @app.route("/api/documents", methods=["GET"])
    @login_required
    def api_list_documents():
        user_id = int(session["user_id"])
        docs = get_user_documents(app.config["DATABASE_PATH"], user_id)
        return jsonify({"documents": docs})

    @app.route("/api/documents/upload", methods=["POST"])
    @login_required
    def api_upload_document():
        if "file" not in request.files:
            return jsonify({"error": "No file part provided in upload request."}), 400

        file = request.files["file"]
        if not file or not file.filename:
            return jsonify({"error": "No document file selected."}), 400

        original_name = sanitize_text(file.filename)
        if not _allowed_file(original_name, app.config["ALLOWED_EXTENSIONS"]):
            exts = ", ".join(sorted(app.config["ALLOWED_EXTENSIONS"]))
            return jsonify({"error": f"Supported formats: {exts}."}), 400

        course_name = sanitize_text(request.form.get("course_name", "General")) or "General"
        safe_name = secure_filename(original_name) or "document.pdf"
        unique_name = f"{uuid.uuid4().hex}_{safe_name}"
        save_path = Path(app.config["UPLOAD_FOLDER"]) / unique_name
        file.save(str(save_path))

        doc_id: Optional[int] = None
        user_id = int(session["user_id"])
        db_path = app.config["DATABASE_PATH"]

        try:
            processed = process_document(
                str(save_path),
                chunk_size=app.config["CHUNK_SIZE"],
                chunk_overlap=app.config["CHUNK_OVERLAP"],
                ocr_callback=app.llm_service.extract_scanned_pdf_text
                if app.llm_service.is_configured()
                else None,
            )
            file_size = save_path.stat().st_size
            doc_id = create_document(
                db_path=db_path,
                user_id=user_id,
                filename=unique_name,
                original_filename=original_name,
                file_path=str(save_path),
                file_size=file_size,
                page_count=processed["page_count"],
                chunk_count=len(processed["chunks"]),
                course_name=course_name,
                file_type=processed["file_type"],
            )
            app.rag_engine.build_index(
                doc_id=doc_id,
                chunks=processed["chunks"],
                doc_name=original_name,
            )
            doc = get_document_by_id(db_path, doc_id, user_id)
            return jsonify({"document": doc}), 201
        except ValueError as exc:
            if doc_id is not None:
                delete_document(db_path, doc_id, user_id)
                app.rag_engine.delete_index(doc_id)
            if save_path.exists():
                save_path.unlink(missing_ok=True)
            return jsonify({"error": str(exc)}), 400
        except Exception as exc:
            if doc_id is not None:
                delete_document(db_path, doc_id, user_id)
                app.rag_engine.delete_index(doc_id)
            if save_path.exists():
                save_path.unlink(missing_ok=True)
            return jsonify({"error": f"Failed to process document: {exc}"}), 500

    @app.route("/api/documents/<int:doc_id>", methods=["PATCH", "DELETE"])
    @login_required
    def api_manage_document(doc_id: int):
        user_id = int(session["user_id"])
        db_path = app.config["DATABASE_PATH"]
        doc = get_document_by_id(db_path, doc_id, user_id)
        if not doc:
            return jsonify({"error": "Document not found."}), 404

        if request.method == "PATCH":
            payload = request.get_json(silent=True) or {}
            course_name = sanitize_text(payload.get("course_name", "General")) or "General"
            update_document_course(db_path, doc_id, user_id, course_name)
            return jsonify({"document": get_document_by_id(db_path, doc_id, user_id)})

        app.rag_engine.delete_index(doc_id)
        pdf_path = Path(doc["file_path"])
        if pdf_path.exists():
            pdf_path.unlink(missing_ok=True)

        delete_document(db_path, doc_id, user_id)
        return jsonify({"deleted": True, "document_id": doc_id})

    @app.route("/api/documents/<int:doc_id>/view", methods=["GET"])
    @login_required
    def api_view_document(doc_id: int):
        """Serve an uploaded PDF inline or return extracted page chunks for non-PDF previews."""
        user_id = int(session["user_id"])
        doc = get_document_by_id(app.config["DATABASE_PATH"], doc_id, user_id)
        if not doc:
            return jsonify({"error": "Document not found."}), 404

        file_path = Path(doc["file_path"])
        if not file_path.exists():
            return jsonify({"error": "Stored file is missing on disk."}), 404

        if doc.get("file_type") == "pdf" or file_path.suffix.lower() == ".pdf":
            pdf_bytes = file_path.read_bytes()
            return send_file(
                io.BytesIO(pdf_bytes),
                mimetype="application/pdf",
                as_attachment=False,
                download_name=doc["original_filename"],
            )

        _, chunks = app.rag_engine.load_index(doc_id)
        return jsonify({"document": doc, "chunks": chunks})

    @app.route("/api/chat", methods=["POST"])
    @login_required
    def api_chat():
        user_id = int(session["user_id"])
        payload = request.get_json(silent=True) or {}
        ctx, err = _prepare_chat_context(app, user_id, payload)
        if err:
            return err

        started = time.perf_counter()
        try:
            answer = app.llm_service.answer_question(
                question=ctx["question"],
                retrieved_chunks=ctx["retrieved"],
                history=ctx["recent_history"],
                mode=ctx["mode"],
            )
        except LLMServiceError as exc:
            return jsonify({"error": exc.message, "code": exc.code}), exc.status_code

        latency_ms = int((time.perf_counter() - started) * 1000)
        engine_used = app.llm_service.last_model_used
        sources = _dedupe_sources(ctx["retrieved"])
        top_score = max((float(c.get("score", 0.0)) for c in ctx["retrieved"]), default=0.0)

        db_path = app.config["DATABASE_PATH"]
        add_chat_message(db_path, user_id, ctx["history_doc_id"], "user", ctx["question"], "[]")
        add_chat_message(
            db_path, user_id, ctx["history_doc_id"], "assistant", answer, json.dumps(sources)
        )
        log_query_metric(
            db_path,
            user_id,
            ctx["history_doc_id"],
            "chat",
            engine_used,
            latency_ms,
            top_score,
            len(ctx["retrieved"]),
        )

        return jsonify(
            {
                "answer": answer,
                "sources": sources,
                "engine_used": engine_used,
                "latency_ms": latency_ms,
                "top_score": round(top_score, 4),
            }
        )

    @app.route("/api/chat/stream", methods=["POST"])
    @login_required
    def api_chat_stream():
        """Stream RAG answers token-by-token via Server-Sent Events (SSE)."""
        user_id = int(session["user_id"])
        payload = request.get_json(silent=True) or {}
        ctx, err = _prepare_chat_context(app, user_id, payload)
        if err:
            return err

        sources = _dedupe_sources(ctx["retrieved"])
        top_score = max((float(c.get("score", 0.0)) for c in ctx["retrieved"]), default=0.0)
        db_path = app.config["DATABASE_PATH"]

        def generate_sse():
            started = time.perf_counter()
            yield f"data: {json.dumps({'type': 'sources', 'sources': sources})}\n\n"
            chunks_acc: list[str] = []
            try:
                for piece in app.llm_service.stream_answer_question(
                    question=ctx["question"],
                    retrieved_chunks=ctx["retrieved"],
                    history=ctx["recent_history"],
                    mode=ctx["mode"],
                ):
                    chunks_acc.append(piece)
                    yield f"data: {json.dumps({'type': 'chunk', 'text': piece})}\n\n"
            except LLMServiceError as exc:
                yield f"data: {json.dumps({'type': 'error', 'error': exc.message, 'code': exc.code})}\n\n"
                return

            full_answer = "".join(chunks_acc)
            latency_ms = int((time.perf_counter() - started) * 1000)
            engine_used = app.llm_service.last_model_used

            add_chat_message(db_path, user_id, ctx["history_doc_id"], "user", ctx["question"], "[]")
            add_chat_message(
                db_path,
                user_id,
                ctx["history_doc_id"],
                "assistant",
                full_answer,
                json.dumps(sources),
            )
            log_query_metric(
                db_path,
                user_id,
                ctx["history_doc_id"],
                "chat_stream",
                engine_used,
                latency_ms,
                top_score,
                len(ctx["retrieved"]),
            )
            yield f"data: {json.dumps({'type': 'done', 'engine_used': engine_used, 'latency_ms': latency_ms, 'top_score': round(top_score, 4)})}\n\n"

        return Response(
            stream_with_context(generate_sse()),
            mimetype="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.route("/api/chat/history", methods=["GET", "DELETE"])
    @login_required
    def api_chat_history():
        user_id = int(session["user_id"])
        db_path = app.config["DATABASE_PATH"]
        doc_param = request.args.get("document_id")
        doc_id = int(doc_param) if doc_param and doc_param.isdigit() else None

        if request.method == "DELETE":
            cleared = clear_chat_history(db_path, user_id, document_id=doc_id)
            return jsonify({"cleared": cleared})

        messages = get_chat_history(db_path, user_id, document_id=doc_id, limit=50)
        return jsonify({"messages": messages})

    @app.route("/api/summarize", methods=["POST"])
    @login_required
    def api_summarize():
        payload = request.get_json(silent=True) or {}
        doc_id = payload.get("document_id")
        mode = str(payload.get("mode", "auto")).lower()
        if not doc_id:
            return jsonify({"error": "document_id is required."}), 400

        user_id = int(session["user_id"])
        db_path = app.config["DATABASE_PATH"]
        doc = get_document_by_id(db_path, int(doc_id), user_id)
        if not doc:
            return jsonify({"error": "Document not found."}), 404

        started = time.perf_counter()
        try:
            chunks = app.rag_engine.get_document_chunks(int(doc_id), max_chunks=30)
            summary = app.llm_service.summarize_document(
                doc_name=doc["original_filename"], chunks=chunks, mode=mode
            )
            latency_ms = int((time.perf_counter() - started) * 1000)
            engine_used = app.llm_service.last_model_used
            update_document_summary(db_path, int(doc_id), user_id, summary)
            artifact_id = create_study_artifact(
                db_path,
                user_id,
                int(doc_id),
                "summary",
                f"Summary: {doc['original_filename']}",
                summary,
                engine_used=engine_used,
            )
            log_query_metric(
                db_path, user_id, int(doc_id), "summarize", engine_used, latency_ms, 1.0, len(chunks)
            )
            return jsonify(
                {
                    "summary": summary,
                    "document_id": int(doc_id),
                    "artifact_id": artifact_id,
                    "engine_used": engine_used,
                    "latency_ms": latency_ms,
                }
            )
        except FileNotFoundError as exc:
            return jsonify({"error": str(exc)}), 404
        except LLMServiceError as exc:
            return jsonify({"error": exc.message, "code": exc.code}), exc.status_code

    @app.route("/api/quiz", methods=["POST"])
    @login_required
    def api_quiz():
        payload = request.get_json(silent=True) or {}
        doc_id = payload.get("document_id")
        mode = str(payload.get("mode", "auto")).lower()
        if not doc_id:
            return jsonify({"error": "document_id is required."}), 400

        num_questions = max(1, min(15, int(payload.get("num_questions", 5))))
        difficulty = str(payload.get("difficulty", "medium")).lower()
        if difficulty not in {"easy", "medium", "hard"}:
            difficulty = "medium"

        user_id = int(session["user_id"])
        db_path = app.config["DATABASE_PATH"]
        doc = get_document_by_id(db_path, int(doc_id), user_id)
        if not doc:
            return jsonify({"error": "Document not found."}), 404

        started = time.perf_counter()
        try:
            chunks = app.rag_engine.get_document_chunks(int(doc_id), max_chunks=25)
            questions, quiz_md = app.llm_service.generate_quiz_structured(
                doc_name=doc["original_filename"],
                chunks=chunks,
                num_questions=num_questions,
                difficulty=difficulty,
                mode=mode,
            )
            latency_ms = int((time.perf_counter() - started) * 1000)
            engine_used = app.llm_service.last_model_used
            artifact_id = create_study_artifact(
                db_path,
                user_id,
                int(doc_id),
                "quiz",
                f"Quiz ({difficulty.title()}): {doc['original_filename']}",
                quiz_md,
                content_json=json.dumps(questions),
                total_questions=len(questions) or num_questions,
                engine_used=engine_used,
            )
            log_query_metric(
                db_path, user_id, int(doc_id), "quiz", engine_used, latency_ms, 1.0, len(chunks)
            )
            return jsonify(
                {
                    "quiz": quiz_md,
                    "questions": questions,
                    "document_id": int(doc_id),
                    "artifact_id": artifact_id,
                    "engine_used": engine_used,
                    "latency_ms": latency_ms,
                }
            )
        except FileNotFoundError as exc:
            return jsonify({"error": str(exc)}), 404
        except LLMServiceError as exc:
            return jsonify({"error": exc.message, "code": exc.code}), exc.status_code

    @app.route("/api/quiz/<int:artifact_id>/score", methods=["POST"])
    @login_required
    def api_save_quiz_score(artifact_id: int):
        payload = request.get_json(silent=True) or {}
        score = int(payload.get("score", 0))
        total_questions = max(1, int(payload.get("total_questions", 1)))
        updated = update_quiz_score(
            app.config["DATABASE_PATH"],
            artifact_id,
            int(session["user_id"]),
            score,
            total_questions,
        )
        if not updated:
            return jsonify({"error": "Quiz artifact not found."}), 404
        return jsonify({"saved": True, "score": score, "total_questions": total_questions})

    @app.route("/api/viva", methods=["POST"])
    @login_required
    def api_viva():
        payload = request.get_json(silent=True) or {}
        doc_id = payload.get("document_id")
        mode = str(payload.get("mode", "auto")).lower()
        if not doc_id:
            return jsonify({"error": "document_id is required."}), 400

        num_questions = max(1, min(15, int(payload.get("num_questions", 5))))
        user_id = int(session["user_id"])
        db_path = app.config["DATABASE_PATH"]
        doc = get_document_by_id(db_path, int(doc_id), user_id)
        if not doc:
            return jsonify({"error": "Document not found."}), 404

        started = time.perf_counter()
        try:
            chunks = app.rag_engine.get_document_chunks(int(doc_id), max_chunks=25)
            viva = app.llm_service.generate_viva(
                doc_name=doc["original_filename"],
                chunks=chunks,
                num_questions=num_questions,
                mode=mode,
            )
            latency_ms = int((time.perf_counter() - started) * 1000)
            engine_used = app.llm_service.last_model_used
            artifact_id = create_study_artifact(
                db_path,
                user_id,
                int(doc_id),
                "viva",
                f"Viva Prep: {doc['original_filename']}",
                viva,
                total_questions=num_questions,
                engine_used=engine_used,
            )
            log_query_metric(
                db_path, user_id, int(doc_id), "viva", engine_used, latency_ms, 1.0, len(chunks)
            )
            return jsonify(
                {
                    "viva": viva,
                    "document_id": int(doc_id),
                    "artifact_id": artifact_id,
                    "engine_used": engine_used,
                    "latency_ms": latency_ms,
                }
            )
        except FileNotFoundError as exc:
            return jsonify({"error": str(exc)}), 404
        except LLMServiceError as exc:
            return jsonify({"error": exc.message, "code": exc.code}), exc.status_code

    @app.route("/api/artifacts", methods=["GET"])
    @login_required
    def api_list_artifacts():
        user_id = int(session["user_id"])
        doc_param = request.args.get("document_id")
        doc_id = int(doc_param) if doc_param and doc_param.isdigit() else None
        artifacts = get_study_artifacts(
            app.config["DATABASE_PATH"], user_id, document_id=doc_id, limit=30
        )
        return jsonify({"artifacts": artifacts})

    @app.route("/api/artifacts/<int:artifact_id>", methods=["DELETE"])
    @login_required
    def api_delete_artifact(artifact_id: int):
        deleted = delete_study_artifact(
            app.config["DATABASE_PATH"], artifact_id, int(session["user_id"])
        )
        if not deleted:
            return jsonify({"error": "Artifact not found."}), 404
        return jsonify({"deleted": True})

    @app.route("/api/metrics", methods=["GET"])
    @login_required
    def api_metrics():
        summary = get_user_metrics_summary(
            app.config["DATABASE_PATH"], int(session["user_id"])
        )
        return jsonify(summary)

    @app.errorhandler(RequestEntityTooLarge)
    def handle_file_too_large(_err):
        return jsonify({"error": "Uploaded file exceeds the 25 MB size limit."}), 413

    return app


app = create_app()

if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(
        host="127.0.0.1",
        port=port,
        debug=os.getenv("FLASK_DEBUG", "false").lower() == "true",
    )
