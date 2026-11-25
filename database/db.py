"""SQLite3 database initialization, migrations, and parameterized query helpers."""

import sqlite3
from pathlib import Path
from typing import Any, Optional


def get_connection(db_path: str) -> sqlite3.Connection:
    """Open a SQLite connection with Row factory and foreign key enforcement."""
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def _ensure_column(
    conn: sqlite3.Connection, table: str, column: str, definition: str
) -> None:
    """Safely add a column to an existing table if missing."""
    existing = {
        row["name"]
        for row in conn.execute(f"PRAGMA table_info({table});").fetchall()
    }
    if column not in existing:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition};")


def init_db(db_path: str) -> None:
    """Create database tables, apply lightweight migrations, and ensure indexes."""
    with get_connection(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                full_name TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                filename TEXT NOT NULL,
                original_filename TEXT NOT NULL,
                file_path TEXT NOT NULL,
                file_size INTEGER NOT NULL,
                page_count INTEGER NOT NULL DEFAULT 0,
                chunk_count INTEGER NOT NULL DEFAULT 0,
                summary TEXT NOT NULL DEFAULT '',
                course_name TEXT NOT NULL DEFAULT 'General',
                file_type TEXT NOT NULL DEFAULT 'pdf',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                document_id INTEGER,
                role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
                content TEXT NOT NULL,
                sources_json TEXT NOT NULL DEFAULT '[]',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE,
                FOREIGN KEY (document_id) REFERENCES documents (id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS study_artifacts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                document_id INTEGER NOT NULL,
                artifact_type TEXT NOT NULL CHECK(artifact_type IN ('summary', 'quiz', 'viva')),
                title TEXT NOT NULL,
                content_json TEXT NOT NULL DEFAULT '[]',
                raw_markdown TEXT NOT NULL DEFAULT '',
                score INTEGER,
                total_questions INTEGER,
                engine_used TEXT NOT NULL DEFAULT 'gemini',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE,
                FOREIGN KEY (document_id) REFERENCES documents (id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS query_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                document_id INTEGER,
                endpoint TEXT NOT NULL,
                engine_used TEXT NOT NULL,
                latency_ms INTEGER NOT NULL DEFAULT 0,
                top_score REAL NOT NULL DEFAULT 0.0,
                chunks_retrieved INTEGER NOT NULL DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE,
                FOREIGN KEY (document_id) REFERENCES documents (id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_documents_user
                ON documents (user_id, created_at DESC);

            CREATE INDEX IF NOT EXISTS idx_chat_user_doc
                ON chat_history (user_id, document_id, created_at ASC);

            CREATE INDEX IF NOT EXISTS idx_artifacts_user
                ON study_artifacts (user_id, created_at DESC);

            CREATE INDEX IF NOT EXISTS idx_metrics_user
                ON query_metrics (user_id, created_at DESC);
            """
        )
        _ensure_column(conn, "documents", "course_name", "TEXT NOT NULL DEFAULT 'General'")
        _ensure_column(conn, "documents", "file_type", "TEXT NOT NULL DEFAULT 'pdf'")
        conn.commit()


def create_user(
    db_path: str, username: str, email: str, password_hash: str, full_name: str
) -> int:
    """Insert a new user record and return its ID."""
    with get_connection(db_path) as conn:
        cursor = conn.execute(
            """
            INSERT INTO users (username, email, password_hash, full_name)
            VALUES (?, ?, ?, ?)
            """,
            (username.strip().lower(), email.strip().lower(), password_hash, full_name.strip()),
        )
        conn.commit()
        return int(cursor.lastrowid)


def get_user_by_identifier(db_path: str, identifier: str) -> Optional[dict[str, Any]]:
    """Fetch a user by username or email."""
    clean_id = identifier.strip().lower()
    with get_connection(db_path) as conn:
        row = conn.execute(
            """
            SELECT id, username, email, password_hash, full_name, created_at
            FROM users
            WHERE username = ? OR email = ?
            """,
            (clean_id, clean_id),
        ).fetchone()
    return dict(row) if row else None


def get_user_by_id(db_path: str, user_id: int) -> Optional[dict[str, Any]]:
    """Fetch a user by primary key ID."""
    with get_connection(db_path) as conn:
        row = conn.execute(
            """
            SELECT id, username, email, full_name, created_at
            FROM users
            WHERE id = ?
            """,
            (user_id,),
        ).fetchone()
    return dict(row) if row else None


def create_document(
    db_path: str,
    user_id: int,
    filename: str,
    original_filename: str,
    file_path: str,
    file_size: int,
    page_count: int,
    chunk_count: int,
    course_name: str = "General",
    file_type: str = "pdf",
) -> int:
    """Insert an uploaded document record and return its ID."""
    clean_course = (course_name or "General").strip() or "General"
    clean_type = (file_type or "pdf").strip().lower()
    with get_connection(db_path) as conn:
        cursor = conn.execute(
            """
            INSERT INTO documents (
                user_id, filename, original_filename, file_path,
                file_size, page_count, chunk_count, course_name, file_type
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_id,
                filename,
                original_filename,
                file_path,
                file_size,
                page_count,
                chunk_count,
                clean_course,
                clean_type,
            ),
        )
        conn.commit()
        return int(cursor.lastrowid)


def get_user_documents(db_path: str, user_id: int) -> list[dict[str, Any]]:
    """Return all documents belonging to a user ordered newest first."""
    with get_connection(db_path) as conn:
        rows = conn.execute(
            """
            SELECT id, user_id, filename, original_filename, file_path,
                   file_size, page_count, chunk_count, summary,
                   course_name, file_type, created_at
            FROM documents
            WHERE user_id = ?
            ORDER BY created_at DESC, id DESC
            """,
            (user_id,),
        ).fetchall()
    return [dict(row) for row in rows]


def get_document_by_id(
    db_path: str, doc_id: int, user_id: int
) -> Optional[dict[str, Any]]:
    """Return a single document owned by the given user."""
    with get_connection(db_path) as conn:
        row = conn.execute(
            """
            SELECT id, user_id, filename, original_filename, file_path,
                   file_size, page_count, chunk_count, summary,
                   course_name, file_type, created_at
            FROM documents
            WHERE id = ? AND user_id = ?
            """,
            (doc_id, user_id),
        ).fetchone()
    return dict(row) if row else None


def update_document_summary(
    db_path: str, doc_id: int, user_id: int, summary: str
) -> bool:
    """Persist a generated summary on the document record."""
    with get_connection(db_path) as conn:
        cursor = conn.execute(
            """
            UPDATE documents
            SET summary = ?
            WHERE id = ? AND user_id = ?
            """,
            (summary, doc_id, user_id),
        )
        conn.commit()
        return cursor.rowcount > 0


def update_document_course(
    db_path: str, doc_id: int, user_id: int, course_name: str
) -> bool:
    """Update the course/subject tag for a document."""
    clean_course = (course_name or "General").strip() or "General"
    with get_connection(db_path) as conn:
        cursor = conn.execute(
            """
            UPDATE documents
            SET course_name = ?
            WHERE id = ? AND user_id = ?
            """,
            (clean_course, doc_id, user_id),
        )
        conn.commit()
        return cursor.rowcount > 0


def delete_document(db_path: str, doc_id: int, user_id: int) -> bool:
    """Delete a document owned by the user."""
    with get_connection(db_path) as conn:
        cursor = conn.execute(
            "DELETE FROM documents WHERE id = ? AND user_id = ?",
            (doc_id, user_id),
        )
        conn.commit()
        return cursor.rowcount > 0


def add_chat_message(
    db_path: str,
    user_id: int,
    document_id: Optional[int],
    role: str,
    content: str,
    sources_json: str = "[]",
) -> int:
    """Insert a chat message into history."""
    with get_connection(db_path) as conn:
        cursor = conn.execute(
            """
            INSERT INTO chat_history (user_id, document_id, role, content, sources_json)
            VALUES (?, ?, ?, ?, ?)
            """,
            (user_id, document_id, role, content, sources_json),
        )
        conn.commit()
        return int(cursor.lastrowid)


def get_chat_history(
    db_path: str,
    user_id: int,
    document_id: Optional[int] = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Fetch recent chat history for a user, optionally filtered by document."""
    with get_connection(db_path) as conn:
        if document_id is not None:
            rows = conn.execute(
                """
                SELECT id, user_id, document_id, role, content, sources_json, created_at
                FROM chat_history
                WHERE user_id = ? AND document_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (user_id, document_id, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT id, user_id, document_id, role, content, sources_json, created_at
                FROM chat_history
                WHERE user_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (user_id, limit),
            ).fetchall()
    return [dict(row) for row in reversed(rows)]


def clear_chat_history(
    db_path: str, user_id: int, document_id: Optional[int] = None
) -> int:
    """Clear chat messages for the user (all or for a specific document)."""
    with get_connection(db_path) as conn:
        if document_id is not None:
            cursor = conn.execute(
                "DELETE FROM chat_history WHERE user_id = ? AND document_id = ?",
                (user_id, document_id),
            )
        else:
            cursor = conn.execute(
                "DELETE FROM chat_history WHERE user_id = ?",
                (user_id,),
            )
        conn.commit()
        return cursor.rowcount


def create_study_artifact(
    db_path: str,
    user_id: int,
    document_id: int,
    artifact_type: str,
    title: str,
    raw_markdown: str,
    content_json: str = "[]",
    total_questions: Optional[int] = None,
    engine_used: str = "gemini",
) -> int:
    """Save a generated summary, quiz, or viva artifact."""
    with get_connection(db_path) as conn:
        cursor = conn.execute(
            """
            INSERT INTO study_artifacts (
                user_id, document_id, artifact_type, title,
                content_json, raw_markdown, total_questions, engine_used
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_id,
                document_id,
                artifact_type,
                title,
                content_json,
                raw_markdown,
                total_questions,
                engine_used,
            ),
        )
        conn.commit()
        return int(cursor.lastrowid)


def get_study_artifacts(
    db_path: str,
    user_id: int,
    document_id: Optional[int] = None,
    limit: int = 30,
) -> list[dict[str, Any]]:
    """List saved study artifacts joined with document metadata."""
    with get_connection(db_path) as conn:
        if document_id is not None:
            rows = conn.execute(
                """
                SELECT a.*, d.original_filename AS doc_name, d.course_name
                FROM study_artifacts a
                JOIN documents d ON d.id = a.document_id
                WHERE a.user_id = ? AND a.document_id = ?
                ORDER BY a.id DESC
                LIMIT ?
                """,
                (user_id, document_id, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT a.*, d.original_filename AS doc_name, d.course_name
                FROM study_artifacts a
                JOIN documents d ON d.id = a.document_id
                WHERE a.user_id = ?
                ORDER BY a.id DESC
                LIMIT ?
                """,
                (user_id, limit),
            ).fetchall()
    return [dict(row) for row in rows]


def update_quiz_score(
    db_path: str, artifact_id: int, user_id: int, score: int, total_questions: int
) -> bool:
    """Persist a user's score on an interactive quiz artifact."""
    with get_connection(db_path) as conn:
        cursor = conn.execute(
            """
            UPDATE study_artifacts
            SET score = ?, total_questions = ?
            WHERE id = ? AND user_id = ? AND artifact_type = 'quiz'
            """,
            (score, total_questions, artifact_id, user_id),
        )
        conn.commit()
        return cursor.rowcount > 0


def delete_study_artifact(db_path: str, artifact_id: int, user_id: int) -> bool:
    """Delete a saved study artifact."""
    with get_connection(db_path) as conn:
        cursor = conn.execute(
            "DELETE FROM study_artifacts WHERE id = ? AND user_id = ?",
            (artifact_id, user_id),
        )
        conn.commit()
        return cursor.rowcount > 0


def log_query_metric(
    db_path: str,
    user_id: int,
    document_id: Optional[int],
    endpoint: str,
    engine_used: str,
    latency_ms: int,
    top_score: float = 0.0,
    chunks_retrieved: int = 0,
) -> int:
    """Record a RAG query performance metric."""
    with get_connection(db_path) as conn:
        cursor = conn.execute(
            """
            INSERT INTO query_metrics (
                user_id, document_id, endpoint, engine_used,
                latency_ms, top_score, chunks_retrieved
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_id,
                document_id,
                endpoint,
                engine_used,
                int(latency_ms),
                float(top_score),
                int(chunks_retrieved),
            ),
        )
        conn.commit()
        return int(cursor.lastrowid)


def get_user_metrics_summary(db_path: str, user_id: int) -> dict[str, Any]:
    """Compute aggregate RAG performance and quiz statistics for a user."""
    with get_connection(db_path) as conn:
        agg = conn.execute(
            """
            SELECT
                COUNT(*) AS total_queries,
                COALESCE(AVG(latency_ms), 0) AS avg_latency_ms,
                COALESCE(AVG(CASE WHEN top_score > 0 THEN top_score END), 0) AS avg_top_score,
                SUM(CASE WHEN engine_used != 'local-extractive-rag' THEN 1 ELSE 0 END) AS cloud_queries,
                SUM(CASE WHEN engine_used = 'local-extractive-rag' THEN 1 ELSE 0 END) AS local_queries
            FROM query_metrics
            WHERE user_id = ?
            """,
            (user_id,),
        ).fetchone()

        quiz_agg = conn.execute(
            """
            SELECT
                COUNT(*) AS quiz_attempts,
                COALESCE(AVG(CAST(score AS REAL) * 100.0 / NULLIF(total_questions, 0)), 0) AS avg_quiz_pct
            FROM study_artifacts
            WHERE user_id = ? AND artifact_type = 'quiz' AND score IS NOT NULL
            """,
            (user_id,),
        ).fetchone()

        recent = conn.execute(
            """
            SELECT m.endpoint, m.engine_used, m.latency_ms, m.top_score,
                   m.chunks_retrieved, m.created_at,
                   COALESCE(d.original_filename, 'Multi-Document') AS doc_name
            FROM query_metrics m
            LEFT JOIN documents d ON d.id = m.document_id
            WHERE m.user_id = ?
            ORDER BY m.id DESC
            LIMIT 15
            """,
            (user_id,),
        ).fetchall()

    return {
        "total_queries": int(agg["total_queries"] or 0),
        "avg_latency_ms": round(float(agg["avg_latency_ms"] or 0), 1),
        "avg_top_score": round(float(agg["avg_top_score"] or 0), 3),
        "cloud_queries": int(agg["cloud_queries"] or 0),
        "local_queries": int(agg["local_queries"] or 0),
        "quiz_attempts": int(quiz_agg["quiz_attempts"] or 0),
        "avg_quiz_pct": round(float(quiz_agg["avg_quiz_pct"] or 0), 1),
        "recent_queries": [dict(row) for row in recent],
    }
