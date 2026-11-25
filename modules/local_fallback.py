"""Local extractive RAG fallback when Gemini cloud endpoints return 503/unavailable."""

import re
from collections import Counter
from typing import Any

from modules.document_processor import sanitize_text

_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "could", "did",
    "do", "does", "for", "from", "had", "has", "have", "how", "i", "if", "in",
    "into", "is", "it", "its", "may", "might", "must", "of", "on", "or",
    "our", "pdf", "should", "so", "such", "than", "that", "the", "their",
    "them", "then", "there", "these", "they", "this", "to", "was", "we",
    "were", "what", "when", "where", "which", "who", "why", "will", "with",
    "would", "you", "your", "document", "file", "explain", "tell", "me", "about",
}


def _split_sentences(text: str) -> list[str]:
    clean = sanitize_text(text)
    if not clean:
        return []
    raw_parts = re.split(r"(?<=[.!?])\s+|\n+", clean)
    return [p.strip() for p in raw_parts if len(p.strip()) >= 12]


def _tokenize(text: str) -> list[str]:
    return [
        w.lower()
        for w in re.findall(r"[A-Za-z0-9][A-Za-z0-9_-]{1,}", text)
        if w.lower() not in _STOPWORDS
    ]


def local_answer_question(
    question: str, retrieved_chunks: list[dict[str, Any]]
) -> str:
    """Produce a grounded extractive answer from retrieved chunks with page citations."""
    q_tokens = set(_tokenize(question))
    q_lower = question.lower()
    asks_title_or_intro = any(
        word in q_lower
        for word in ("title", "name", "heading", "topic", "subject", "about", "overview")
    )

    scored_sentences: list[tuple[float, int, str, str]] = []
    seen_texts: set[str] = set()

    for chunk_rank, chunk in enumerate(retrieved_chunks):
        page = int(chunk.get("page", 1))
        doc_name = chunk.get("doc_name") or "Document"
        base_score = float(chunk.get("score", 0.5))
        sentences = _split_sentences(chunk.get("text", ""))
        if not sentences and chunk.get("text"):
            sentences = [sanitize_text(chunk["text"])]

        for sent_idx, sent in enumerate(sentences):
            norm = sent.lower()
            if norm in seen_texts:
                continue
            seen_texts.add(norm)

            s_tokens = set(_tokenize(sent))
            overlap = len(q_tokens & s_tokens) / max(1, len(q_tokens))
            score = base_score + (1.5 * overlap) - (chunk_rank * 0.05)
            if asks_title_or_intro and page == 1 and sent_idx < 3:
                score += 1.2
            scored_sentences.append((score, page, doc_name, sent))

    if not scored_sentences:
        return "No readable text passages were found in the selected document."

    scored_sentences.sort(key=lambda item: item[0], reverse=True)
    top_items = scored_sentences[:5]

    lines = ["### Extractive Answer from Your Document", ""]
    if asks_title_or_intro:
        top_items.sort(key=lambda item: (item[1], -item[0]))

    for _, page, doc_name, sent in top_items:
        lines.append(f"- {sent} *(Page {page} — {doc_name})*")

    lines.extend(
        [
            "",
            "> **Local RAG Mode:** Generated directly from your local Hybrid FAISS + BM25 index.",
        ]
    )
    return "\n".join(lines)


def local_summarize_document(doc_name: str, chunks: list[dict[str, Any]]) -> str:
    """Build a structured academic summary locally from representative document chunks."""
    if not chunks:
        return "No document content available to summarize."

    all_tokens: list[str] = []
    page_highlights: list[tuple[int, str]] = []
    seen_pages: set[int] = set()

    for chunk in chunks:
        text = sanitize_text(chunk.get("text", ""))
        all_tokens.extend(_tokenize(text))
        page = int(chunk.get("page", 1))
        sentences = _split_sentences(text)
        if sentences and page not in seen_pages:
            seen_pages.add(page)
            excerpt = " ".join(sentences[:2])
            page_highlights.append((page, excerpt))

    top_terms = [term for term, _ in Counter(all_tokens).most_common(10)]
    first_sentences = _split_sentences(chunks[0].get("text", ""))
    overview = (
        " ".join(first_sentences[:3])
        if first_sentences
        else sanitize_text(chunks[0].get("text", ""))[:350]
    )

    lines = [
        f"## Study Summary: {sanitize_text(doc_name)}",
        "",
        "### 1. Executive Overview",
        f"{overview} *(Page {chunks[0].get('page', 1)})*",
        "",
        "### 2. Key Terms & Focus Areas",
        ", ".join(f"`{t}`" for t in top_terms) if top_terms else "General academic concepts",
        "",
        "### 3. Section-by-Section Highlights",
    ]
    for page, excerpt in page_highlights[:10]:
        lines.append(f"- **Page {page}:** {excerpt}")

    lines.extend(
        [
            "",
            "> **Local RAG Mode:** Compiled directly from your indexed document.",
        ]
    )
    return "\n".join(lines)


def local_generate_quiz_structured(
    doc_name: str,
    chunks: list[dict[str, Any]],
    num_questions: int = 5,
    difficulty: str = "medium",
) -> tuple[list[dict[str, Any]], str]:
    """Generate structured interactive MCQs + Markdown locally from document chunks."""
    candidate_sentences: list[tuple[int, str]] = []
    term_pool: list[str] = []

    for chunk in chunks:
        page = int(chunk.get("page", 1))
        for sent in _split_sentences(chunk.get("text", "")):
            if 35 <= len(sent) <= 240:
                candidate_sentences.append((page, sent))
            for word in re.findall(r"\b[A-Za-z][A-Za-z0-9_-]{4,}\b", sent):
                if word.lower() not in _STOPWORDS and word not in term_pool:
                    term_pool.append(word)

    if not candidate_sentences:
        return [], "Not enough structured sentences were found in this document to build a quiz."

    step = max(1, len(candidate_sentences) // max(1, num_questions))
    selected = candidate_sentences[::step][:num_questions]

    structured: list[dict[str, Any]] = []
    questions_md = [
        f"## Practice Quiz ({difficulty.title()}): {sanitize_text(doc_name)}",
        "",
    ]
    answers_md = ["## Answer Key & Explanations", ""]
    letters = ("A", "B", "C", "D")

    for idx, (page, sent) in enumerate(selected, start=1):
        words = [
            w for w in re.findall(r"\b[A-Za-z][A-Za-z0-9_-]{4,}\b", sent)
            if w.lower() not in _STOPWORDS
        ]
        target = max(words, key=len) if words else "concept"
        blanked = re.sub(rf"\b{re.escape(target)}\b", "______", sent, count=1)

        distractors = [t for t in term_pool if t.lower() != target.lower()][:3]
        while len(distractors) < 3:
            distractors.append(f"Concept {len(distractors) + 1}")

        correct_idx = (idx - 1) % 4
        options = list(distractors[:3])
        options.insert(correct_idx, target)

        structured.append(
            {
                "question": f"Complete the statement from the course material: {blanked}",
                "options": options,
                "correct_index": correct_idx,
                "explanation": f'Based on Page {page}: "{sent}"',
                "page": page,
            }
        )

        questions_md.append(f"**{idx}. Complete the statement from the course material:**")
        questions_md.append(f"> {blanked}")
        for letter, opt in zip(letters, options):
            questions_md.append(f"- {letter}) {opt}")
        questions_md.append("")
        answers_md.append(
            f"- **Q{idx}: {letters[correct_idx]}) {target}** — Based on Page {page}: *\"{sent}\"*"
        )

    markdown = "\n".join(
        questions_md
        + answers_md
        + [
            "",
            "> **Local RAG Mode:** Generated directly from your indexed document.",
        ]
    )
    return structured, markdown


def local_generate_quiz(
    doc_name: str,
    chunks: list[dict[str, Any]],
    num_questions: int = 5,
    difficulty: str = "medium",
) -> str:
    """Backward-compatible Markdown-only quiz helper."""
    _, md = local_generate_quiz_structured(
        doc_name, chunks, num_questions=num_questions, difficulty=difficulty
    )
    return md


def local_generate_viva(
    doc_name: str, chunks: list[dict[str, Any]], num_questions: int = 5
) -> str:
    """Generate Viva Voce oral exam questions and grounded model answers locally."""
    candidate_sentences: list[tuple[int, str]] = []
    for chunk in chunks:
        page = int(chunk.get("page", 1))
        for sent in _split_sentences(chunk.get("text", "")):
            if len(sent) >= 35:
                candidate_sentences.append((page, sent))

    if not candidate_sentences:
        return "Not enough text was found in this document to build Viva questions."

    step = max(1, len(candidate_sentences) // max(1, num_questions))
    selected = candidate_sentences[::step][:num_questions]

    lines = [f"## Viva Voce Preparation: {sanitize_text(doc_name)}", ""]
    for idx, (page, sent) in enumerate(selected, start=1):
        keywords = [
            w for w in re.findall(r"\b[A-Za-z][A-Za-z0-9_-]{4,}\b", sent)
            if w.lower() not in _STOPWORDS
        ]
        focus_term = keywords[0] if keywords else "this concept"
        lines.extend(
            [
                f"### Question {idx}: What does the text explain about **{focus_term}**?",
                f"- **Model Answer (Page {page}):** {sent}",
                f"- **Examiner Follow-up Tip:** Be ready to define `{focus_term}` in your own words and link it to practical examples from Page {page}.",
                "",
            ]
        )

    lines.append(
        "> **Local RAG Mode:** Generated directly from your indexed document."
    )
    return "\n".join(lines)
