"""Gemini Flash service using google-genai with streaming, structured quizzes, and local fallback."""

import json
import re
import time
from pathlib import Path
from typing import Any, Iterator, Optional

from google import genai
from google.genai import types

from modules.document_processor import sanitize_text
from modules.local_fallback import (
    local_answer_question,
    local_generate_quiz_structured,
    local_generate_viva,
    local_summarize_document,
)

DEFAULT_FALLBACK_MODELS = (
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-2.5-flash-lite",
    "gemma-3-27b-it",
    "gemma-3-12b-it",
)


class LLMServiceError(Exception):
    """Structured error raised for missing/invalid API keys or fatal Gemini failures."""

    def __init__(
        self, message: str, code: str = "API_ERROR", status_code: int = 502
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.status_code = status_code


class GeminiService:
    """Wrapper around google-genai with live model discovery, streaming, and local RAG fallback."""

    def __init__(
        self,
        api_key: str,
        model_name: str = "gemini-3.8-flash",
        fallback_models: tuple[str, ...] = DEFAULT_FALLBACK_MODELS,
        retry_delay: float = 0.6,
        enable_local_fallback: bool = True,
    ) -> None:
        self.api_key = (api_key or "").strip()
        self.model_name = model_name
        self.fallback_models = tuple(m for m in fallback_models if m != model_name)
        self.retry_delay = retry_delay
        self.enable_local_fallback = enable_local_fallback
        self._client: Optional[genai.Client] = None
        self._discovered_models: Optional[list[str]] = None
        self._discovered_at: float = 0.0
        self.last_model_used: str = model_name

    def is_configured(self) -> bool:
        """Return True if a non-placeholder GEMINI_API_KEY is present."""
        return bool(
            self.api_key
            and self.api_key != "your_api_key_here"
            and not self.api_key.startswith("your_")
        )

    def _get_client(self) -> genai.Client:
        if not self.is_configured():
            raise LLMServiceError(
                "GEMINI_API_KEY is missing or not configured. "
                "Add a valid key to CampusGPT/.env and restart the server.",
                code="MISSING_API_KEY",
                status_code=400,
            )
        if self._client is None:
            self._client = genai.Client(api_key=self.api_key)
        return self._client

    @staticmethod
    def _is_auth_error(err_str: str) -> bool:
        lowered = err_str.lower()
        return any(
            marker in lowered
            for marker in (
                "api_key_invalid",
                "api key not valid",
                "invalid api key",
                "permission_denied",
                "401",
                "403",
            )
        )

    @staticmethod
    def _is_transient_or_unavailable(err_str: str) -> bool:
        lowered = err_str.lower()
        return any(
            marker in lowered
            for marker in (
                "503",
                "unavailable",
                "high demand",
                "overloaded",
                "429",
                "resource_exhausted",
                "404",
                "not_found",
                "timeout",
                "timed out",
                "connection",
            )
        )

    def _discover_candidate_models(self, client: genai.Client) -> list[str]:
        """Discover live generateContent models for the API key, ranked by suitability."""
        now = time.time()
        if self._discovered_models and (now - self._discovered_at) < 300:
            return self._discovered_models

        ordered: list[str] = [self.model_name]
        discovered_flash: list[str] = []
        discovered_pro: list[str] = []
        discovered_gemma: list[str] = []

        try:
            for model_info in client.models.list():
                raw_name = getattr(model_info, "name", "") or ""
                short_name = raw_name.replace("models/", "").strip()
                if not short_name or short_name == self.model_name:
                    continue

                methods = getattr(model_info, "supported_actions", None) or getattr(
                    model_info, "supported_generation_methods", None
                )
                if methods and "generateContent" not in methods:
                    continue

                lower_name = short_name.lower()
                if any(
                    skip in lower_name
                    for skip in ("embedding", "imagen", "veo", "tts", "audio", "live")
                ):
                    continue

                if "flash" in lower_name:
                    discovered_flash.append(short_name)
                elif "gemma" in lower_name and "-it" in lower_name:
                    discovered_gemma.append(short_name)
                elif "pro" in lower_name:
                    discovered_pro.append(short_name)
        except Exception:
            pass

        discovered_flash.sort(reverse=True)
        discovered_pro.sort(reverse=True)
        discovered_gemma.sort(reverse=True)

        for candidate in (
            *discovered_flash,
            *self.fallback_models,
            *discovered_pro,
            *discovered_gemma,
        ):
            if candidate not in ordered:
                ordered.append(candidate)

        self._discovered_models = ordered[:8]
        self._discovered_at = now
        return self._discovered_models

    @staticmethod
    def _build_config(
        model_name: str,
        system_instruction: str,
        temperature: float,
        include_thinking: bool = True,
    ) -> tuple[types.GenerateContentConfig, Optional[str]]:
        lower_name = model_name.lower()
        if "gemma" in lower_name:
            return (
                types.GenerateContentConfig(temperature=temperature),
                system_instruction,
            )

        kwargs: dict[str, Any] = {
            "system_instruction": system_instruction,
            "temperature": temperature,
        }
        if include_thinking:
            if "2.5" in lower_name:
                kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
            elif "3" in lower_name:
                kwargs["thinking_config"] = types.ThinkingConfig(thinking_level="low")

        return types.GenerateContentConfig(**kwargs), None

    def _generate(
        self, system_instruction: str, prompt: str, temperature: float = 0.25
    ) -> str:
        """Execute a Gemini request across discovered live models before raising."""
        client = self._get_client()
        clean_prompt = sanitize_text(prompt)
        clean_system = sanitize_text(system_instruction)

        candidate_models = self._discover_candidate_models(client)
        last_exc: Optional[Exception] = None

        for model_idx, target_model in enumerate(candidate_models):
            attempts = 2 if model_idx == 0 else 1
            include_thinking = True

            for attempt in range(attempts):
                config, inline_sys = self._build_config(
                    target_model,
                    clean_system,
                    temperature,
                    include_thinking=include_thinking,
                )
                effective_prompt = (
                    f"{inline_sys}\n\n{clean_prompt}" if inline_sys else clean_prompt
                )

                try:
                    response = client.models.generate_content(
                        model=target_model,
                        contents=effective_prompt,
                        config=config,
                    )
                    text = sanitize_text(getattr(response, "text", "") or "")
                    if not text:
                        raise LLMServiceError(
                            "Gemini returned an empty response. Please try rephrasing.",
                            code="EMPTY_RESPONSE",
                            status_code=502,
                        )
                    self.last_model_used = target_model
                    return text
                except LLMServiceError:
                    raise
                except Exception as exc:
                    last_exc = exc
                    err_str = str(exc)
                    if self._is_auth_error(err_str):
                        raise LLMServiceError(
                            "Invalid or unauthorized GEMINI_API_KEY. "
                            "Please verify your key in CampusGPT/.env.",
                            code="INVALID_API_KEY",
                            status_code=401,
                        ) from exc

                    if include_thinking and (
                        "400" in err_str or "invalid_argument" in err_str.lower()
                    ):
                        include_thinking = False
                        try:
                            plain_cfg = types.GenerateContentConfig(temperature=temperature)
                            response = client.models.generate_content(
                                model=target_model,
                                contents=f"{clean_system}\n\n{clean_prompt}",
                                config=plain_cfg,
                            )
                            text = sanitize_text(getattr(response, "text", "") or "")
                            if text:
                                self.last_model_used = target_model
                                return text
                        except Exception as inner_exc:
                            last_exc = inner_exc
                            err_str = str(inner_exc)

                    if not self._is_transient_or_unavailable(err_str):
                        raise LLMServiceError(
                            f"Gemini API request failed: {err_str}",
                            code="API_ERROR",
                            status_code=502,
                        ) from exc

                    if attempt < attempts - 1 and self.retry_delay > 0:
                        time.sleep(self.retry_delay * (attempt + 1))

        err_msg = str(last_exc) if last_exc else "Service unavailable"
        raise LLMServiceError(
            f"Gemini cloud models are temporarily unavailable: {err_msg}",
            code="SERVICE_UNAVAILABLE",
            status_code=503,
        ) from last_exc

    def _generate_stream(
        self, system_instruction: str, prompt: str, temperature: float = 0.2
    ) -> Iterator[str]:
        """Stream response chunks from Gemini; falls back to single non-streamed call if unsupported."""
        client = self._get_client()
        clean_prompt = sanitize_text(prompt)
        clean_system = sanitize_text(system_instruction)
        candidate_models = self._discover_candidate_models(client)
        last_exc: Optional[Exception] = None

        for target_model in candidate_models:
            config, inline_sys = self._build_config(
                target_model, clean_system, temperature, include_thinking=True
            )
            effective_prompt = (
                f"{inline_sys}\n\n{clean_prompt}" if inline_sys else clean_prompt
            )
            try:
                stream = client.models.generate_content_stream(
                    model=target_model,
                    contents=effective_prompt,
                    config=config,
                )
                emitted = False
                for part in stream:
                    chunk_text = sanitize_text(getattr(part, "text", "") or "", strip=False)
                    if chunk_text:
                        emitted = True
                        yield chunk_text
                if emitted:
                    self.last_model_used = target_model
                    return
            except Exception as exc:
                last_exc = exc
                if self._is_auth_error(str(exc)):
                    raise LLMServiceError(
                        "Invalid or unauthorized GEMINI_API_KEY. "
                        "Please verify your key in CampusGPT/.env.",
                        code="INVALID_API_KEY",
                        status_code=401,
                    ) from exc
                if not self._is_transient_or_unavailable(str(exc)):
                    # Try non-streaming generation once on the same/next model
                    break

        # Final fallback via _generate (handles 400 retry or raises SERVICE_UNAVAILABLE)
        full_text = self._generate(clean_system, clean_prompt, temperature=temperature)
        yield full_text

    def extract_scanned_pdf_text(self, pdf_path: str) -> list[dict[str, Any]]:
        """Multimodal OCR fallback for scanned image-only PDFs using Gemini."""
        if not self.is_configured():
            return []
        client = self._get_client()
        pdf_bytes = Path(pdf_path).read_bytes()
        prompt = (
            "Extract all readable academic text from this scanned PDF cleanly, "
            "preserving headings and formulas. Separate pages using '--- PAGE N ---' markers."
        )
        for target_model in self._discover_candidate_models(client):
            if "gemma" in target_model.lower():
                continue
            try:
                response = client.models.generate_content(
                    model=target_model,
                    contents=[
                        types.Part.from_bytes(
                            data=pdf_bytes, mime_type="application/pdf"
                        ),
                        prompt,
                    ],
                )
                text = sanitize_text(getattr(response, "text", "") or "")
                if text:
                    self.last_model_used = target_model
                    return [{"page": 1, "text": text}]
            except Exception as exc:
                if self._is_auth_error(str(exc)):
                    break
                continue
        return []

    @staticmethod
    def _format_context(chunks: list[dict[str, Any]]) -> str:
        blocks: list[str] = []
        for idx, chunk in enumerate(chunks, start=1):
            doc_name = chunk.get("doc_name") or f"Document {chunk.get('doc_id', '')}"
            page = chunk.get("page", "?")
            text = sanitize_text(chunk.get("text", ""))
            blocks.append(f"[Source {idx} | {doc_name} | Page {page}]\n{text}")
        return "\n\n---\n\n".join(blocks)

    @staticmethod
    def _build_qa_prompts(
        question: str,
        retrieved_chunks: list[dict[str, Any]],
        history: Optional[list[dict[str, Any]]] = None,
    ) -> tuple[str, str]:
        context_str = GeminiService._format_context(retrieved_chunks)
        history_lines: list[str] = []
        if history:
            for item in history[-6:]:
                role_label = "Student" if item.get("role") == "user" else "CampusGPT"
                history_lines.append(
                    f"{role_label}: {sanitize_text(item.get('content', ''))}"
                )
        history_block = "\n".join(history_lines)

        system_instruction = (
            "You are CampusGPT, an expert university academic assistant. "
            "Answer the student's question accurately using ONLY the provided course excerpts. "
            "Cite page numbers inline like (Page 3) whenever referencing material. "
            "If the excerpts do not contain the answer, clearly state what is missing rather than guessing. "
            "Format responses cleanly in Markdown with bullet points, bold terms, or LaTeX math ($...$) where helpful."
        )

        prompt_parts = [f"### Retrieved Course Material\n{context_str}"]
        if history_block:
            prompt_parts.append(f"### Recent Conversation\n{history_block}")
        prompt_parts.append(f"### Student Question\n{sanitize_text(question)}")
        return system_instruction, "\n\n".join(prompt_parts)

    def answer_question(
        self,
        question: str,
        retrieved_chunks: list[dict[str, Any]],
        history: Optional[list[dict[str, Any]]] = None,
        mode: str = "auto",
    ) -> str:
        """Answer a student question grounded strictly in retrieved RAG chunks."""
        if not retrieved_chunks:
            raise LLMServiceError(
                "No relevant passages were found in the selected documents.",
                code="NO_CONTEXT",
                status_code=400,
            )

        if mode == "local":
            self.last_model_used = "local-extractive-rag"
            return local_answer_question(question, retrieved_chunks)

        system_instruction, prompt = self._build_qa_prompts(
            question, retrieved_chunks, history
        )
        try:
            return self._generate(
                system_instruction=system_instruction,
                prompt=prompt,
                temperature=0.2,
            )
        except LLMServiceError as exc:
            if self.enable_local_fallback and mode == "auto" and exc.code == "SERVICE_UNAVAILABLE":
                self.last_model_used = "local-extractive-rag"
                return local_answer_question(question, retrieved_chunks)
            raise

    def stream_answer_question(
        self,
        question: str,
        retrieved_chunks: list[dict[str, Any]],
        history: Optional[list[dict[str, Any]]] = None,
        mode: str = "auto",
    ) -> Iterator[str]:
        """Yield streamed answer chunks for SSE real-time UI updates."""
        if not retrieved_chunks:
            raise LLMServiceError(
                "No relevant passages were found in the selected documents.",
                code="NO_CONTEXT",
                status_code=400,
            )

        if mode == "local":
            self.last_model_used = "local-extractive-rag"
            yield local_answer_question(question, retrieved_chunks)
            return

        system_instruction, prompt = self._build_qa_prompts(
            question, retrieved_chunks, history
        )
        try:
            yield from self._generate_stream(
                system_instruction=system_instruction,
                prompt=prompt,
                temperature=0.2,
            )
        except LLMServiceError as exc:
            if self.enable_local_fallback and mode == "auto" and exc.code == "SERVICE_UNAVAILABLE":
                self.last_model_used = "local-extractive-rag"
                yield local_answer_question(question, retrieved_chunks)
                return
            raise

    def summarize_document(
        self, doc_name: str, chunks: list[dict[str, Any]], mode: str = "auto"
    ) -> str:
        """Generate a structured university revision summary for a document."""
        if mode == "local":
            self.last_model_used = "local-extractive-rag"
            return local_summarize_document(doc_name, chunks)

        context_str = self._format_context(chunks)
        system_instruction = (
            "You are CampusGPT, an academic study assistant. "
            "Produce a clear, well-structured study summary in Markdown strictly from the provided document excerpts."
        )
        prompt = (
            f"Create a comprehensive academic revision summary for **{sanitize_text(doc_name)}**.\n\n"
            "Include these Markdown sections:\n"
            "1. **Executive Overview**\n"
            "2. **Key Concepts & Definitions**\n"
            "3. **Important Processes, Theorems, or Formulas** (use LaTeX `$...$` if helpful)\n"
            "4. **Exam Revision Takeaways**\n\n"
            f"### Document Excerpts\n{context_str}"
        )
        try:
            return self._generate(
                system_instruction=system_instruction,
                prompt=prompt,
                temperature=0.25,
            )
        except LLMServiceError as exc:
            if self.enable_local_fallback and mode == "auto" and exc.code == "SERVICE_UNAVAILABLE":
                self.last_model_used = "local-extractive-rag"
                return local_summarize_document(doc_name, chunks)
            raise

    @staticmethod
    def _parse_quiz_json(raw_text: str) -> list[dict[str, Any]]:
        """Extract and validate a JSON list of MCQ objects from Gemini output."""
        cleaned = raw_text.strip()
        fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", cleaned)
        if fence_match:
            cleaned = fence_match.group(1).strip()

        bracket_match = re.search(r"\[[\s\S]*\]", cleaned)
        if bracket_match:
            cleaned = bracket_match.group(0)

        parsed = json.loads(cleaned)
        if not isinstance(parsed, list):
            return []

        validated: list[dict[str, Any]] = []
        for item in parsed:
            if not isinstance(item, dict):
                continue
            q_text = sanitize_text(str(item.get("question", "")))
            options = [sanitize_text(str(o)) for o in (item.get("options") or [])]
            if not q_text or len(options) != 4:
                continue
            try:
                correct_idx = int(item.get("correct_index", 0))
            except (TypeError, ValueError):
                correct_idx = 0
            correct_idx = max(0, min(3, correct_idx))
            try:
                page = int(item.get("page", 1))
            except (TypeError, ValueError):
                page = 1
            validated.append(
                {
                    "question": q_text,
                    "options": options,
                    "correct_index": correct_idx,
                    "explanation": sanitize_text(str(item.get("explanation", ""))),
                    "page": page,
                }
            )
        return validated

    @staticmethod
    def _quiz_to_markdown(doc_name: str, questions: list[dict[str, Any]], difficulty: str) -> str:
        letters = ("A", "B", "C", "D")
        lines = [f"## Practice Quiz ({difficulty.title()}): {sanitize_text(doc_name)}", ""]
        answers = ["## Answer Key & Explanations", ""]
        for idx, q in enumerate(questions, start=1):
            lines.append(f"**{idx}. {q['question']}**")
            for letter, opt in zip(letters, q["options"]):
                lines.append(f"- {letter}) {opt}")
            lines.append("")
            c_idx = q["correct_index"]
            answers.append(
                f"- **Q{idx}: {letters[c_idx]}) {q['options'][c_idx]}** — {q.get('explanation', '')} *(Page {q.get('page', 1)})*"
            )
        return "\n".join(lines + answers)

    def generate_quiz_structured(
        self,
        doc_name: str,
        chunks: list[dict[str, Any]],
        num_questions: int = 5,
        difficulty: str = "medium",
        mode: str = "auto",
    ) -> tuple[list[dict[str, Any]], str]:
        """Generate structured MCQs (for interactive grading) plus Markdown fallback."""
        if mode == "local":
            self.last_model_used = "local-extractive-rag"
            return local_generate_quiz_structured(
                doc_name, chunks, num_questions=num_questions, difficulty=difficulty
            )

        context_str = self._format_context(chunks)
        system_instruction = (
            "You are CampusGPT, a university exam preparation assistant. "
            "Return ONLY a valid JSON array of multiple-choice questions grounded strictly in the provided text."
        )
        prompt = (
            f"Generate exactly {num_questions} multiple-choice questions (difficulty: {difficulty}) "
            f"from {sanitize_text(doc_name)}.\n\n"
            "Return ONLY a JSON array where each element has:\n"
            '- "question": string\n'
            '- "options": array of exactly 4 strings\n'
            '- "correct_index": integer (0, 1, 2, or 3)\n'
            '- "explanation": concise 1-2 sentence explanation\n'
            '- "page": integer page number from the source\n\n'
            f"### Document Excerpts\n{context_str}"
        )
        try:
            raw = self._generate(
                system_instruction=system_instruction,
                prompt=prompt,
                temperature=0.3,
            )
            questions = self._parse_quiz_json(raw)
            if questions:
                return questions, self._quiz_to_markdown(doc_name, questions, difficulty)
            # If the model returned non-JSON prose (e.g. a mocked test string), preserve it cleanly
            return [], raw
        except LLMServiceError as exc:
            if self.enable_local_fallback and mode == "auto" and exc.code == "SERVICE_UNAVAILABLE":
                self.last_model_used = "local-extractive-rag"
                return local_generate_quiz_structured(
                    doc_name, chunks, num_questions=num_questions, difficulty=difficulty
                )
            raise

    def generate_quiz(
        self,
        doc_name: str,
        chunks: list[dict[str, Any]],
        num_questions: int = 5,
        difficulty: str = "medium",
        mode: str = "auto",
    ) -> str:
        """Generate multiple-choice questions as Markdown."""
        _, md = self.generate_quiz_structured(
            doc_name,
            chunks,
            num_questions=num_questions,
            difficulty=difficulty,
            mode=mode,
        )
        return md

    def generate_viva(
        self,
        doc_name: str,
        chunks: list[dict[str, Any]],
        num_questions: int = 5,
        mode: str = "auto",
    ) -> str:
        """Generate oral exam (Viva Voce) questions with model answers."""
        if mode == "local":
            self.last_model_used = "local-extractive-rag"
            return local_generate_viva(doc_name, chunks, num_questions=num_questions)

        context_str = self._format_context(chunks)
        system_instruction = (
            "You are CampusGPT, a university professor preparing a student for a Viva Voce (oral examination). "
            "Create insightful conceptual and applied oral exam questions grounded in the provided document."
        )
        prompt = (
            f"Generate **{num_questions}** Viva Voce (oral exam) questions and model answers "
            f"based on **{sanitize_text(doc_name)}**.\n\n"
            "For each question include:\n"
            "- **Question** (focus on 'why', 'how', trade-offs, or conceptual understanding)\n"
            "- **Model Answer** (concise, crisp oral response with page reference)\n"
            "- **Examiner Follow-up Tip** (what an examiner expects to hear)\n\n"
            f"### Document Excerpts\n{context_str}"
        )
        try:
            return self._generate(
                system_instruction=system_instruction,
                prompt=prompt,
                temperature=0.35,
            )
        except LLMServiceError as exc:
            if self.enable_local_fallback and mode == "auto" and exc.code == "SERVICE_UNAVAILABLE":
                self.last_model_used = "local-extractive-rag"
                return local_generate_viva(doc_name, chunks, num_questions=num_questions)
            raise
