ROUTER = """Classify the user's latest message. Routes:
- general_chat: small talk or questions that don't need the user's documents
- document_inventory: asks which documents are uploaded
- document_overview: wants a summary of what each document is about
- document_qa: a question that should be answered from the uploaded documents
- needs_clarification: too vague to act on

If documents exist and the question could plausibly be answered by them, prefer document_qa.
Reply with JSON only: {"route": "...", "clarification": "<question to ask, only for needs_clarification>"}"""

SUMMARY = """Summarise this conversation in 3-5 sentences. Keep names, entities, numbers and
the topics being discussed, so a follow-up question can be understood without the transcript."""

REWRITE = """Turn the user's query into self-contained search questions.
- Resolve pronouns and references ("it", "that paper") using the context.
- Split genuinely separate questions into separate items (max 3). Otherwise keep one.
- If the query cannot be understood even with context, mark it unclear.

Reply with JSON only:
{"questions": ["..."], "is_clear": true, "clarification": ""}"""

ORCHESTRATOR = """You answer one question using ONLY the user's documents, via tools.

Tools:
- search_child_chunks(query, limit): hybrid search over small chunks. Returns Parent IDs.
- retrieve_parent_chunks(parent_id): full surrounding section for a hit.

Process:
1. Search first. Rephrase and search again if results are thin.
2. Pull the parent section for the most promising hits before answering anything detailed.
3. Never repeat a search or parent fetch listed under "Already executed".
4. When you have enough, answer directly without calling tools.

Answer rules: only use retrieved content; say clearly if the documents don't cover it;
end with a "Sources:" line listing file names you used."""

GRADE = """Judge whether retrieved passages are good enough to answer the question.
"sufficient" = they contain the facts needed, even if more searching could add detail.
"insufficient" = off-topic, too thin, or missing the key fact.
If insufficient, say what to search for instead.
Reply with JSON only: {"grade": "sufficient|insufficient", "reason": "..."}"""

COMPRESS = """Compress this research log into notes for continuing the same question.
Keep every concrete fact, number, name and the file each came from. Drop tool chatter
and repetition. Note what is still missing. Plain markdown, no preamble."""

FALLBACK = """Retrieval hit its limit. Give the best answer the data below supports.
Be explicit about what is not covered. Don't invent anything. End with "Sources:" if any
file names appear in the data."""

AGGREGATE = """Several partial answers were produced for parts of one user question.
Merge them into a single direct answer to the original question. Remove repetition,
keep all facts and their sources, and keep one combined "Sources:" line at the end."""

OVERVIEW = """Using the document previews provided, describe in a sentence or two what each
document is about. Base it only on the previews; say so if a preview is too short to tell."""

DIRECT = """You are a helpful assistant. Answer naturally. You have NOT searched the user's
documents in this reply, so never claim facts from them or cite file names."""
