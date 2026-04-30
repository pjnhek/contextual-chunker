CONTEXTUAL_CHUNK_PROMPT = """You are a document context specialist.

Given this FULL DOCUMENT:
---
{full_document}
---

And this CHUNK from the document:
---
{chunk_text}
---

Generate a concise contextual summary (2-3 sentences, max 100 tokens) that:
1. Identifies what this document is about (product, topic, use case)
2. Explains where this chunk fits in the overall document structure
3. Preserves key identifiers (product names, model numbers, brand names)

RULES:
- Be concise and factual
- Focus on context that helps retrieval (what is this document about?)
- Do NOT summarize the chunk itself - provide CONTEXT for it
- Preserve exact product/device/brand names from the document
- Use present tense, active voice
- Return ONLY the contextual summary, no JSON, no preamble

Contextual Summary:"""

CONTEXT_SEPARATOR = "\n\n---\n\n"
