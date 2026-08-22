"""Prompt templates and context assembly for RAG."""

from nexus.models.generation import Citation
from nexus.models.search import SearchResult

DEFAULT_SYSTEM_PROMPT = """You are a helpful AI assistant that answers questions based on provided context documents.

Rules:
1. Only use information from the provided context to answer questions.
2. If the context does not contain enough information to answer, say "I don't have enough information in the provided documents to answer this question."
3. Do not make up or infer facts not supported by the context.
4. Be concise and direct in your answers.
5. When referencing information, indicate which source it came from using [Source N] notation."""

CONTEXT_TEMPLATE = """Context documents:

{context}

---

Question: {query}

Answer based only on the context above:"""

CONTEXT_BLOCK_TEMPLATE = """[Source {index}] (document: {source}, relevance: {score:.2f})
{content}"""


class PromptManager:
    """Builds prompts and citations from retrieved context."""

    def __init__(
        self,
        system_prompt: str | None = None,
        context_template: str = CONTEXT_TEMPLATE,
        excerpt_length: int = 200,
    ) -> None:
        self._system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        self._context_template = context_template
        self._excerpt_length = excerpt_length

    def build_system_prompt(self) -> str:
        return self._system_prompt

    def build_context_block(self, results: list[SearchResult]) -> str:
        if not results:
            return "No relevant context found."
        blocks = []
        for i, result in enumerate(results, start=1):
            blocks.append(
                CONTEXT_BLOCK_TEMPLATE.format(
                    index=i,
                    source=result.source,
                    score=result.score,
                    content=result.content,
                )
            )
        return "\n\n".join(blocks)

    def build_user_prompt(self, query: str, context: list[SearchResult]) -> str:
        context_block = self.build_context_block(context)
        return self._context_template.format(context=context_block, query=query)

    def build_citations(self, results: list[SearchResult]) -> list[Citation]:
        citations = []
        for result in results:
            excerpt = result.content[: self._excerpt_length]
            if len(result.content) > self._excerpt_length:
                excerpt += "..."
            citations.append(
                Citation(
                    chunk_id=result.chunk_id,
                    document_id=result.document_id,
                    source=result.source,
                    excerpt=excerpt,
                    score=result.score,
                )
            )
        return citations
