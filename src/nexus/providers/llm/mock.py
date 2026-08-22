"""Mock LLM provider for testing."""

from collections.abc import AsyncIterator

from nexus.generation.prompts import PromptManager
from nexus.models.generation import GenerationRequest, GenerationResponse, StreamChunk
from nexus.providers.base import LLMProvider


class MockLLMProvider(LLMProvider):
    """Returns deterministic responses for testing."""

    def __init__(self, prompt_manager: PromptManager | None = None) -> None:
        self._prompt_manager = prompt_manager or PromptManager()

    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        context_count = len(request.context)
        answer = f"Mock answer for: {request.query} (using {context_count} context chunks)"
        return GenerationResponse(
            answer=answer,
            citations=self._prompt_manager.build_citations(request.context),
            model="mock-llm",
        )

    async def stream(self, request: GenerationRequest) -> AsyncIterator[StreamChunk]:
        response = await self.generate(request)
        words = response.answer.split()
        for word in words:
            yield StreamChunk(content=word + " ")
        yield StreamChunk(content="", is_final=True, citations=response.citations)
