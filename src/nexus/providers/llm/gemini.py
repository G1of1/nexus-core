"""Google Gemini LLM provider."""

from collections.abc import AsyncIterator
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from nexus.config import NexusSettings
from nexus.exceptions import ConfigurationError, GenerationError
from nexus.generation.prompts import PromptManager
from nexus.models.generation import GenerationRequest, GenerationResponse, StreamChunk
from nexus.providers.base import LLMProvider


class GeminiLLMProvider(LLMProvider):
    """Generates answers using Google Gemini chat models."""

    def __init__(
        self,
        settings: NexusSettings,
        prompt_manager: PromptManager | None = None,
    ) -> None:
        if not settings.gemini_api_key:
            raise ConfigurationError("NEXUS_GEMINI_API_KEY is required for Gemini LLM")
        self._settings = settings
        self._prompt_manager = prompt_manager or PromptManager()
        self._client = ChatGoogleGenerativeAI(
            model=settings.llm_model,
            google_api_key=settings.gemini_api_key,
            temperature=settings.llm_temperature,
            max_output_tokens=settings.llm_max_tokens,
        )

    def _build_messages(self, request: GenerationRequest) -> list[SystemMessage | HumanMessage]:
        system = request.system_prompt or self._prompt_manager.build_system_prompt()
        user = self._prompt_manager.build_user_prompt(request.query, request.context)
        return [SystemMessage(content=system), HumanMessage(content=user)]

    def _extract_token_usage(self, metadata: dict[str, Any]) -> dict[str, int]:
        usage = metadata.get("token_usage") or metadata.get("usage_metadata") or {}
        if not usage:
            return {}

        prompt = usage.get("prompt_tokens") or usage.get("prompt_token_count") or 0
        completion = (
            usage.get("completion_tokens")
            or usage.get("candidates_token_count")
            or usage.get("output_token_count")
            or 0
        )
        total = usage.get("total_tokens") or usage.get("total_token_count") or prompt + completion
        return {
            "prompt_tokens": int(prompt),
            "completion_tokens": int(completion),
            "total_tokens": int(total),
        }

    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        messages = self._build_messages(request)
        try:
            response = await self._client.ainvoke(messages)
            content = response.content if isinstance(response.content, str) else str(response.content)
            return GenerationResponse(
                answer=content,
                citations=self._prompt_manager.build_citations(request.context),
                model=self._settings.llm_model,
                token_usage=self._extract_token_usage(response.response_metadata),
            )
        except Exception as e:
            raise GenerationError(f"Failed to generate response: {e}") from e

    async def stream(self, request: GenerationRequest) -> AsyncIterator[StreamChunk]:
        messages = self._build_messages(request)
        try:
            async for chunk in self._client.astream(messages):
                content = chunk.content if isinstance(chunk.content, str) else str(chunk.content)
                if content:
                    yield StreamChunk(content=content)
            yield StreamChunk(
                content="",
                is_final=True,
                citations=self._prompt_manager.build_citations(request.context),
            )
        except Exception as e:
            raise GenerationError(f"Failed to stream response: {e}") from e
