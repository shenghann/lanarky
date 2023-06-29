from typing import Any

from lanarky.register import (
    register_streaming_callback,
    register_streaming_json_callback,
    register_websocket_callback,
)
from lanarky.schemas import StreamingJSONResponse

from .base import AsyncLanarkyCallback
from .llm import (
    AsyncLLMChainStreamingCallback,
    AsyncLLMChainStreamingJSONCallback,
    AsyncLLMChainWebsocketCallback,
)

SOURCE_DOCUMENT_TEMPLATE = """
product_data: {document_metadata}
"""

class AsyncAgentsLanarkyCallback(AsyncLanarkyCallback):
    """Base AsyncCallback handler for AgentExecutor.

    Adapted from `langchain/callbacks/streaming_stdout_final_only.py <https://github.com/hwchase17/langchain/blob/master/langchain/callbacks/streaming_stdout_final_only.py>`_
    """

    answer_prefix_tokens: list[str] = ['Final', ' Answer', '",\n', '   ', ' "', 'action', '_input', '":', ' "', ] # ["Final", " Answer", ":"]
    last_tokens: list[str] = [""] * len(answer_prefix_tokens)
    answer_reached: bool = False

    async def on_llm_start(
        self, serialized: dict[str, Any], prompts: list[str], **kwargs: Any
    ) -> None:
        """Run when LLM starts running."""
        self.last_tokens = [""] * len(self.answer_prefix_tokens)
        self.answer_reached = False

    def _check_if_answer_reached(self, token: str):
        """Checks if the final answer has been reached."""
        self.last_tokens.append(token)
        if len(self.last_tokens) > len(self.answer_prefix_tokens):
            self.last_tokens.pop(0)
        # print("_check_if_answer_reached", token, self.last_tokens)

        if self.last_tokens == self.answer_prefix_tokens:
            print('answer_reached!!!!!!')
            self.answer_reached = True
            return

        return self.answer_reached


@register_streaming_callback("AgentExecutor")
class AsyncAgentsStreamingCallback(
    AsyncAgentsLanarkyCallback, AsyncLLMChainStreamingCallback
):
    """AsyncStreamingResponseCallback handler for AgentExecutor."""

    source_document_template: str = SOURCE_DOCUMENT_TEMPLATE

    async def on_chain_end(self, outputs: dict[str, Any], **kwargs: Any) -> None:
        """Run when chain ends running."""
        if "source_documents" in outputs:
            message = self._construct_message("\n\nSOURCE DOCUMENTS:\n")
            await self.send(message)
            for document in outputs["source_documents"]:
                document_metadata = "\n".join(
                    [f"{k}: {v}" for k, v in document.metadata.items()]
                )
                message = self._construct_message(
                    self.source_document_template.format(
                        page_content=document.page_content,
                        document_metadata=document_metadata,
                    )
                )
                await self.send(message)

    async def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Run on new LLM token. Only available when streaming is enabled."""
        if self._check_if_answer_reached(token):
            message = self._construct_message(token)
            await self.send(message)


@register_websocket_callback("AgentExecutor")
class AsyncAgentsWebsocketCallback(
    AsyncAgentsLanarkyCallback, AsyncLLMChainWebsocketCallback
):
    """AsyncWebsocketCallback handler for AgentExecutor."""

    source_document_template: str = SOURCE_DOCUMENT_TEMPLATE

    async def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Run on new LLM token. Only available when streaming is enabled."""
        if self._check_if_answer_reached(token):
            message = self._construct_message(token)
            await self.websocket.send_json(message)

    async def on_chain_end(self, outputs: dict[str, Any], **kwargs: Any) -> None:
        """Run when chain ends running."""

        if "source_documents" in outputs:
            # message = self._construct_message("\n\nSOURCE DOCUMENTS:\n")
            # await self.websocket.send_json(message)
            for document in outputs["source_documents"]:
                document_metadata = "\n".join(
                    [f"{k}: {v}" for k, v in document.metadata.items()]
                )
                message = self._construct_message(
                    self.source_document_template.format(
                        page_content=document.page_content,
                        document_metadata=document_metadata,
                    )
                )
                # print(message)
                await self.websocket.send_json(message)


@register_streaming_json_callback("AgentExecutor")
class AsyncAgentsStreamingJSONCallback(
    AsyncAgentsLanarkyCallback, AsyncLLMChainStreamingJSONCallback
):
    """AsyncStreamingJSONCallback handler for AgentExecutor."""

    async def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Run on new LLM token. Only available when streaming is enabled."""
        if self._check_if_answer_reached(token):
            message = self._construct_message(StreamingJSONResponse(token=token))
            await self.send(message)
