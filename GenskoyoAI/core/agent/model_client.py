"""模型客户端 - 封装 Ollama 异步调用"""

# GenskoyoAI/core/agent/model_client.py

import asyncio
from typing import AsyncIterator, Optional

from ollama import AsyncClient as OllamaAsyncClient
from ollama import ChatResponse
from msgspec import Struct

from ..config import ModelConfig
from ..exceptions import ModelError
from ...utils.logging import logger


class StreamChunk(Struct):
    """流式响应块"""

    content: str = ""
    is_tool_call: bool = False
    tool_info: dict | None = None


class ModelClient:
    """
    模型客户端 - 纯粹封装 Ollama 调用

    职责：
    - 管理 Ollama 异步客户端实例
    - 提供非流式和流式调用接口
    - 处理超时和异常
    - 不涉及任何业务逻辑（记忆、会话、工具等）
    """

    def __init__(self, config: ModelConfig):
        """
        初始化模型客户端

        Args:
            config: 模型配置
        """
        self.config = config
        self._client = self._build_client()
        logger.debug(f"ModelClient 初始化完成，模型: {config.name}")

    def _build_client(self) -> OllamaAsyncClient:
        """构建 Ollama 异步客户端"""
        return OllamaAsyncClient(host=self.config.base_url)

    def _build_options(self) -> dict:
        """构建模型选项"""
        return {
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "num_predict": self.config.max_tokens,
        }

    async def chat(
        self,
        messages: list[dict[str, str]],
        tools: Optional[list[dict]] = None,
    ) -> ChatResponse:
        """
        非流式调用模型

        Args:
            messages: 消息列表
            tools: 工具 schema 列表（可选）

        Returns:
            ChatResponse: 完整的模型响应

        Raises:
            ModelError: 调用失败或超时
        """
        kwargs = {
            "model": self.config.name,
            "messages": messages,
            "tools": tools,
            "options": self._build_options(),
            "stream": False,
        }

        # think 参数是可选的，只有支持 think 的模型才传
        if hasattr(self.config, "think"):
            kwargs["think"] = self.config.think

        try:
            logger.debug(f"非流式调用模型，消息数: {len(messages)}")
            response = await asyncio.wait_for(
                self._client.chat(**kwargs),
                timeout=self.config.timeout,
            )
            logger.debug(f"模型响应完成，长度: {len(response.message.content or '')}")
            return response

        except asyncio.TimeoutError:
            logger.error(f"模型调用超时 ({self.config.timeout}s)")
            raise ModelError(f"模型调用超时 ({self.config.timeout}秒)")

        except Exception as e:
            logger.error(f"模型调用失败: {e}")
            raise ModelError(f"模型调用失败: {e}") from e

    async def chat_stream(
        self,
        messages: list[dict[str, str]],
        tools: Optional[list[dict]] = None,
    ) -> AsyncIterator[StreamChunk]:
        """
        流式调用模型

        Args:
            messages: 消息列表
            tools: 工具 schema 列表（可选）

        Yields:
            StreamChunk: 流式响应块

        Raises:
            ModelError: 调用失败
        """
        kwargs = {
            "model": self.config.name,
            "messages": messages,
            "tools": tools,
            "options": self._build_options(),
            "stream": True,
        }

        if hasattr(self.config, "think"):
            kwargs["think"] = self.config.think

        try:
            logger.debug(f"流式调用模型，消息数: {len(messages)}")
            stream = await self._client.chat(**kwargs)

            async for chunk in stream:
                message = chunk.message

                # 检查是否有工具调用
                if message.tool_calls:
                    yield StreamChunk(
                        is_tool_call=True,
                        tool_info={"message": message},
                    )
                elif message.content:
                    yield StreamChunk(content=message.content)

        except Exception as e:
            logger.error(f"流式模型调用失败: {e}")
            raise ModelError(f"流式模型调用失败: {e}") from e

    def update_config(self, config: ModelConfig) -> None:
        """
        更新配置（例如运行时切换模型）

        Args:
            config: 新的模型配置
        """
        self.config = config
        self._client = self._build_client()
        logger.info(f"ModelClient 配置已更新，模型: {config.name}")

    @property
    def model_name(self) -> str:
        """获取当前使用的模型名称"""
        return self.config.name
