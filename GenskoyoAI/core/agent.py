"""Agent 主类 - 异步优化版"""

from typing import AsyncIterator
from pathlib import Path

import ollama
from ollama import Message, ChatResponse
from msgspec import Struct

from .config import AppConfig, ConfigLoader
from .events import EventBus
from .exceptions import AgentError, ModelError
from ..memory.working import WorkingMemoryManager
from ..memory.episodic import EpisodicMemoryManager
from ..memory.semantic import SemanticMemoryManager
from ..memory.types import MemoryRecord
from ..tools.registry import ToolRegistry
from ..tools.executor import ToolExecutor
from ..session.manager import SessionManager
from ..session.context import SessionContext
from ..utils.logging import logger
from ..utils.helpers import safe_get, sync_to_async


class StreamChunk(Struct):
    """流式响应块"""

    content: str = ""
    is_tool_call: bool = False
    tool_info: dict | None = None


class Agent:
    """AI 角色扮演 Agent - 只负责模型交互，不负责输出展示"""

    def __init__(
        self,
        config: AppConfig | None = None,
        config_file: Path | None = None,
        character_file: Path | None = None,
    ):
        # 加载配置
        loader = ConfigLoader()
        self.config = config or loader.load(config_file)

        # 加载角色
        if character_file:
            self.config.character = loader.load_character(character_file)
        elif self.config.character_file:
            self.config.character = loader.load_character(self.config.character_file)

        if not self.config.character:
            raise AgentError("No character loaded")

        # 初始化事件总线
        self.event_bus = EventBus()

        self._working_memory: WorkingMemoryManager | None = None

        # 初始化会话管理
        character_name = safe_get(self.config, "character.name", "default")
        self.session_manager = SessionManager(self.config.session, character_name)

        # 初始化记忆系统
        base_path = self.config.session.save_path / "memory"
        self.episodic_memory = EpisodicMemoryManager(
            self.config.memory,
            character_name,
            None,
        )
        self.semantic_memory = SemanticMemoryManager(
            self.config.memory, character_name, base_path
        )

        # 初始化工具系统
        self.tool_registry = ToolRegistry()
        self.tool_executor = ToolExecutor(self.tool_registry)
        self._current_response = ""

        # 创建异步版本的 ollama 调用
        self._ollama_chat_async = sync_to_async(ollama.chat)

        self._setup()

    @property
    def working_memory(self) -> WorkingMemoryManager:
        """获取当前会话的工作记忆"""
        if not (current_session := self.session_manager.get_current_session()):
            raise AgentError("No active session")

        if not self._working_memory:
            self._working_memory = self.session_manager.get_working_memory(
                current_session.session_id
            )
        return self._working_memory

    @working_memory.setter
    def working_memory(self, memory: WorkingMemoryManager) -> None:
        """设置当前会话的工作记忆"""
        self._working_memory = memory

    def _setup(self) -> None:
        """设置 Agent"""
        self.system_prompt = self._build_system_prompt()
        character_name = safe_get(self.config, "character.name", "unknown")
        logger.info(f"Agent 初始化完成，角色: {character_name}")

    def _build_system_prompt(self) -> str:
        """构建系统提示词"""
        prompt = safe_get(self.config, "character.system_prompt", "")

        # 添加工具说明
        if self.config.tool.enabled and (tools := self.tool_registry.list()):
            tools_desc = "\n\n【可用工具】\n"
            tools_desc += "\n".join(f"- {t.name}: {t.description}" for t in tools)
            prompt += tools_desc
            prompt += "\n当需要获取外部信息时，请调用相应的工具。调用工具后，将结果整合到回复中。"

        return prompt

    def _build_messages(self, user_input: str) -> list[dict[str, str]]:
        """构建消息列表"""
        messages: list[dict[str, str]] = [
            {"role": "system", "content": self.system_prompt}
        ]

        # 情景记忆（摘要）
        if episodic_context := self.episodic_memory.get_relevant_context(user_input):
            messages.append(
                {
                    "role": "system",
                    "content": "【历史记忆摘要】\n" + "\n".join(episodic_context),
                }
            )

        # 语义记忆（相关记忆）
        if semantic_context := self.semantic_memory.get_relevant_context(user_input):
            messages.append(
                {
                    "role": "system",
                    "content": "【相关记忆】\n" + "\n".join(semantic_context),
                }
            )

        # 工作记忆（当前对话）
        messages.extend(self.working_memory.get_context())

        # 当前用户输入
        messages.append({"role": "user", "content": user_input})

        return messages

    async def _call_model(
        self, messages: list[dict[str, str]], tools: list[dict] | None = None
    ) -> ChatResponse:
        """调用模型（异步）"""
        kwargs: dict = {
            "model": self.config.model.name,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": self.config.model.temperature,
                "top_p": self.config.model.top_p,
                "num_predict": self.config.model.max_tokens,
            },
        }

        if self.config.model.think:
            kwargs["think"] = True

        if tools:
            kwargs["tools"] = tools
            logger.debug(f"传递了 {len(tools)} 个工具到模型")

        try:
            # 使用异步版本
            return await self._ollama_chat_async(**kwargs)
        except Exception as e:
            raise ModelError(f"模型调用失败: {e}") from e

    async def _call_model_stream(
        self, messages: list[dict[str, str]], tools: list[dict] | None = None
    ) -> AsyncIterator[StreamChunk]:
        """流式调用模型，返回流式迭代器

        注意：ollama.chat 的 stream 模式返回的是生成器，不是异步生成器。
        我们使用 sync_to_async 包装整个流式调用，但需要特殊处理。
        """
        kwargs: dict = {
            "model": self.config.model.name,
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": self.config.model.temperature,
                "top_p": self.config.model.top_p,
                "num_predict": self.config.model.max_tokens,
            },
        }

        if self.config.model.think:
            kwargs["think"] = True

        if tools:
            kwargs["tools"] = tools

        try:
            # ollama 的 stream 返回的是同步生成器
            # 我们在线程池中运行整个流式处理
            import asyncio
            from concurrent.futures import ThreadPoolExecutor

            def run_stream():
                for chunk in ollama.chat(**kwargs):
                    yield chunk

            executor = ThreadPoolExecutor(max_workers=1)
            loop = asyncio.get_event_loop()

            # 创建一个队列来传递数据
            queue = asyncio.Queue()

            def producer():
                try:
                    for chunk in run_stream():
                        loop.call_soon_threadsafe(queue.put_nowait, chunk)
                except Exception as e:
                    loop.call_soon_threadsafe(queue.put_nowait, e)
                finally:
                    loop.call_soon_threadsafe(queue.put_nowait, None)  # 结束标记

            executor.submit(producer)

            while True:
                item = await queue.get()
                if item is None:
                    break
                if isinstance(item, Exception):
                    raise ModelError(f"流式模型调用失败: {item}") from item

                if hasattr(item, "message"):
                    if hasattr(item.message, "tool_calls") and item.message.tool_calls:
                        yield StreamChunk(
                            is_tool_call=True, tool_info={"message": item.message}
                        )
                    elif content := getattr(item.message, "content", ""):
                        yield StreamChunk(content=content)
                elif isinstance(item, dict):
                    if message := item.get("message", {}):
                        if message.get("tool_calls"):
                            yield StreamChunk(
                                is_tool_call=True, tool_info={"message": message}
                            )
                        elif content := message.get("content"):
                            yield StreamChunk(content=content)

            executor.shutdown(wait=False)

        except Exception as e:
            raise ModelError(f"流式模型调用失败: {e}") from e

    async def _handle_tool_calls(self, response: ChatResponse) -> list[dict] | None:
        """处理工具调用"""
        message = getattr(response, "message", None) or response.get("message", {})
        tool_calls = getattr(message, "tool_calls", []) or message.get("tool_calls", [])

        if not tool_calls:
            return None

        logger.info(f"检测到 {len(tool_calls)} 个工具调用")

        if parsed_calls := self.tool_executor.parse_tool_calls(message):
            return await self.tool_executor.execute_batch(parsed_calls)

        return None

    async def _handle_tool_calls_from_message(
        self, message: Message
    ) -> list[dict] | None:
        """处理来自流式响应的 Message 对象的工具调用"""
        if not (tool_calls := getattr(message, "tool_calls", [])):
            return None

        logger.info(f"检测到 {len(tool_calls)} 个工具调用")
        for tc in tool_calls:
            if hasattr(tc, "function"):
                logger.info(f"工具调用: {tc.function.name}")

        if parsed_calls := self.tool_executor.parse_tool_calls_from_message(message):
            return await self.tool_executor.execute_batch(parsed_calls)

        return None

    async def send(self, user_input: str) -> str:
        """发送消息并获取完整回复（非流式）"""
        self._record_user_message(user_input)

        tools = self.tool_registry.get_schemas() if self.config.tool.enabled else None
        messages = self._build_messages(user_input)
        response = await self._call_model(messages, tools)
        content = response.get("message", {}).get("content", "")

        if tool_results := await self._handle_tool_calls(response):
            content = await self._handle_tool_chain(content, tool_results)

        if content:
            self._record_assistant_message(content)
            await self._auto_memory(user_input, content)

        self._save_session_if_needed()
        return content

    async def send_stream(self, user_input: str) -> AsyncIterator[StreamChunk]:
        """发送消息并获取流式回复迭代器"""
        self._record_user_message(user_input)

        tools = self.tool_registry.get_schemas() if self.config.tool.enabled else None
        messages = self._build_messages(user_input)

        full_content = ""
        tool_calls_message = None

        async for chunk in self._call_model_stream(messages, tools):
            if chunk.is_tool_call and chunk.tool_info:
                tool_calls_message = chunk.tool_info["message"]
            else:
                full_content += chunk.content
            yield chunk

        if tool_calls_message and (
            tool_results := await self._handle_tool_calls_from_message(
                tool_calls_message
            )
        ):
            self._record_tool_results(tool_results)
            self._save_working_memory_if_needed()

            async for chunk in self._continue_with_tool_results_stream():
                full_content += chunk.content
                yield chunk

        if full_content:
            self._record_assistant_message(full_content)
            await self._auto_memory(user_input, full_content)

        self._save_session_if_needed()

    async def _continue_with_tool_results_stream(self) -> AsyncIterator[StreamChunk]:
        """带着工具结果继续对话 - 流式版本"""
        messages = self._build_continuation_messages()
        async for chunk in self._call_model_stream(messages, None):
            yield chunk

    async def _handle_tool_chain(
        self, partial_content: str, tool_results: list[dict]
    ) -> str:
        """处理工具调用链，返回完整响应"""
        self._record_tool_results(tool_results)

        if partial_content:
            self.working_memory.add_message("assistant", partial_content)

        self._save_working_memory_if_needed()
        continuation = await self._continue_with_tool_results()

        return partial_content + continuation

    async def _continue_with_tool_results(self) -> str:
        """带着工具结果继续对话"""
        messages = self._build_continuation_messages()
        response = await self._call_model(messages, None)
        return response.get("message", {}).get("content", "")

    def _build_continuation_messages(self) -> list[dict[str, str]]:
        """构建继续对话的消息列表"""
        messages: list[dict[str, str]] = [
            {"role": "system", "content": self.system_prompt}
        ]
        messages.extend(self.working_memory.get_context())
        messages.append(
            {
                "role": "system",
                "content": "工具调用已完成，请自然地将结果信息融入你的回复中，保持角色风格。",
            }
        )
        return messages

    def _record_user_message(self, content: str) -> None:
        """记录用户消息"""
        character_name = safe_get(self.config, "character.name", "default")

        self.working_memory.add_message("user", content)
        self.episodic_memory.add_message(
            MemoryRecord(
                content=content,
                role="user",
                character_id=character_name,
            )
        )

    def _record_assistant_message(self, content: str) -> None:
        """记录助手消息（带去重保护）"""
        if not content:
            return

        wm = self.working_memory
        character_name = safe_get(self.config, "character.name", "default")

        if wm._memory.messages:
            if (last_msg := wm._memory.messages[-1])[
                "role"
            ] == "assistant" and last_msg["content"] == content:
                logger.debug("跳过重复的助手消息记录")
                return

        self.working_memory.add_message("assistant", content)
        self.episodic_memory.add_message(
            MemoryRecord(
                content=content,
                role="assistant",
                character_id=character_name,
            )
        )

    def _record_tool_results(self, results: list[dict]) -> None:
        """记录工具调用结果"""
        character_name = safe_get(self.config, "character.name", "default")

        for result in results:
            self.working_memory.add_message(result["role"], result["content"])
            self.episodic_memory.add_message(
                MemoryRecord(
                    content=result["content"],
                    role="tool",
                    character_id=character_name,
                    metadata={"tool_name": result.get("name", "")},
                )
            )

    def _save_working_memory_if_needed(self) -> None:
        """根据需要保存工作记忆"""
        if self.config.session.auto_save:
            self.session_manager.save_working_memory()

    def _save_session_if_needed(self) -> None:
        """根据需要保存会话"""
        if self.config.session.auto_save:
            self.session_manager.save_current()

    async def _auto_memory(self, user_input: str, assistant_response: str) -> None:
        """自动记忆重要信息"""
        if not self.config.memory.auto_memory_enabled:
            return

        importance = 0.0
        if len(user_input) > 50:
            importance += 0.3
        if any(kw in user_input for kw in ["记住", "重要", "我叫", "我是"]):
            importance += 0.4
        if len(assistant_response) > 100:
            importance += 0.2

        if importance > 0.5:
            await self.semantic_memory.add_async(user_input, importance)

    def rollback(self, turns: int = 1) -> None:
        """回滚对话"""
        wm = self.working_memory
        for _ in range(turns * 2):
            if wm._memory.messages:
                wm._memory.messages.pop()

    def create_session(self) -> SessionContext:
        """创建新会话"""
        session = self.session_manager.create_session()
        self.working_memory.clear()
        return session

    def resume_session(self, session_id: str) -> bool:
        """恢复会话"""
        if self.session_manager.set_current_session(session_id):
            self.working_memory = self.session_manager.get_working_memory(session_id)
            return True
        return False

    def shutdown(self) -> None:
        """关闭 Agent"""
        if self.config.session.auto_save:
            self.session_manager.save_working_memory()
            self.session_manager.save_current()
        logger.info("Agent 已关闭")
