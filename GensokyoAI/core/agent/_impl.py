"""Agent 主类 - 异步优化版"""

# GensokyoAI/core/agent/_impl.py

from typing import AsyncIterator
from pathlib import Path
import asyncio
from contextvars import ContextVar
from uuid import uuid4

from ollama import Message

from .model_client import ModelClient, StreamChunk
from .save_coordinator import SaveCoordinator
from .message_builder import MessageBuilder
from .response_handler import ResponseHandler
from .lifecycle import LifecycleManager

from ..config import AppConfig, ConfigLoader
from ..events import EventBus
from ..exceptions import AgentError

from ...memory.working import WorkingMemoryManager
from ...memory.episodic import EpisodicMemoryManager
from ...memory.semantic import SemanticMemoryManager
from ...tools.registry import ToolRegistry
from ...tools.executor import ToolExecutor
from ...session.manager import SessionManager
from ...session.context import SessionContext
from ...utils.logging import logger
from ...utils.helpers import safe_get
from ...background import (
    BackgroundManager,
    TaskResult,
    MemoryWorker,
    PersistenceWorker,
)

# 请求上下文
request_id_var: ContextVar[str] = ContextVar("request_id", default="")


def get_request_id() -> str:
    """获取当前请求ID"""
    rid = request_id_var.get()
    if not rid:
        rid = str(uuid4())[:8]
        request_id_var.set(rid)
    return rid


class Agent:
    """AI 角色扮演 Agent - 只负责模型交互，不负责输出展示"""

    def __init__(
        self,
        config: AppConfig | None = None,
        config_file: Path | None = None,
        character_file: Path | None = None,
    ) -> None:
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
        self.session_manager = SessionManager(
            self.config.session,
            character_name,
            working_max_turns=self.config.memory.working_max_turns,
        )
        
        current_session = self.session_manager.get_current_session()
        if not current_session:
            session = self.session_manager.create_session()
            logger.debug(f"Agent 初始化时自动创建新会话: {session.session_id[:8]}...")
        else:
            current = self.session_manager.get_current_session()
            logger.debug(f"Agent 初始化时恢复会话: {current_session.session_id[:8]}... (轮数: {current_session.total_turns})")

        # 初始化记忆系统
        base_path = self.config.session.save_path / "memory"
        
        # 初始化工具系统
        self.tool_registry = ToolRegistry()
        self.tool_executor = ToolExecutor(self.tool_registry)

        # 创建模型客户端
        self._ollama_client = ModelClient(self.config.model)

        self.episodic_memory = EpisodicMemoryManager(
            self.config.memory,
            character_name,
            None,
            self._ollama_client
        )
        self.semantic_memory = SemanticMemoryManager(
            self.config.memory, 
            character_name, 
            base_path,
            self._ollama_client
        )

        # 请求管理
        self._request_semaphore = asyncio.Semaphore(1)

        # 构建系统提示词
        self.system_prompt = self._build_system_prompt()

        # 创建消息构建器（现在 working_memory 可以正常工作了）
        self.message_builder = MessageBuilder(
            system_prompt=self.system_prompt,
            working_memory=self.working_memory,
            episodic_memory=self.episodic_memory,
            semantic_memory=self.semantic_memory,
            tool_registry=self.tool_registry,
            tool_enabled=self.config.tool.enabled,
        )

        self.save_coordinator = SaveCoordinator(
            session_manager=self.session_manager,
            session_config=self.config.session,
        )

        self.response_handler = ResponseHandler(
            config=self.config,
            working_memory=self.working_memory,
            episodic_memory=self.episodic_memory,
            tool_executor=self.tool_executor,
            model_client=self._ollama_client,
            message_builder=self.message_builder,
            save_coordinator=self.save_coordinator,
        )

        # 创建生命周期管理器
        self.lifecycle = LifecycleManager(on_shutdown=self._async_save)

        # 注册需要接收关闭状态的组件
        self.lifecycle.register_component(self.save_coordinator)
        self.lifecycle.register_component(self.response_handler)

        # 设置信号处理器
        self.lifecycle.setup_signal_handlers()

        character_name = safe_get(self.config, "character.name", "unknown")
        logger.info(f"Agent 初始化完成，角色: {character_name}")

    # ==================== 属性 ====================

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

    @property
    def background_manager(self) -> BackgroundManager:
        """获取后台管理器（懒加载）"""
        # 懒加载：第一次访问时创建
        if not hasattr(self, "_background_manager") or self._background_manager is None:
            self._background_manager = self._create_background_manager()
        return self._background_manager

    @property
    def is_shutting_down(self) -> bool:
        """是否正在关闭"""
        return self.lifecycle.is_shutting_down

    # ==================== 初始化辅助方法 ====================

    def _build_system_prompt(self) -> str:
        """构建系统提示词（纯角色设定，工具说明由 MessageBuilder 添加）"""
        if not self.config.character:
            raise AgentError("No Character be roleplayed.")

        prompt = self.config.character.system_prompt

        if metadata := self.config.character.metadata:
            prompt += "\n\n【角色档案】\n"
            for key, value in metadata.items():
                prompt += f"- {key}: {value}\n"

        return prompt

    def _create_background_manager(self) -> BackgroundManager:
        """创建后台管理器"""
        manager = BackgroundManager(max_workers=2, max_queue_size=50)

        # 注册工作器
        manager.register_memory_worker(
            MemoryWorker(self.semantic_memory, self.config.memory)
        )
        manager.register_persistence_worker(
            PersistenceWorker(self.session_manager._persistence)
        )

        # 注册完成回调
        manager.on_complete(self._on_background_task_complete)

        # 注入依赖
        self.save_coordinator.set_background_manager(manager)
        self.response_handler.set_background_manager(manager)

        # 设置到生命周期管理器
        self.lifecycle.set_background_manager(manager)

        return manager

    # ==================== 保存相关 ====================

    def _sync_save(self) -> None:
        """同步保存（确保数据写入磁盘）"""
        self.save_coordinator.sync_save(self.working_memory)

    async def _async_save(self) -> None:
        """异步保存（用于关闭回调）"""
        # 在线程池中执行同步保存
        await asyncio.to_thread(self._sync_save)

    async def _start_background_manager(self) -> None:
        """启动后台管理器"""
        if not self.lifecycle.is_shutting_down:
            task = asyncio.create_task(self.background_manager.start())
            self.lifecycle.set_background_task(task)
            logger.debug("后台管理器已启动")

    async def _on_background_task_complete(self, task: TaskResult) -> None:
        """后台任务完成回调"""
        if not task.success:
            logger.warning(f"后台任务失败 [{task.task_id}]: {task.error}")

        # 委托给保存协调器
        operation = task.result.get("operation") if task.result else None
        self.save_coordinator.on_task_complete(operation)

    # ==================== 核心 API ====================

    async def send(self, user_input: str) -> Message | None:
        """发送消息并获取完整回复（非流式）"""
        if self.lifecycle.is_shutting_down:
            return None

        # 使用信号量限制并发
        async with self._request_semaphore:
            request_id = get_request_id()
            logger.debug(f"[{request_id}] 开始处理请求")

            try:
                self.lifecycle.set_current_task(asyncio.current_task())
                return await self._do_send(user_input)
            except asyncio.CancelledError:
                logger.info(f"[{request_id}] 请求被取消")
                raise
            finally:
                self.lifecycle.set_current_task(None)
                logger.debug(f"[{request_id}] 请求处理完成")

    async def _do_send(self, user_input: str) -> Message:
        """实际执行发送逻辑"""
        # 确保后台管理器已启动
        await self._start_background_manager()

        # 记录用户消息
        self.response_handler.record_user_message(user_input)

        # 构建消息
        messages = self.message_builder.build(user_input)
        tools = self.tool_registry.get_schemas() if self.config.tool.enabled else None

        # 委托给 ResponseHandler 处理
        return await self.response_handler.process_non_stream(
            user_input, messages, tools
        )

    async def send_stream(self, user_input: str) -> AsyncIterator[StreamChunk]:
        """发送消息并获取流式回复迭代器"""
        if self.lifecycle.is_shutting_down:
            return

        async with self._request_semaphore:
            request_id = get_request_id()
            logger.debug(f"[{request_id}] 开始处理流式请求")

            try:
                self.lifecycle.set_current_task(asyncio.current_task())
                async for chunk in self._do_send_stream(user_input):
                    yield chunk
            except asyncio.CancelledError:
                logger.info(f"[{request_id}] 流式请求被取消")
                raise
            finally:
                self.lifecycle.set_current_task(None)
                logger.debug(f"[{request_id}] 流式请求处理完成")

    async def _do_send_stream(self, user_input: str) -> AsyncIterator[StreamChunk]:
        """实际执行流式发送逻辑"""
        # 确保后台管理器已启动
        await self._start_background_manager()

        # 记录用户消息
        self.response_handler.record_user_message(user_input)

        # 构建消息
        messages = self.message_builder.build(user_input)
        tools = self.tool_registry.get_schemas() if self.config.tool.enabled else None

        # 委托给 ResponseHandler 处理
        async for chunk in self.response_handler.process_stream(
            user_input, messages, tools
        ):
            if self.lifecycle.is_shutting_down:
                break
            yield chunk

    # ==================== 会话管理 ====================

    def rollback(self, turns: int = 1) -> None:
        """回滚对话"""
        wm = self.working_memory
        for _ in range(turns * 2):
            if wm._memory.messages:
                wm._memory.messages.pop()

    def create_session(self) -> SessionContext:
        """创建新会话"""
        if self.config.session.auto_save:
            self._sync_save()

        session = self.session_manager.create_session()
        self._working_memory = None  # 清除缓存，让 property 重新加载
        self.save_coordinator.reset()

        return session

    def resume_session(self, session_id: str) -> bool:
        """恢复会话"""
        if self.config.session.auto_save:
            self._sync_save()

        if self.session_manager.set_current_session(session_id):
            self._working_memory = None  # 清除缓存，让 property 重新加载
            self.save_coordinator.reset()
            return True
        return False

    def save_session(self) -> None:
        """保存当前会话"""
        self._sync_save()

    async def shutdown(self) -> None:
        """关闭 Agent"""
        # 创建异步任务执行关闭
        await self.lifecycle.shutdown()

        logger.info("Agent 已关闭")

    async def wait_for_shutdown(self, timeout: float = 5.0) -> None:
        """等待关闭完成"""
        await self.lifecycle.wait_for_shutdown(timeout)
