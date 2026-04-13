"""保存协调器 - 管理异步保存的去重和状态"""

# GenskoyoAI/core/agent/save_coordinator.py

import asyncio
from typing import TYPE_CHECKING

from ...utils.logging import logger
from ...background import BackgroundManager, TaskPriority

if TYPE_CHECKING:
    from ...session.manager import SessionManager
    from ...memory.working import WorkingMemoryManager
    from ...core.config import SessionConfig


class SaveCoordinator:
    """
    保存协调器 - 管理异步保存的去重和状态

    职责：
    - 判断是否应该保存（去重逻辑）
    - 管理 pending 状态
    - 提交保存任务到后台管理器
    - 处理保存完成回调
    """

    def __init__(
        self,
        session_manager: "SessionManager",
        session_config: "SessionConfig",
    ):
        """
        初始化保存协调器

        Args:
            session_manager: 会话管理器
            session_config: 会话配置
        """
        self._session_manager = session_manager
        self._session_config = session_config

        # 去重状态
        self._save_pending = False
        self._last_saved_turn = 0

        # 后台管理器引用（延迟注入）
        self._background_manager: BackgroundManager | None = None
        self._bg_started = False

        # 关闭状态（外部传入）
        self._shutting_down = False

    def set_background_manager(self, manager: BackgroundManager) -> None:
        """注入后台管理器"""
        self._background_manager = manager

    def set_shutting_down(self, value: bool) -> None:
        """设置关闭状态"""
        self._shutting_down = value

    @property
    def save_pending(self) -> bool:
        """是否有保存任务 pending"""
        return self._save_pending

    @property
    def last_saved_turn(self) -> int:
        """最后保存的轮数"""
        return self._last_saved_turn

    def reset(self) -> None:
        """重置状态（新会话时调用）"""
        self._save_pending = False
        self._last_saved_turn = 0

    def should_save(self, working_memory: "WorkingMemoryManager") -> bool:
        """
        判断是否应该保存（去重）

        Args:
            working_memory: 工作记忆管理器

        Returns:
            True 表示应该保存
        """
        # 关闭时强制保存
        if self._shutting_down:
            return True

        current_turn = len(working_memory) // 2

        # 如果没有新消息，不保存
        if current_turn <= self._last_saved_turn:
            return False

        # 如果已有保存任务在队列中，不重复提交
        if self._save_pending:
            logger.debug("保存任务已在队列中，跳过")
            return False

        return True

    def mark_pending(self, working_memory: "WorkingMemoryManager") -> None:
        """
        标记保存任务为 pending

        Args:
            working_memory: 工作记忆管理器
        """
        self._save_pending = True
        self._last_saved_turn = len(working_memory) // 2

    def on_task_complete(self, operation: str | None = None) -> None:
        """
        任务完成回调

        Args:
            operation: 操作类型
        """
        if operation == "save_messages":
            self._save_pending = False
            logger.debug("保存任务完成，重置 pending 状态")

    async def start_background_manager(self) -> None:
        """启动后台管理器（幂等）"""
        if self._background_manager is None:
            logger.warning("后台管理器未注入，无法启动")
            return

        if not self._bg_started and not self._shutting_down:
            asyncio.create_task(self._background_manager.start())
            self._bg_started = True
            logger.debug("后台管理器已启动")

    async def save_async_if_needed(
        self,
        working_memory: "WorkingMemoryManager",
    ) -> bool:
        """
        异步保存（如果需要）

        Args:
            working_memory: 工作记忆管理器

        Returns:
            True 表示提交了保存任务
        """
        # 检查是否启用自动保存
        if not self._session_config.auto_save:
            return False

        # 检查是否应该保存
        if not self.should_save(working_memory):
            return False

        # 标记 pending
        self.mark_pending(working_memory)

        # 获取当前会话
        current_session = self._session_manager.get_current_session()
        if current_session is None:
            self._save_pending = False
            return False

        # 确保后台管理器已启动
        await self.start_background_manager()

        if self._background_manager is None:
            self._save_pending = False
            return False

        # 获取消息
        messages = working_memory.get_context()

        # 提交持久化任务
        self._background_manager.submit_persistence_task(
            operation="save_messages",
            data={
                "session_id": current_session.session_id,
                "messages": messages,
            },
            priority=TaskPriority.LOW,
            timeout=10.0,
        )

        logger.debug(f"已提交异步保存任务 (轮数: {len(messages) // 2})")
        return True

    def sync_save(self, working_memory: "WorkingMemoryManager") -> None:
        """
        同步保存（用于关闭时）

        Args:
            working_memory: 工作记忆管理器
        """
        try:
            if self._session_config.auto_save:
                current = self._session_manager.get_current_session()
                if current:
                    self._session_manager.save_working_memory()
                    self._session_manager.save_current()
                    logger.info(f"已保存会话数据 (轮数: {len(working_memory) // 2})")
        except Exception as e:
            logger.error(f"保存数据时出错: {e}")
