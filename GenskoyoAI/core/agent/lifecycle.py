"""生命周期管理器 - 处理启动、关闭、信号等"""

# GenskoyoAI/core/agent/lifecycle.py

import asyncio
import signal
import sys
import signal as sig
from typing import Callable, Awaitable

from ...utils.logging import logger
from ...background.manager import BackgroundManager


class LifecycleManager:
    """
    生命周期管理器 - 处理启动、关闭、信号等

    职责：
    - 设置信号处理器（SIGINT, SIGTERM）
    - 管理关闭状态
    - 优雅关闭流程
    - Windows 平台的信号处理兼容
    """

    def __init__(self, on_shutdown: Callable[[], Awaitable[None]] | None = None):
        """
        初始化生命周期管理器

        Args:
            on_shutdown: 关闭时的回调函数（用于保存数据等）
        """
        self._shutting_down = False
        self._shutdown_event = asyncio.Event()
        self._on_shutdown = on_shutdown

        # 后台任务引用
        self._bg_task: asyncio.Task | None = None
        self._background_manager = None

        # 当前请求任务
        self._current_task: asyncio.Task | None = None

        # 子组件（用于传递关闭状态）
        self._components: list = []

    # ==================== 状态管理 ====================

    @property
    def is_shutting_down(self) -> bool:
        """是否正在关闭"""
        return self._shutting_down

    @property
    def shutdown_event(self) -> asyncio.Event:
        """关闭事件"""
        return self._shutdown_event

    def set_shutting_down(self, value: bool) -> None:
        """设置关闭状态，并通知所有注册的组件"""
        self._shutting_down = value
        for component in self._components:
            if hasattr(component, "set_shutting_down"):
                component.set_shutting_down(value)

    def register_component(self, component) -> None:
        """
        注册需要接收关闭状态的组件

        Args:
            component: 具有 set_shutting_down 方法的组件
        """
        if hasattr(component, "set_shutting_down"):
            self._components.append(component)

    # ==================== 后台任务管理 ====================

    def set_background_task(self, task: asyncio.Task | None) -> None:
        """设置后台任务引用"""
        self._bg_task = task

    def set_background_manager(self, manager: BackgroundManager) -> None:
        """设置后台管理器引用"""
        self._background_manager = manager

    def set_current_task(self, task: asyncio.Task | None) -> None:
        """设置当前请求任务"""
        self._current_task = task

    # ==================== 信号处理 ====================

    def setup_signal_handlers(self) -> None:
        """设置信号处理器"""
        try:
            loop = asyncio.get_running_loop()
            for sig_num in (signal.SIGINT, signal.SIGTERM):
                loop.add_signal_handler(
                    sig_num,
                    lambda s=sig_num: asyncio.create_task(self._handle_signal(s)),
                )
            logger.debug("信号处理器已设置")
        except NotImplementedError:
            # Windows 不支持 add_signal_handler
            logger.debug("当前平台不支持 add_signal_handler")
            self._setup_windows_signal_handler()

    def _setup_windows_signal_handler(self) -> None:
        """Windows 平台的信号处理"""

        def windows_handler(signum, frame):
            if not self._shutting_down:
                self._shutting_down = True
                logger.info("收到中断信号，正在保存数据...")

                # 同步执行关闭回调
                if self._on_shutdown:
                    try:
                        # 在 Windows 信号处理器中不能运行异步代码
                        # 所以这里需要特殊处理，或者只做同步保存
                        logger.warning("Windows 下请使用 Ctrl+Break 或等待程序自然退出")
                    except Exception as e:
                        logger.error(f"关闭回调执行失败: {e}")

                logger.info("正在退出...")
                sys.exit(0)

        sig.signal(signal.SIGINT, windows_handler)
        sig.signal(signal.SIGTERM, windows_handler)

    async def _handle_signal(self, signum: int) -> None:
        """异步处理信号"""
        if self._shutting_down:
            return

        self.set_shutting_down(True)
        signal_name = signal.Signals(signum).name

        logger.info(f"收到 {signal_name} 信号，正在优雅关闭...")

        # 取消当前请求
        if self._current_task and not self._current_task.done():
            self._current_task.cancel()

        # 执行优雅关闭
        await self._graceful_shutdown()

        # 执行关闭回调
        if self._on_shutdown:
            try:
                await self._on_shutdown()
            except Exception as e:
                logger.error(f"关闭回调执行失败: {e}")

        logger.info("正在退出...")
        self._shutdown_event.set()
        sys.exit(0)

    async def _graceful_shutdown(self) -> None:
        """优雅关闭"""
        # 取消后台任务
        if self._bg_task and not self._bg_task.done():
            self._bg_task.cancel()
            try:
                # 使用 asyncio.shield 防止任务被意外取消导致状态异常
                await asyncio.shield(self._bg_task)
            except (asyncio.CancelledError, asyncio.InvalidStateError):
                pass
            except Exception as e:
                logger.debug(f"后台任务关闭时出现异常: {e}")

        # 停止后台管理器
        if self._background_manager:
            try:
                await self._background_manager.stop(wait=True)
            except Exception as e:
                logger.debug(f"后台管理器停止时出现异常: {e}")

    # ==================== 关闭流程 ====================

    async def shutdown(self) -> None:
        """主动关闭"""
        if self._shutting_down:
            return

        self.set_shutting_down(True)
        await self._graceful_shutdown()

        if self._on_shutdown:
            await self._on_shutdown()

        self._shutdown_event.set()

    async def wait_for_shutdown(self, timeout: float = 5.0) -> None:
        """等待关闭完成"""
        try:
            await asyncio.wait_for(self._shutdown_event.wait(), timeout)
        except asyncio.TimeoutError:
            pass
