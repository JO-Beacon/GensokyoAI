"""自定义异常"""
# GensokyoAI\core\exceptions.py


class GensokyoError(Exception):
    """基础异常类"""

    pass


class ConfigError(GensokyoError):
    """配置错误"""

    pass


class AgentError(GensokyoError):
    """Agent 错误"""

    pass


class MemorySystemError(GensokyoError):
    """记忆系统错误"""

    pass


class ToolError(GensokyoError):
    """工具系统错误"""

    pass


class SessionError(GensokyoError):
    """会话错误"""

    pass


class ModelError(GensokyoError):
    """模型调用错误"""

    pass
