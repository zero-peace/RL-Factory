from .config_manager import ConfigManager
from .qwen3_manager import QwenManager
from .qwen2_5_manager import Qwen25Manager
from .centralized_qwen3_manager import CentralizedQwenManager

__all__ = ['ConfigManager', 'QwenManager', 'CentralizedQwenManager']

TOOL_MANAGER_REGISTRY = {
    'config': ConfigManager,
    'qwen3': QwenManager,
    'qwen2_5': Qwen25Manager,
    'centralized_qwen3': CentralizedQwenManager,
}