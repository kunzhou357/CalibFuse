"""CalibFuse 模型定义包。

只暴露主模型类 :class:`~nets.fusion.CalibFuse`；
字典模块（``nets.dictionary``）与 Restormer 基础模块
（``nets.restormer``）请从各自子模块导入。
"""

from .fusion import CalibFuse

__all__ = ["CalibFuse"]
