"""网络包 ``nets``：CalibFuse 的模型定义。

对外只暴露主模型类 :class:`~nets.fusion.CalibFuse`；字典、交互、
Restormer 构件等子模块请从各自文件导入（或经 ``nets.fusion`` 间接
引用）。保持 ``__all__`` 最小化，避免无意中依赖内部实现细节。
"""

from .fusion import CalibFuse

__all__ = ["CalibFuse"]
