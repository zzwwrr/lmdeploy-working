# Copyright (c) OpenMMLab. All rights reserved.
from .multimodal import MultimodalProcessor

# TODO(intern-decision): [01] 任务启用后再加载 intern_decision 内的输入/输出处理器，
# 由共享 AsyncEngine 持有；具体契约见该目录的 protocol.py，不能依赖外部仓库 src.*。
__all__ = ['MultimodalProcessor']
