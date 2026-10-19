# Copyright (c) OpenMMLab. All rights reserved.

# TODO(intern-decision): [03] 实现 DecisionInputProcessor：DecisionRequest -> PreparedDecision。
# 移植来源为 Intern-Decision 的 src/inputs/schema.py 中固定模板、_options、compile_row，
# 以及 src/inference/encoding.py / lmdeploy_inputs.py；移植时保留来源、版本及许可说明，
# 将代码收进 LMDeploy，不能在运行时 import 外部 src.* 或加载 Transformers 模型权重。
#
# 处理顺序：请求校验 -> 业务选项映射为 A/B/C/... -> 构造带 <decision> 的 assistant
# JSON 骨架 -> checkpoint 自带 tokenizer/processor 编码 -> 定位每个 marker 的前一位置。
# 保留训练协议的 system prompt、JSON 格式、字段/候选顺序、图片在 user 文本前的排列，
# 以及 add_generation_prompt=False / enable_thinking=False / add_vision_id=True。
# 通用 HFChatTemplate.messages2prompt 会裁剪 assistant_end，不能直接替代这段编码；
# 需要逐 token 与原编码器核对，不以“提示词看起来一样”作为兼容依据。
#
# 图片的传输/读取与编译分开：HTTP 解码受限的 data/base64，Python 本地图片按明确
# 契约读取；可复用现有图片工具，但尺寸、RGB/方向处理及图片顺序须与原行为一致。
# 已解码图片在 processor 执行后释放；CPU 工作不能长期阻塞服务事件循环。
# 编码结果保持 CPU Tensor，验证 <decision> 已注册、候选是单 token、marker 数量匹配。
#
# 把 pixel_values / image_grid_thw 转为 LMDeploy 每图字典：pixel_values、
# image_grid_thw、offset、image_tokens、image_token_id。按未合并 T*H*W 切 patch，
# 按 T*H*W/merge**2 校验占位 token；首版保持静态图 T=1、merge=2 等验证边界。
# 核对 vision_start/end、图片数量和 patch 总量；纯文本返回 multimodal=None。
# 后续复用原生 Qwen3.5 InputProcessor、视觉网络及 MRoPE，不在这里计算视觉 embedding。
