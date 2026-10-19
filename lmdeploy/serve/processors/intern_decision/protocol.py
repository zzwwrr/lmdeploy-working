# Copyright (c) OpenMMLab. All rights reserved.

# TODO(intern-decision): [02] 定义 DecisionRequest / PreparedDecision / DecisionResponse，
# 供 Python 与 HTTP 共用；这里只规划数据契约，不创建模型或 FastAPI 应用。
# DecisionRequest 沿用 Intern-Decision 的 state、questions 和可选 images，问题类型为
# choice / score / noul；HTTP 图片兼容原来的 data/base64 协议，不接收任意服务器路径。
# 输入校验保留 1..16 个字段、每字段至多 62 个选项、至多 8 张静态图片及长度限制；
# 字段与选项保持原始顺序，推理时忽略训练 answer/targets，拒绝保留标记注入与静默截断。
# HTTP 的 model 选择、请求标识和错误响应与 LMDeploy 服务协议衔接；不把 questions
# 转成普通聊天 messages 后交给生成接口。thinking.enabled=true 在首版明确不支持。
#
# PreparedDecision 拟包含：input_ids、multimodal、read_positions（marker-1）、
# candidate_ids、fields、option_values 和 input_length；每个字段的业务值/符号/ID
# 在输入阶段固定，输出阶段直接使用该映射，避免重新排序或重新编码候选。
#
# DecisionResponse 保留 answers 中的 type / probabilities / confidence 和
# choice / score / noul 等业务语义；兼容外层 decision/source、request_id/model 字段。
# 显式定义 usage：decision_count 是字段数，实际生成 token 数为 0；原协议若保留
# output_tokens=字段数的兼容字段，需说明口径，不能误报成 LMDeploy 生成 token 数。
# 首版不携带原项目的外部 thinking 服务、浏览器演示或训练监督逻辑。
