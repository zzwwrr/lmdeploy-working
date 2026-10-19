# Copyright (c) OpenMMLab. All rights reserved.

# TODO(intern-decision): [04] 实现 DecisionOutputProcessor：PreparedDecision + 决策 logits
# -> DecisionResponse；从 Intern-Decision src/inference/engine.py 的 predict 后处理提取。
# 此处只做输出转换，不加载模型、不建立 session、不调用任何远程 thinking 服务。
#
# 使用输入中固定的 fields / candidate_ids / option_values：先校验 Q 行及有限分数，
# 再为每字段取合法候选列，在候选集合内用 float32 softmax 得到 probabilities。
# choice 取最高概率业务值，平局保留原来的字典序规则；score 返回数值选项的概率
# 加权期望及 legend；noul 返回 yes 的概率，confidence 保持候选最大概率语义。
# 保留 decision/source 等响应包装语义，返回结构化结果而非生成并解析一段 JSON 文本。
#
# 如支持温度校准，单独迁入 temperature.py 的相关逻辑并明确校准产物与 backend 的
# 绑定；未经验证不能默认沿用 HF 的温度。概率与校准回归不能只检查 argmax 相同。
# usage/timing 的口径由协议定义并与 HTTP/Python 一致，尤其区分决策字段数与生成 token 数。
