# Copyright (c) OpenMMLab. All rights reserved.

# TODO(intern-decision): [09] 此文件仅记录待实现的测试，不表示已有测试或验证通过。
# 输入一致性：使用同一 checkpoint 和样例，比对原协议与迁入后的字段 / 选项顺序、
# 模板文本、input_ids、marker-1、候选 ID、图片 patch / grid / offset 及 MRoPE 位置。
# 包含纯文本、多图片、三种问题类型与非法输入；分别说明不变式与可接受的数值误差。
# 输出一致性：验证合法候选集合内 softmax、choice 的稳定平局规则、score 的加权期望、
# noul 的 yes 概率，以及概率 / 置信度字段；不能只写重复实现逻辑的自证测试。
# 原生入口闭环：仅安装 LMDeploy 并提供 checkpoint，验证 Pipeline.decide 与 HTTP 返回一致；
# 确认没有 import 原 Intern-Decision 项目的 src，也不需要外置 LMDeployBackend。
# 接口回归：验证普通聊天、奖励评分、原 async_get_logits 调用及模型架构选择仍正常。
# 资源生命周期：覆盖成功、模型异常、客户端断开、任务取消、推理超时、清理超时及并发失败；
# 检查 END_SESSION 确认、handle 归还、session 移除及缓存释放，覆盖 step==0 与持有 handle 场景。
# 数值验收：在可用硬件上对齐 HF / LMDeploy 的候选 logits 和最终概率，记录 dtype 与误差阈值；
# 没有 checkpoint / GPU 时明确跳过相应集成测试，不能把 CPU 协议测试当作模型数值验收。
