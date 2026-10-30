# Dense C1 H100 推理优化记录（2026-10-31）

范围：单卡完整 A 榜，同一原始 Dense C1 权重与数据。每个方案使用独立搜索的最大稳定 batch。权重不改写，无 MoE expert/router 专项优化；Dense C4 尚无权重，H200 尚无测试机。

## 已验证结果

|方案|最大 batch|EvalTime|bpb|
|---|---:|---:|---:|
|原始 BF16|926|24.72494|1.12129183|
|原通用 FP8|942|21.49954|1.12341596|
|输出融合 + 预量化 + 输入生命周期修复|942|15.16911|1.12341569|
|上述 + BF16 残差存储|942|14.63642|1.12342790|
|上述 BF16 残差 + cuDNN BF16 attention|942|13.85378|1.12342276|
|全部组合 + FlashQLA D64|942|**11.67973**|**1.12342475**|

各行是完整 A 榜；中间阶段并非彼此单一变量的严格消融。最后组合 batch943 OOM；bpb 相比 BF16 增加0.00213293，低于0.03上限。EvalTime 较原通用FP8减少45.7%，较原BF16减少52.8%。最终提交入口已在另一张H100独立复核：EvalTime **11.63638934476569**，bpb **1.123424751074623**，forward **47.22010002136232秒**，32步。bpb与首轮基本逐位一致；两次耗时约0.37%的差异视为运行波动。详见evidence/gpu7-run-metadata.json。

## 采用的优化

- 普通 Linear 与 Dense SwiGLU 原生 E4M3 W8A8；投影输入量化复用，融合中间激活和量化。
- attention 输出布局、门控与后续 FP8 量化融合，减少大张量拷贝。
- 预量化 residual/RMSNorm producer；残差持久存储 BF16，加法、归约仍 FP32。
- eval 循环及时释放 inputs/labels/batch，消除下一轮分配与旧输入共存造成的 allocator retries。未改计时边界、loss 或数据。
- 全注意力强制 cuDNN BF16 causal GQA 后端，profiling 确认真实路由，禁止静默 fallback。
- GDN 核心采用 FlashQLA0.1.3 的隔离 D64 特化，保持原归一化、scale、BF16核心/FP32状态；限定修改两处 guard/block_DV，auto_cp关闭。单元比较、memcheck、源文件 guard 与完整模型质量验证共同约束。

完整词表 logits 保留 BF16。EvalTime 包含生成全词表 logits 的 forward；不含预热、packing、H2D、CE。完整计分120142976 tokens、445910984 UTF-8 bytes，Na=33776160。

## 已尝试但未采用

- FP8 logits：节省显存，batch可到1500以上，bpb在容忍范围；但满batch完整A更慢。按用户要求保存代码、停用。
- 单形状 CUDA Graph：本模型已大batch饱和，收益很小，另增加图内存和尾批处理问题，未选。
- torch.compile fullgraph：隔离包装原FLA后完成B1数值smoke，但编译结果与eager有数值差异，未完成bpb/满batch资格，不能当作提速结果。
- DeepGEMM：真实权重FFN组件初筛约1.06–1.19倍，包含不同量化配方与控制组漂移；不是全模型收益，未选入提交。
- FA4 BF16：独立算子可运行，模型短测相比cuDNN收益很小；未增加正式依赖。
- cuDNN FP8 attention：实际Hopper路径已尝试，同量化策略数值参考尚有未解差异，未选。
- SageAttention纯Triton1.0.6：小形状可运行，实际batch942非法显存访问，未进入模型评测。官方SM90 INT8 QK+FP8 PV源码路径缺匹配工具链/wheel，未绕过版本检查。
- FP8残差存储：短测速度收益不足，未选。

## 发布与H200口径

正式评测使用我们的脚本。发布入口支持CLI batch > EVAL_BATCH_SIZE > 自动搜索；不固定H100的942，不使用pilot序列数限制H200容量。CPU外层调度，新进程容量探测和正式完整评测；非OOM错误不能当成容量失败。

H200满显存实际batch与EvalTime仍需实机确定。之前预留10%显存的估计已作废，不能按峰值TFLOPS倍率直接缩放整个模型，也不能把当前C1结果称为C4实测。

## 证据与公开一手资料

本PR随附evidence/中的两次独立完整A结果、容量/profiling摘要和FlashQLA数值/memcheck证据。原始实验日志由实验归档保留。

- [NVIDIA Transformer Engine attention](https://docs.nvidia.com/deeplearning/transformer-engine/examples/attention/attention.html)
- [FlashQLA 官方实现](https://github.com/QwenLM/FlashQLA)
- [FlashAttention 官方实现](https://github.com/Dao-AILab/flash-attention)
- [DeepGEMM 官方实现](https://github.com/deepseek-ai/DeepGEMM)
- [PyTorch 2.8 CUDA Graph 内存说明](https://docs.pytorch.org/docs/2.8/notes/cuda.html#graph-memory-management)
- [SageAttention 官方实现](https://github.com/thu-ml/SageAttention)

公开benchmark仅用于形成实验假设；上述模型成绩均来自本机完整A，未使用论文倍率替代实测。
