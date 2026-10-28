# 两周 CUDA 面试入门计划（每天约 60 分钟）

目标：能独立写出简单 FP32 CUDA 算子，验证正确性，做基础性能比较，并解释一次有数据支撑的优化。14 小时用于建立入门能力，不承诺达到专业算子工程师水平。默认已了解 C/C++ 指针、数组、Python 基础；不熟悉时优先保证加法、归约、转置三项，矩阵乘法做到理解即可。

每天固定节奏：10 分钟阅读 → 35 分钟编码/实验 → 10 分钟验证/记录 → 5 分钟口述复盘。环境安装和首次扩展编译可提前完成，不占每日练习时间。每天只推进一个主题，超过一小时先记录卡点。

所有路径相对项目根目录；原项目作为参考，自己的代码放在 `learning/exercises/dayXX/`。不要一开始整份照抄优化代码，先自己写最小正确版本。参考答案 `learning/exercises/add.cu` 在需要时查看。

| 天 | 阅读入口与重点 | 35 分钟练习 | 当天验收与口述题 |
|---|---|---|---|
| 1 | `samples/deviceQuery/`；host/device、grid/block/thread | 在 learning 下运行 probe；自己写一个 32 线程填数组 kernel | 记录 GPU、SM 8.6、4GB 显存；解释线程索引和 kernel launch |
| 2 | `kernels/elementwise/elementwise.cu` 的标量版 | 不看答案写 FP32 add，处理 N=0/1/257/1025；与 torch 对比 | 正确性通过；解释为何需要 `i<N`、为何 CUDA 调用异步 |
| 3 | `learning/exercises/bench_add.py` | 加入 CUDA Event 计时、预热；比较 128/256/512 线程，N 从 1K 到 4M | 保存时间和 shape；解释为什么不能只用 Python wall time、首次编译不计入 |
| 4 | `kernels/elementwise/elementwise.cu` 的 float4；`kernels/relu/` | 写 float4 add 并处理尾部/地址对齐；有余力写 ReLU | 奇数长度和偏移切片正确；解释合并访存、向量化为什么未必更快 |
| 5 | `kernels/reduce/reduce_sum.cu`；先看归约结构 | 从每 block 一行开始，写 shared-memory sum，列长先固定 256 | 与 torch.sum 比较并使用浮点容差；解释 `__syncthreads()`、跨 block 不能这样同步 |
| 6 | 同上，warp shuffle 部分 | 将 block sum 改为 warp 内 shuffle，再归约各 warp 结果 | 固定整 warp block，越界 lane 输入 0；解释 mask、warp=32、为何浮点结果可不同 |
| 7 | 复习 2–6 天 | 从空文件重写 add；修复一个归约 bug；整理首周笔记 | 20 分钟写 add；口述“正确性→瓶颈猜测→修改→测量”；来不及则归约停在 shared 版 |
| 8 | `kernels/mat_transpose/transpose.cu` | 写 naive transpose 和 32×32 shared tile；可用 32×8 线程搬多行 | 32×32、256×512、257×513 正确；解释读写哪一端不合并 |
| 9 | 同上 `transpose_smem_bcf`；`samples/bank_conflict_ncu/` | 比较 shared tile `[32][32]` 与 `[32][33]`；可运行 ncu | 记录时间；画出列访问 bank 映射；性能计数器不可用时用代码解释+Event 数据 |
| 10 | `kernels/softmax/softmax.cu`，只理解基础稳定公式 | 写每 block 一行的 FP32 softmax：max → exp(x-max) → sum → 除法；行长先 128/256/1024 | 与 torch.softmax 比较；测试 ±1000、大量相同值；说明减 max 防溢出，暂不支持 NaN/Inf |
| 11 | `kernels/sgemm/sgemm.cu` 中最简单 SIMT 版本 | 写 naive FP32 GEMM，先 128×128；使用 torch.mm 作参考 | 非方阵和非 tile 整除 shape 正确；解释 C[m,n] 的 K 维循环 |
| 12 | 同上 shared-memory 分块思路 | 写 16×16 shared tiled GEMM，加载越界补零，所有线程参与同步 | 比较 naive/tiled/torch，记录 M/N/K；解释数据复用和同步，性能不要求超过 cuBLAS |
| 13 | 选 add、transpose 或 softmax 之一 | 做一次完整优化实验：三个 shape，baseline/优化、正确性、时间、硬件；整理一页报告 | 能解释收益来源、无收益的情况和限制；不要把别人的速度当成自己的结果 |
| 14 | `learning/INTERVIEW.md` | 20 分钟默写 add 或 block reduce；20 分钟讲解实验；15 分钟问答，5 分钟列知识缺口 | 形成一个可展示目录、一页优化报告、2 分钟项目描述 |

## 范围和取舍

必会：索引与边界、global/shared/register 基本区别、合并访存、同步、warp shuffle 基础、正确计时、正确性对照。能讲清楚：memory-bound/compute-bound、bank conflict、occupancy 不等于性能、融合减少中间访存。

两周内不要求：PTX、Tensor Core MMA、cp.async/TMA、FlashAttention、SM120、复杂 HGEMM、DSL、量化和多 GPU。RTX 3050 是 SM86；原 README 的 RTX 5060 测试结论不能直接套用。源文件名和参考范围来自当前本地项目，不表示所有高级算子均适配此环境。

## 实验规范

- 先小 shape 对照 torch；覆盖 0、1、非整除、不同形状。自写接口明确 dtype、连续性、shape、device；不支持的输入显式报错。
- FP32 普通算子默认 `rtol=1e-5, atol=1e-6`；归约/GEMM根据累加长度调整并记录误差，不用浮点完全相等。GEMM 比较时设置 `torch.backends.cuda.matmul.allow_tf32=False`，避免把精度变化当优化。
- 输出提前分配；计时不含 JIT 编译/随机初始化/CPU-GPU 拷贝；使用当前 stream 的 CUDA Event，预热、多次重复取中位数。小算子可能包含 Python 提交造成的 GPU 空闲。
- 示例是热缓存基准；有效带宽 = add 的 `3*N*4 / 秒 / 1e9`，不能直接当显存实测带宽。cache 和计时方式不同，不直接比较项目 cold-L2 benchmark。
- 每次只改一个因素；记录 GPU、shape、dtype、block、时间、误差、重复次数。笔记记录失败优化也有价值。
- 4GB 显存从 1K–4M 元素、矩阵 128–1024 开始，逐项执行。先不要一键运行全部原项目 benchmark。

## 每日笔记模板

日期 / 用时：
今天能解释的概念：
代码路径与运行命令：
正确性覆盖与最大误差：
shape / dtype / block / baseline ms / optimized ms / speedup：
现象 → 原因猜测 → 证据：
明天第一步：

## 第 14 天过关标准

1. 20 分钟内从空文件写出并验证一个简单 add/ReLU kernel。
2. 能解释并实现至少一种 block reduction；能解释 shared tiled transpose 或 GEMM。
3. 有至少一个包含正确性、三组 shape、可复现命令和局限的调优报告，允许部分 shape 不加速。
4. 能解释 thread/block/warp、合并访存、同步、bank conflict、CUDA Event、误差容差。

这些是面试入门证据，简历措辞应为“实现并验证基础 CUDA 算子，完成线程块/访存等基础调优”，仅填写真实测得的加速数字。
