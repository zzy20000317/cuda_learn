# 首周复盘与可展示成果

## 索引与访存

`i=blockIdx.x*blockDim.x+threadIdx.x` 确定每个线程的起始元素；grid-stride 让它循环处理后续元素。边界判断保护非整除长度；n=0 时 host 跳过 kernel。连续切片仍可能有非零 storage offset，标量加载不依赖 float4 的 16 字节对齐。

同一 warp 的 lane 访问相邻 global 元素，有利于合并访存；线程块越大不保证越快。读写效率、寄存器、调度、输入规模与提交开销共同影响性能。浮点加法结合顺序变化可能改变结果。

## Warp 与 block 归约

shuffle 只在一个 warp 内传递值；lane0 的 warp sum 不是 block sum。block=256 有 8 个 warp，需要 shared memory 中转和 block 屏障，然后 warp0 才能合并各 warp 的结果。warp0 中只有前 8 个 lane 有输入，其余 lane 输入 0，但全部 32 lane 都参加同一个 mask 的 shuffle。

`__syncthreads()` 是 block 屏障。跨 block 的最终合并可以用下一次 kernel launch 或 atomicAdd；原子加之前输出必须清零。atomicAdd 不等于全 grid barrier。现代 GPU 的 shared-memory warp 通信也需要适当同步，volatile 不能代替屏障。

## 一次可复现 bug 修复

1. 最小输入：两行，每行 256 个 1。
2. 现象：每行得到 32；错误稳定复现，排除随机浮点误差。
3. 代码定位：只将第一个 warp 的结果写到输出，其余 7 个 warp 丢失。
4. 修复：warp lane0 写 shared 数组，block 同步，warp0 读取 8 个结果再 shuffle。
5. 验证：精确输入 256；另测空行、1/31/257/1025 列、随机输入和非默认 stream。

## 正确性 → 瓶颈猜测 → 修改 → 测量

先明确 shape/dtype/布局/stream，构造边界与对照结果；再用代码分析提出瓶颈假设。每次只改一项，用预热后的 CUDA Event 测量，多次重复。编译/分配/拷贝是否包含必须写清楚。当前 add 比较 128/256/512 线程，实际数字见 RESULTS.md；无收益也是实验结果。

## 限时自测与口述

20 分钟在新文件写 add 的索引、边界、host 检查及 torch 对照，不看 review.cu。
随后回答：为什么 warp sum 不是 block sum？为什么所有参与 lane 必须使用一致 mask？为什么随机归约不能要求位级相等？为何只测 Python wall time 容易误判 GPU 时间？什么时候增加线程块大小反而没有收益？

本笔记是辅助复盘材料，不代表学习者已独立通过限时编程或面试口述。
