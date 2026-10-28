# day05 实现与测试记录

已实现 reduce0～reduce7，另加 atomic_shuffle 作为两阶段/原子合并对照。
参考学习笔记为 reduce_learn.md；该学习笔记未修改。

## 正确性

使用 zzy 用户运行 ./test.sh，通过 13 组配置：

- block=128/256/512/1024 × grid-cap=1/17/256，N=65539。
- 默认 block/grid，N=1048579。
- 每组同时测试全部九个版本和内置 12 个边界长度。
- 全 1、可精确表示的正负模式要求零误差；随机数对照 CPU FP64，并使用 README 中说明的容差。
- 覆盖空输入、单元素、非整除长度、grid-stride 多轮累加、第二阶段部分和数量超过线程块大小。

完整日志：correctness.log。

## 一次性能实验

硬件：RTX 3050 Laptop，SM86，16 SM。
N=16777216，block=256；reduce6/7/atomic 默认 grid-cap=256。
10 次预热，3 组各 20 次重复，CUDA Event 取中位数，热缓存；不计分配、H2D/D2H。
0～7 的 full 包含第一阶段和统一第二阶段，atomic full 包含输出清零。

| 版本 | 第一阶段 ms | 完整求和 ms | 完整求和相对 baseline |
|---|---:|---:|---:|
| reduce0_modulo | 1.654272 | 1.544806 | 1.000x |
| reduce1_index | 1.117389 | 1.134438 | 1.362x |
| reduce2_sequential | 1.014426 | 1.096122 | 1.409x |
| reduce3_two_loads | 0.525414 | 0.568525 | 2.717x |
| reduce4_warp_tail | 0.367206 | 0.406067 | 3.804x |
| reduce5_unrolled | 0.365363 | 0.405139 | 3.813x |
| reduce6_grid_stride | 0.363715 | 0.366778 | 4.212x |
| reduce7_shuffle | 0.364218 | 0.364954 | 4.233x |
| atomic_shuffle | — | 0.364083 | 4.243x |

原始数据：benchmark_16M.csv；命令输出：benchmark_16M.log。
这只是一次设备上的教学实验，不保证每个优化都加速，也不是与 PyTorch/CUB 的性能对照。0～4 刻意保留循环，5 引入模板特化/展开；教程 baseline 不代表最优实现。input_GB_s 仅统计输入字节，不能当作物理显存带宽。

## racecheck 限制

尝试检查 reduce4 的 shared warp 尾部，工具退出码为 1，报告 Failed to initialize WDDM debugger interface / Device not supported。
因此 racecheck 未完成，不能声称已排除所有共享内存竞争，也不能把此工具初始化失败当作已确认的 kernel 数据竞争。
日志：racecheck.log。Windows 开关步骤：[WINDOWS_TOOLS.md](../../../WINDOWS_TOOLS.md)。

实现使用 __syncwarp() 而非 volatile/隐式 warp 锁步同步。启用 Windows 调试接口后可按 README 的 memcheck/racecheck 命令补测。

## 可复现入口

在 day05 下执行 ./build.sh、./test.sh、../../run ./reduce_bench --mode both。
README.md 提供单版本、数据规模、block/grid 扫描和 CSV 命令。

源码 SHA256：`4a98d01e2a46715e09e4b551ef46068bceb91023aad9816fde6807634938c3d4`。
