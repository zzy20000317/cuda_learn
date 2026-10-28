# 第 5 天

阅读：`kernels/reduce/reduce_sum.cu`；先看归约结构

练习：从每 block 一行开始，写 shared-memory sum，列长先固定 256

验收：与 torch.sum 比较并使用浮点容差；解释 `__syncthreads()`、跨 block 不能这样同步

自己的代码保存在这里；阅读入口相对项目根目录。
每日约 60 分钟，见 [完整计划](../../PLAN.md)。


## 按 reduce_learn.md 实现的实验

`reduce.cu` 包含 reduce0～reduce7，以及补充的 atomic 对比版本。使用纯 CUDA C++，不需要 PyTorch 扩展/JIT；`reduce_learn.md` 保留原样。接口语义为 **一维连续 FP32 数组的全局求和**，不是按行求和。

| 版本 | 输入分配 | block 内方法 | 对照时关注 |
|---|---|---|---|
| 0 baseline | 每线程 1 个元素 | 上升 stride，`tid % (2*stride)` | 活跃 lane 分散、循环与同步开销 |
| 1 index | 每线程 1 个元素 | `index=2*stride*tid` | 前几轮活跃 warp 更完整，但共享访存有 bank conflict |
| 2 sequential | 每线程 1 个元素 | 下降 stride，`tid<stride` | 同一 shared load 指令的活跃线程访问连续地址 |
| 3 two_loads | 每线程 2 个元素 | 先在寄存器相加，再用版本 2 的树 | block 数减半，部分和数量减少 |
| 4 warp_tail | 每线程 2 个元素 | 剩余 64 个值交给 warp 0；shared + `__syncwarp` | 末尾由 block 屏障改为 warp 屏障 |
| 5 unrolled | 每线程 2 个元素 | 模板特化、展开 shared 树 | 编译期消除循环控制；现代编译器下收益未必明显 |
| 6 grid_stride | 每线程循环处理多对元素 | 寄存器累加 + 版本 5 的树 | 限制 grid、调节每线程工作量 |
| 7 shuffle | 与版本 6 相同 | warp 内 shuffle；每 warp 仅存 1 个 partial | 减少 shared memory 通信与 block 屏障 |
| atomic（补充） | 与版本 7 相同 | shuffle 后 `atomicAdd` 到标量 | 对比两阶段与原子合并；计时包含清零 |

为便于逐段阅读，0～4 各自保留完整 kernel，并用 `#pragma unroll 1` 保留归约循环以展示版本 5 的展开变化；这是教学对照，不是声称 baseline 已经过最优编译调优。5～7 只抽出展开的树与 warp 求和小函数。所有 block 支持 128/256/512/1024，尾部越界补零，空数组结果为 0。默认 reduce6/7/atomic 的 grid 上限为 `SM_count*16`，不是声称该值最优。`--grid` 可修改上限；实际 block 数还受输入长度限制。0～5 的 grid 按输入长度确定，忽略这个上限。

### 编译和第一次运行

从普通终端执行以下命令；`../../run` 自动使用之前配置的隔离环境，无需手动激活 Conda 或系统安装 nvcc。

```bash
cd /home/zzy/cuda_learn/learning/exercises/day05
./build.sh

# 全部版本；同时显示第一阶段和完整求和；打印 CSV 风格对比表
../../run ./reduce_bench --n 16777216 --block 256 --mode both

# 完整正确性检查：四种 block、三种 grid 上限、边界与随机输入
./test.sh
```

### 分别测试版本

```bash
# 单独看 baseline
../../run ./reduce_bench --version 0 --n 16777216 --mode both
# 单独看 shuffle
../../run ./reduce_bench --version 7 --n 16777216 --mode both
# 单独看 atomic（只有 full 模式的数据）
../../run ./reduce_bench --version atomic --n 16777216 --mode full
# 只做正确性检查，不计时；同时验证内置边界样例和指定 N
../../run ./reduce_bench --check-only --n 1048579 --block 512 --grid 17
```

选单个版本时，也会计时 reduce0 以生成加速比。`--mode stage1 --version atomic` 没有可比较的第一阶段，程序会明确拒绝该组合。

### 扫描数据规模和线程块

```bash
mkdir -p results
for n in 1024 65536 1048576 16777216 33554432; do
  ../../run ./reduce_bench --n "$n" --block 256 --csv "results/n${n}.csv"
done
for block in 128 256 512 1024; do
  ../../run ./reduce_bench --n 16777216 --block "$block" --csv "results/block${block}.csv"
done
```

### 单独研究 grid 上限（版本 6/7）

```bash
for version in 6 7; do
  for grid in 16 32 64 128 256 512 1024; do
    ../../run ./reduce_bench --version "$version" --grid "$grid" --n 16777216 \
      --mode full --csv "results/v${version}_grid${grid}.csv"
  done
done
```

此示例的 16 对应 RTX 3050 的 SM 数；程序也会打印实际设备 SM 数。先执行 `mkdir -p results`。若只想快速试运行，可加 `--warmup 3 --repeats 10 --trials 3`；正式比较建议默认 10 次预热、5 组各 50 次启动。`./reduce_bench --help` 查看所有选项。

### 怎样读结果

- `stage1`：仅部分和 kernel；0～7 分别写出多个 block 的部分和，**还不是最终标量**。博客主要优化的就是这一阶段。
- `full`：0～7 的第一阶段 + 统一的 `finish_sum`，得到最终标量。第二阶段使用一个 256 线程 block 循环处理所有部分和，避免“部分和超过线程数就漏算”。同一 stream 中按 launch 顺序建立阶段依赖。
- atomic 的 `full`：每次 `cudaMemsetAsync` 清零 + atomic kernel；没有独立部分和阶段。清零不可遗漏，也不应从完整算子的计时里隐藏。
- `ms`：CUDA Event 测量；每组时间除以重复数，多个组取中位数。编译、分配、CPU/GPU 拷贝和正确性检查不计入。采用热缓存重复执行，小 N 的结果可能受到提交间隙影响。
- `speedup_vs_reduce0`：同 N、block、计时模式下 `reduce0时间/本版本时间`；大于 1 表示这次测量更快。不要只看一次微小差异。
- `input_GB_s`：`N*4/耗时`，仅统计输入字节，不包含 partial 读写、输出清零等，也可能受缓存影响；它不是实测物理显存带宽。
- `abs_error/tolerance`：指定 N 的完整求和与 CPU FP64 累加的误差/阈值。即使该行测 stage1，也展示独立 full 正确性检查结果。

内置测试含 N=0/1、warp/block 边界附近、65539 个元素；输入覆盖全 1、可精确表示的正负模式、固定种子随机数，并额外验证 `--n` 指定规模。全 1 和可精确表示的正负模式要求零误差；随机输入的容差为 `1e-4 + 5e-5*max(abs(FP64参考值), sqrt(N))`，用于这些入门测试的近零和及不同累加顺序；不是任意数值分布的严格误差保证。实现仅处理有限 FP32 输入，暂不测试 NaN/Inf 或严重病态的大动态范围。

### 笔记中需要修正的细节

1. **现代 GPU 的 warp 内共享内存通信仍需要正确同步。** `volatile` 不能替代同步。本例版本 4/5/6 每一步采用 `__syncwarp()`；不能把博客的无同步尾部原样放到 SM86 上。版本 7 使用带 mask 的 shuffle，所有 32 lane 参与，缺少有效数据的 lane 贡献 0。
2. 对 `float sdata[i]`，bank 编号是 **`i % 32`**；若用字节地址才是 `(addr/4)%32`。AI 总结把版本 1 的 `2*stride*tid` 地址分析套到了 baseline，且混用了元素索引与字节地址。
3. 分支发散不等于“时间必然翻倍”；此处没有非空 else 分支，编译器还可能使用谓词指令。版本 1 改善活跃线程布局，仍需实测。
4. 版本 3 减少 block 和 partial 数，但 0～7 **没有原子竞争**；只有补充的 atomic 版本存在同地址原子累加。
5. `__syncthreads()` 的范围是一个 block；普通 kernel 常用另一次 launch 合并 partial。CUDA 也有 cooperative groups 等跨 block 协作机制，不能说“硬件禁止一切跨 block 通信”或“atomic 是唯一方式”。它们超出本练习范围。
6. shuffle 的延迟不是固定“1 个周期”，优化版本也不保证逐个变快。两阶段在固定配置下采用固定累加结构；改 block/grid/版本仍会改变浮点累加顺序，atomic 顺序还可能随运行变化。

### 可选的工具检查

```bash
../../run compute-sanitizer --tool memcheck --error-exitcode 1 \
  ./reduce_bench --check-only --n 65539
../../run compute-sanitizer --tool racecheck --error-exitcode 1 \
  ./reduce_bench --version 4 --check-only --n 65539
../../run ncu --set basic --kernel-name 'regex:reduce[0-7].*' \
  ./reduce_bench --n 16777216 --repeats 1 --trials 1 --warmup 0 --mode stage1
```

测试记录见 [results/TEST_REPORT.md](results/TEST_REPORT.md)。这些工具可能需要之前记录的 [Windows 调试接口/性能计数器权限](../../WINDOWS_TOOLS.md)。不要将 ncu/sanitizer 下的时间当成正常性能基准。普通运行通过不能证明 racecheck 已通过。
