# cuda_learn

这是我的个人 CUDA 编程学习项目，用来记录从基础 kernel 编写到简单算子调优的练习、笔记、正确性测试和实测结果。目标是建立能够解释、编写和验证基础 CUDA 算子的能力，为相关技术面试积累实际案例。

目前已完成前 **9 天**的代码练习与阶段验收；第 10～14 天仍在计划中。部分代码与复盘材料由 AI 辅助完成，学习时会结合参考实现分析、修改并运行测试。代码通过测试不等于已经独立掌握，限时编程和口述仍需持续练习。

## 已完成的学习内容

| 天数 | 主题 | 主要成果 |
|---|---|---|
| [Day 01](learning/exercises/day01/) | GPU 属性、线程索引、kernel launch | 填充数组与最小 CUDA 程序 |
| [Day 02](learning/exercises/day02/) | FP32 elementwise add | 索引、边界判断与基本验证 |
| [Day 03](learning/exercises/day03/) | CUDA Event 计时 | 预热、重复测量与线程块实验 |
| [Day 04](learning/exercises/day04/) | 向量化与尾部处理 | float4、地址对齐及非整除长度 |
| [Day 05](learning/exercises/day05/) | Reduce 优化路线 | reduce0～7、两阶段求和与 atomic 对照 |
| [Day 06](learning/exercises/day06/) | Warp shuffle 归约 | 多 warp 合并、参数检查、stream 与误差测试 |
| [Day 07](learning/exercises/day07/) | 首周复盘与 bug 修复 | Add 复习、归约漏算复现与修复、复盘笔记 |
| [Day 08](learning/exercises/day08/) | 矩阵转置 | Naive 读/写合并、shared tile、边界处理 |
| [Day 09](learning/exercises/day09/) | Shared memory bank conflict | 32×32/32×33 tile 对比、bank 映射图与实测记录 |

完整安排见 [两周学习计划](learning/PLAN.md)，面试复盘见 [INTERVIEW.md](learning/INTERVIEW.md)。

## 项目结构与来源

```text
learning/               我的学习工作区
  exercises/day01..09/  已完成的逐日练习、测试与报告
  exercises/day10..14/  后续学习计划与占位目录
  results/              基础环境和性能实验记录
  run                   隔离环境运行入口
  setup.sh              环境安装与重建
kernels/                原 Vitamin-CUDA 算子参考实现
kernels_dsl/            原项目 DSL 参考实现
samples/                原项目硬件与性能学习示例
posts/                  原项目文章材料
infer/                  原项目推理相关示例
README.upstream.md      原项目 README，保留来源与参考说明
```

本项目以 [**Vitamin-CUDA**](https://github.com/WingEdge777/vitamin-cuda) 的源码作为学习参考，保留其原始代码、[README](README.upstream.md) 和 [MIT License](LICENSE)（原版权署名：wingedge）。`kernels/` 等参考目录中的实现不属于我个人独立完成的成果；个人练习、测试、实验记录主要集中在 `learning/`。笔记中的外部博客及 AI 总结是学习资料，具体来源见相应文件。

## 开发环境

- WSL2 / Ubuntu 22.04
- NVIDIA RTX 3050 Laptop，4GB 显存，Compute Capability 8.6
- CUDA Toolkit 12.8、GCC 13、Python 3.11
- PyTorch 2.9.1 cu128、CMake、Ninja

Python、CUDA Toolkit 和编译依赖安装在 `learning/` 内，通过 `learning/run` 为子进程设置环境。仓库保留环境定义和版本锁文件，不提交环境目录、依赖下载缓存、编译产物或凭据。

### 已配置的本机环境

```bash
cd /home/zzy/cuda_learn/learning
./run nvcc --version
./run python exercises/day06/test_reduce_shfl.py
```

也可以用 `./run` 进入学习 shell，`exit` 退出。普通终端的 Conda `(base)` 环境中直接执行 `nvcc`，可能找不到项目内的编译器。

### 在新机器重建环境

需要先具备可用的 NVIDIA GPU 驱动与 CUDA 运行支持，然后在克隆的项目目录运行：

```bash
cd cuda_learn/learning
bash setup.sh
```

安装包含数 GB 依赖，首次编译也需要时间。版本和安装说明见 [学习环境 README](learning/README.md)、[environment.yml](learning/environment.yml) 与 [requirements-lock.txt](learning/requirements-lock.txt)。其他 GPU 需要相应修改 `learning/run` 中的 `TORCH_CUDA_ARCH_LIST` 和练习中的 `-arch` 参数；本仓库的验收主要针对 SM86。

## 运行练习

```bash
# Day 05：baseline 到 shuffle 的逐版对比
cd /home/zzy/cuda_learn/learning/exercises/day05
./build.sh
./test.sh
../../run ./reduce_bench --n 16777216 --mode both

# Day 06～09：依次检查并更新成果报告
cd /home/zzy/cuda_learn/learning/exercises
./run_days06_09.sh
```

每天目录中的 README 给出单独运行、不同 shape、线程块和 grid 的测试方法；`RESULTS.md` 记录对应成果。小规模 CSV/JSON 和 bank 映射图保留在仓库内，便于查看历史实验。重新运行会更新这些记录。

## 如何看待性能结果

先对照 CPU/PyTorch 验证正确性，再进行 CUDA Event 计时。各实验明确是否包含输出分配、清零、第二阶段求和；编译和输入初始化不计入测量。记录基于本机 GPU 的热缓存实验，不能直接与其他硬件或其他计时方式比较。

例如，Day 05 的一次 16M 元素实验中，完整求和的 baseline 约 1.545 ms，shuffle 版约 0.365 ms；Day 09 的一次 2048×2048 转置实验中，padding 约快 1.07 倍。完整参数和原始记录以各日目录内的报告、CSV 为准，这些数字不构成普遍的性能保证，也不表示超过成熟算子库。

Nsight Compute 当前受 Windows 性能计数器权限限制，未采集到硬件 bank-conflict 计数；bank 图展示理论地址映射。处理步骤见 [Windows 工具权限说明](learning/WINDOWS_TOOLS.md)。

## 后续计划

继续完成稳定 Softmax、基础 GEMM、shared tiled GEMM、单算子调优报告与模拟面试；优先做到能解释和复现，不追求覆盖所有高级优化技术。
