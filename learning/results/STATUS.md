# 环境验收状态

**核心学习开发环境：通过。** 最终验收使用普通用户 `zzy`，退出码 0。

## 已验证

| 项目 | 结果 |
|---|---|
| WSL2 / Ubuntu 22.04 / RTX 3050 Laptop | SM86，16 SM，4095 MiB 显存 |
| 独立 Python / 编译器 | Python 3.11、GCC 13.4、nvcc 12.8.93 |
| PyTorch | 2.9.1+cu128，GPU 张量计算通过 |
| 纯 CUDA / CMake+Ninja | 编译并执行 probe 通过 |
| 原项目 elementwise.cu | JIT 编译、FP32/FP16 对照通过 |
| FlashInfer 0.6.2 | 原项目计时工具导入及实际运行通过 |
| 学习版 add | 空输入、尾部、未对齐输入、非默认 stream、128/256/512 线程通过 |
| 性能实验 | 4 个元素规模 × 3 个线程块 × 标量/float4，共 24 组记录 |
| Python 依赖 | pip check 通过；requirements-lock.txt 保存最终版本 |
| 普通用户权限 | learning 全目录归 zzy，核心验收无需 sudo |
| CUDA-GDB | 12.8 可执行程序启动通过；设备调试受下述 Windows 开关限制 |

完整日志：[verify.log](../logs/verify.log)。入口：[README](../README.md)。两周计划：[PLAN](../PLAN.md)。

## 尚需 Windows 侧启用的工具权限

- Compute Sanitizer：当前 WDDM 调试接口未启用，**内存检查未通过**，详见 logs/sanitizer.log。
- Nsight Compute：当前返回 ERR_NVGPUCTRPERM，**硬件性能计数器尚不可采集**，详见 logs/ncu.log。
- CUDA-GDB 的设备调试同样需要 Windows 调试接口；目前仅确认调试器可启动。

操作步骤：[WINDOWS_TOOLS.md](../WINDOWS_TOOLS.md)。这两项不阻止 CUDA 编写、编译、PyTorch 对照及 Event 计时。未修改 Windows 驱动设置或注册表。

## 隔离范围

Python、CUDA Toolkit、GCC、依赖与运行时构建缓存位于 learning 内；通过 ./run 为子进程设置环境，exit 即退出。不修改 shell 启动文件或系统 CUDA，Windows GPU 驱动为共享依赖。高级 SM120/TMA/FlashAttention 实现不属于此学习环境的运行验收范围。

## 之前的中断

前次安装进程退出时未留下明确错误原因；续跑发现合并 wheel 文件被截断。所有原始分段完整，已重建并校验官方 SHA256。已提供 tools/finish-install.sh 可续跑脚本，但关闭执行会话仍可能终止任务，因此不能把启动后台任务视为安装完成。本报告是在最终实际验收通过后生成。

报告生成：2026-09-15T19:13:59.572047+08:00
