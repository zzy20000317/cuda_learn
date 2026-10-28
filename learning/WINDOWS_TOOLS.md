# Windows 侧调试/性能分析权限

当前基础 CUDA 编译、运行不依赖以下开关。实际检测到的工具限制：

## Nsight Compute：ERR_NVGPUCTRPERM

在 **Windows NVIDIA 控制面板**中启用开发者设置（菜单 Desktop/桌面 → Enable Developer Settings/启用开发者设置），进入 Developer/开发者 → Manage GPU Performance Counters/管理 GPU 性能计数器，允许 GPU 性能计数器访问。不同驱动界面措辞可能不同。

回到 WSL 复测：

```bash
cd /home/zzy/cuda_learn/learning
./run ncu --set basic -o results/probe-enabled ./build/probe
```

依据：[NVIDIA 性能计数器权限说明](https://developer.nvidia.com/ERR_NVGPUCTRPERM)、[WSL 性能分析支持](https://docs.nvidia.com/nsight-compute/ReleaseNotes/topics/system-requirements.html)。

## Compute Sanitizer：Failed to initialize WDDM debugger interface

在 **Windows 管理员 PowerShell** 中执行以下命令启用 NVIDIA 调试接口（这是 Windows 全局驱动设置，不属于 learning 目录的依赖隔离）：

```powershell
reg.exe add "HKLM\SOFTWARE\NVIDIA Corporation\GPUDebugger" /v EnableInterface /t REG_DWORD /d 1 /f
```

如设置后仍无效，先保存 WSL 工作，再从 Windows 重启 WSL；必要时重启 Windows。不要在当前有安装/编译任务时执行 `wsl --shutdown`。

```bash
./run compute-sanitizer --tool memcheck --error-exitcode 1 ./build/probe
./run nvcc -g -G -arch=sm_86 exercises/probe.cu -o build/probe-debug
./run cuda-gdb ./build/probe-debug
```

依据：[CUDA-GDB 12.8 官方 WSL 要求](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-gdb/index.html)。当前任务仅提供命令，未修改 Windows 注册表。当前 sanitizer 错误不能算作“内存检查已通过”。

## 学习安排

这些权限未启用时，仍可以完成两周计划中的编写、PyTorch 对照、CUDA Event 计时。第 9 天使用代码分析和时间比较记录 bank conflict 实验；不能声称已采集硬件计数器。启用权限后补跑工具验证。
