# CUDA 学习工作区

硬件：WSL2 / Ubuntu 22.04，RTX 3050 Laptop 4GB（SM86）。目标：每天一小时、两周完成基本 CUDA 算子编写与调优入门。

## 开始学习

```bash
cd /home/zzy/cuda_learn/learning
./run python exercises/bench_add.py
# 或进入独立 shell（exit 退出）：
./run
```

`./run` 只为子进程设置环境，不修改 ~/.bashrc、不安装系统驱动、不改变其他 Python 环境。依赖、nvcc、GCC、Python、缓存都放在本目录；共享的是宿主 Windows GPU 驱动和 WSL 系统内核。这是依赖隔离，不是容器级安全隔离。

- [两周计划](PLAN.md)：每天的阅读路径、练习与验收。
- [面试速查](INTERVIEW.md)：复盘问题与项目讲述。
- `exercises/add.cu`：有边界和对齐处理的 FP32 标量/float4 加法参考。
- `exercises/bench_add.py`：正确性检查与线程块/向量化实验。
- `exercises/project_smoke.py`：编译原项目 elementwise.cu 并验证 FP32/FP16。
- `exercises/probe.cu`：最小纯 CUDA 程序。
- `exercises/day01` 至 `day14`：每日自己的代码。
- `notes/`：每日笔记；`results/`：基准与验证报告；`logs/`：安装日志。

## 编辑器

在 VS Code 的 WSL 窗口中打开本 learning 目录，可使用 `.vscode/tasks.json` 的验证和基准任务。运行 CUDA/Python 命令统一通过 `./run`，确保 nvcc 和缓存设置生效。

## 工具与验证

```bash
./verify.sh                       # 核心验收：GPU、nvcc、原项目扩展、学习版、依赖一致性
./run nvcc --version
./run python -m pip list
./run compute-sanitizer --tool memcheck --error-exitcode 1 ./build/probe
./run ncu --set basic -o results/probe ./build/probe
./run cuda-gdb ./build/probe
```

实际检测到的 Windows 权限限制及启用步骤见 [WINDOWS_TOOLS.md](WINDOWS_TOOLS.md)。CUDA-GDB 的官方二进制补充在 tools/cuda-gdb，已校验 SHA256。

Nsight Compute 的计数器采集依赖 Windows 驱动权限和 WSL 支持；若提示 ERR_NVGPUCTRPERM，记录限制，继续使用 CUDA Event 计时。调试自己的程序时另建 `-g -G` 调试版本；性能测量使用 `-O3 -lineinfo`，不能用 `-G`。

原项目测试依赖 FlashInfer 的计时工具；先使用本目录的小规模验证，原项目大 shape benchmark 按显存容量逐步运行。`sm120`、TMA 等高级代码不在本学习环境验收范围。

## 中断后继续

新克隆的项目若安装中断，可重新运行 `bash setup.sh`。本机还保留 `tools/finish-install.sh` 供已有分段缓存的恢复使用，该本机工具目录不提交到 Git。安装日志在 `logs/finish-install.log`；`logs/install-exit-code.txt` 是最近一次已退出任务的退出码（运行期间可能仍是旧值）。

## 安装与复现

环境定义在 `environment.yml`；安装成功后 `conda-explicit.txt`、`requirements-lock.txt` 保存实际版本。CUDA Toolkit 12.8 与 PyTorch cu128 对齐。首次扩展编译需要等待，MAX_JOBS 默认 2 以控制内存；内存紧张用 `MAX_JOBS=1 ./run ...`。

```bash
# 在当前已配置的 learning 目录恢复 Python 包：
./run python -m pip install -r requirements-lock.txt --extra-index-url https://download.pytorch.org/whl/cu128
# 重新建环境时（先自行备份已有环境）：
tools/bin/micromamba --no-rc create -r "$PWD/.mamba" -p "$PWD/.env" -f conda-explicit.txt
```

不要随意移动 `.env`，环境内部可能含绝对路径；迁移时复制学习源码、按锁文件重建。退出环境使用 `exit`。未来卸载时备份 exercises/notes/results，然后删除 learning 内的 `.env`、`.mamba`、`.cache` 和 tools 即可释放依赖空间。

## 官方依据

- [NVIDIA WSL 指南](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)：使用 Windows 驱动，WSL 内不安装 Linux 显示驱动。
- [CUDA 12.8 Conda 安装指南](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-installation-guide-linux/)：Toolkit 可安装到独立 Conda 环境。
- [PyTorch 历史版本](https://pytorch.org/get-started/previous-versions/)：2.9.1 cu128 官方安装组合。

环境是否验收通过以 `results/STATUS.md` 和 `logs/verify.log` 为准；不能仅凭目录存在判断安装完成。

## 本机目录改名

项目已改名为 `/home/zzy/cuda_learn`。本机保留旧目录名的兼容软链接，供已安装环境内部的旧绝对路径使用；该链接不属于仓库。新克隆项目应按 setup.sh 重建环境。
