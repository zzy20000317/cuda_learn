# day09 成果与实测结果

自动测试通过；使用普通用户 zzy。GPU：NVIDIA GeForce RTX 3050 Laptop GPU，CUDA 12.8，PyTorch 2.9.1+cu128。

## 交付与验收

对照只改变 shared 行跨度的 32×32 / 32×33 tile。交付逐 lane bank CSV 和 SVG 图。正确性精确对照通过。

```json
{
  "exact_cases": 50,
  "nondefault_stream": true,
  "invalid_inputs": true,
  "empty_dimensions": true,
  "offset_view": true
}
```

## 本次性能实验

CUDA events; warm cache; 20 warmup; median of 5 x 50 calls; excludes JIT and input initialization。首次 JIT/输入初始化不计时；这是一次热缓存实验，提交间隙、频率与运行负载可能影响小 kernel。

| shape | 版本 | torch ms | 自写 ms | 相对 torch |
|---|---|---:|---:|---:|
| 32×32 | shared_32x32 | 0.017154 | 0.010317 | 1.663x |
| 32×32 | padded_32x33 | 0.017154 | 0.026658 | 0.643x |
| 256×512 | shared_32x32 | 0.023775 | 0.016767 | 1.418x |
| 256×512 | padded_32x33 | 0.023775 | 0.015320 | 1.552x |
| 257×513 | shared_32x32 | 0.029000 | 0.015073 | 1.924x |
| 257×513 | padded_32x33 | 0.029000 | 0.014111 | 2.055x |
| 1024×1024 | shared_32x32 | 0.077640 | 0.052838 | 1.469x |
| 1024×1024 | padded_32x33 | 0.077640 | 0.049295 | 1.575x |
| 2048×2048 | shared_32x32 | 0.337754 | 0.199803 | 1.690x |
| 2048×2048 | padded_32x33 | 0.337754 | 0.187144 | 1.805x |

PyTorch 对照为将 a.t() 拷贝进预分配连续输出；a.t() 本身只产生视图。有效带宽包含缓存影响，不等于实测物理显存带宽。

### padding 对照

| shape | 无 padding ms | padding ms | 相对无 padding |
|---|---:|---:|---:|
| 32×32 | 0.010317 | 0.026658 | 0.387x |
| 256×512 | 0.016767 | 0.015320 | 1.094x |
| 257×513 | 0.015073 | 0.014111 | 1.068x |
| 1024×1024 | 0.052838 | 0.049295 | 1.072x |
| 2048×2048 | 0.199803 | 0.187144 | 1.068x |

[bank 映射图](results/bank_mapping.svg) 和 [bank 映射 CSV](results/bank_mapping.csv) 展示理论地址映射。float 元素的 bank=(row*stride+col)%32；列读取时 stride32 对应同 bank 不同字，stride33 分散到全部 32 bank。shared 内存增加 128 字节。

Nsight Compute 返回 ERR_NVGPUCTRPERM：Windows 未授权 GPU 性能计数器。本次未采集硬件冲突计数；性能时间和理论映射不能代替硬件计数证据。步骤见 [Windows 工具说明](../../WINDOWS_TOOLS.md)。

## 复现与学习复盘

运行命令和接口约束见本目录 README.md。原始数据在 results/results.json、results/benchmark.csv，命令日志为 results/test.log。自动检查通过不代表用户已独立掌握；可用 README 的解释题继续口述复盘。
