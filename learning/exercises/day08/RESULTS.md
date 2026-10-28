# day08 成果与实测结果

自动测试通过；使用普通用户 zzy。GPU：NVIDIA GeForce RTX 3050 Laptop GPU，CUDA 12.8，PyTorch 2.9.1+cu128。

## 交付与验收

交付读合并/写合并 naive 转置、32×32 shared tile、flat copy 参考。计划的三个 shape 及更多边界输入精确对照通过。

```json
{
  "exact_cases": 75,
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
| 32×32 | naive_read | 0.025779 | 0.011551 | 2.232x |
| 32×32 | naive_write | 0.025779 | 0.012223 | 2.109x |
| 32×32 | shared_32x32 | 0.025779 | 0.010076 | 2.558x |
| 256×512 | naive_read | 0.014602 | 0.023839 | 0.613x |
| 256×512 | naive_write | 0.014602 | 0.013804 | 1.058x |
| 256×512 | shared_32x32 | 0.014602 | 0.014950 | 0.977x |
| 257×513 | naive_read | 0.021052 | 0.023060 | 0.913x |
| 257×513 | naive_write | 0.021052 | 0.012223 | 1.722x |
| 257×513 | shared_32x32 | 0.021052 | 0.013332 | 1.579x |
| 1024×1024 | naive_read | 0.084296 | 0.169206 | 0.498x |
| 1024×1024 | naive_write | 0.084296 | 0.071598 | 1.177x |
| 1024×1024 | shared_32x32 | 0.084296 | 0.052779 | 1.597x |
| 2048×2048 | naive_read | 0.333476 | 0.577270 | 0.578x |
| 2048×2048 | naive_write | 0.333476 | 0.340378 | 0.980x |
| 2048×2048 | shared_32x32 | 0.333476 | 0.191140 | 1.745x |

PyTorch 对照为将 a.t() 拷贝进预分配连续输出；a.t() 本身只产生视图。有效带宽包含缓存影响，不等于实测物理显存带宽。

### copy 参考

| shape | flat copy ms |
|---|---:|
| 32×32 | 0.012452 |
| 256×512 | 0.010752 |
| 257×513 | 0.012247 |
| 1024×1024 | 0.051180 |
| 2048×2048 | 0.187453 |

naive_read 输入连续但输出跨行；naive_write 则相反。shared tile 让 global 两侧都访问行，但 tile 列读取仍有 bank conflict，下一日继续优化。copy 不是转置，仅作数据搬运参考。

## 复现与学习复盘

运行命令和接口约束见本目录 README.md。原始数据在 results/results.json、results/benchmark.csv，命令日志为 results/test.log。自动检查通过不代表用户已独立掌握；可用 README 的解释题继续口述复盘。
