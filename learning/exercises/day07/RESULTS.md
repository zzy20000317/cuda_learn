# day07 成果与实测结果

自动测试通过；使用普通用户 zzy。GPU：NVIDIA GeForce RTX 3050 Laptop GPU，CUDA 12.8，PyTorch 2.9.1+cu128。

## 交付与验收

交付独立 add 参考实现、可控归约 bug 与修复、首周复盘笔记。256 个全 1 的行归约：bug 版本得到 32，修复版得到 256。个人限时默写和口述仍需学习者自测。

```json
{
  "add_cases": 54,
  "row_reduce_shapes": 6,
  "nondefault_stream": true,
  "invalid_inputs": true,
  "bug_expected": 256,
  "bug_observed": 32,
  "fixed_observed": 256
}
```

## 本次性能实验

CUDA events; warm cache; 20 warmup; median of 5 x 50 calls; excludes JIT and input initialization。首次 JIT/输入初始化不计时；这是一次热缓存实验，提交间隙、频率与运行负载可能影响小 kernel。

| N | block | torch ms | 自写 ms | 相对 torch |
|---:|---:|---:|---:|---:|
| 1024 | 128 | 0.011776 | 0.009057 | 1.300x |
| 1024 | 256 | 0.011776 | 0.011039 | 1.067x |
| 1024 | 512 | 0.011776 | 0.009725 | 1.211x |
| 65536 | 128 | 0.010568 | 0.009567 | 1.105x |
| 65536 | 256 | 0.010568 | 0.012063 | 0.876x |
| 65536 | 512 | 0.010568 | 0.008194 | 1.290x |
| 1048576 | 128 | 0.071819 | 0.071823 | 1.000x |
| 1048576 | 256 | 0.071819 | 0.071516 | 1.004x |
| 1048576 | 512 | 0.071819 | 0.072172 | 0.995x |

add 使用预分配输出，torch 使用 out=。大 N 时接近的时间不证明存在稳定优化收益；此日主要成果是从简单实现出发验证与定位归约 bug。

## 复现与学习复盘

运行命令和接口约束见本目录 README.md。原始数据在 results/results.json、results/benchmark.csv，命令日志为 results/test.log。自动检查通过不代表用户已独立掌握；可用 README 的解释题继续口述复盘。
