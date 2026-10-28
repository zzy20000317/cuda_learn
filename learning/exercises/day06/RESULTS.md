# day06 成果与实测结果

自动测试通过；使用普通用户 zzy。GPU：NVIDIA GeForce RTX 3050 Laptop GPU，CUDA 12.8，PyTorch 2.9.1+cu128。

## 交付与验收

修复编译拼写、64 位索引、参数边界、device guard 和 launch check；保留 shuffle + atomic 主体。原始文件已备份。

```json
{
  "numerical_cases": 360,
  "nondefault_stream": true,
  "invalid_inputs": true,
  "repeated_calls": true,
  "max_abs_error": 0.00025771832224563695
}
```

## 本次性能实验

CUDA events; warm cache; 20 warmup; median of 5 x 50 calls; excludes JIT and input initialization。首次 JIT/输入初始化不计时；这是一次热缓存实验，提交间隙、频率与运行负载可能影响小 kernel。

| N | block | torch ms | 自写 ms | 相对 torch |
|---:|---:|---:|---:|---:|
| 1024 | 128 | 0.073885 | 0.027956 | 2.643x |
| 1024 | 256 | 0.073885 | 0.031457 | 2.349x |
| 1024 | 512 | 0.073885 | 0.032243 | 2.292x |
| 65536 | 128 | 0.017274 | 0.029102 | 0.594x |
| 65536 | 256 | 0.017274 | 0.035410 | 0.488x |
| 65536 | 512 | 0.017274 | 0.029034 | 0.595x |
| 1048576 | 128 | 0.041341 | 0.122648 | 0.337x |
| 1048576 | 256 | 0.041341 | 0.103240 | 0.400x |
| 1048576 | 512 | 0.041341 | 0.035820 | 1.154x |

自写归约包含分配、清零、atomic 累加；torch.sum 同样返回新输出。相同 max_blocks 不等于每个 block 相同工作量。当前简单实现未必比成熟库快，不能只挑最快配置作为普遍收益。随机数使用 FP64 对照与规模相关容差，atomic 累加不保证每次位级相同。

## 复现与学习复盘

运行命令和接口约束见本目录 README.md。原始数据在 results/results.json、results/benchmark.csv，命令日志为 results/test.log。自动检查通过不代表用户已独立掌握；可用 README 的解释题继续口述复盘。
