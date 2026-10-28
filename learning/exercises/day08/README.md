# 第 8 天

阅读：`kernels/mat_transpose/transpose.cu`

练习：写 naive transpose 和 32×32 shared tile；可用 32×8 线程搬多行

验收：32×32、256×512、257×513 正确；解释读写哪一端不合并

自己的代码保存在这里；阅读入口相对项目根目录。
每日约 60 分钟，见 [完整计划](../../PLAN.md)。


## 当天成果与产出

`transpose.cu` 实现三个 FP32 转置版本：0=naive_read（读合并、写跨行）；1=naive_write（写合并、读跨行）；2=32×32 shared tile、32×8 线程，每线程搬 4 行。附 flat copy 作为访存参考，它不是转置。

输入 row-major [M,N]，输出独立连续 [N,M]。共享 tile 完整加载/补零后全部线程进入屏障，再交换 block 坐标并读取 tile 的列。不能让越界线程在 __syncthreads 之前提前返回。边界条件分别针对输入和交换后的输出坐标。

```bash
cd /home/zzy/cuda_learn/learning/exercises/day08
../../run python test_transpose.py
```

`test_transpose.py` 覆盖计划的 32×32、256×512、257×513，还包括空维度、单行/单列、反向长方形、跨 tile 尾部、offset view、非默认 stream。索引模式 arange 与随机数据均与 torch 转置精确相等；拒绝 CPU、FP16、非连续输入、错误输出 shape 与重叠输入输出。

性能对照采用 `b.copy_(a.t())`，torch 和自写版本都使用提前分配的连续输出，不把零拷贝的 a.t() 视为物理转置性能。copy 仅作为参考；有效带宽按输入+输出 8*M*N 字节计算，缓存可能影响结果。

产出：`results/results.json`、`results/benchmark.csv`、`RESULTS.md`。`transpose_contract.cuh` 是 day08/day09 共用的输入检查；计时/导出工具复用 day07。

### 坐标例子

```text
输入 2×3                  输出 3×2
0 1 2                     0 3
3 4 5                     1 4
                          2 5
b[col*M + row] = a[row*N + col]
```

naive_read 的一个 warp 固定 row、连续 col：输入连续，输出相隔 M 个 float。
shared tiled 的加载访问输入行，写回时访问输出行；tile 内部的列访问在 day09 继续分析 bank conflict。
