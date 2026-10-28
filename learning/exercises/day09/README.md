# 第 9 天

阅读：同上 `transpose_smem_bcf`；`samples/bank_conflict_ncu/`

练习：比较 shared tile `[32][32]` 与 `[32][33]`；可运行 ncu

验收：记录时间；画出列访问 bank 映射；性能计数器不可用时用代码解释+Event 数据

自己的代码保存在这里；阅读入口相对项目根目录。
每日约 60 分钟，见 [完整计划](../../PLAN.md)。


## 当天成果与产出

`transpose_padding.cu` 用同一个模板对比 `tile[32][32]`（version=0）与 `tile[32][33]`（version=1）。线程布局、global 访存、循环与同步相同，只改变 shared 行跨度。共享内存从 4096 增至 4224 字节，多 128 字节。

```bash
cd /home/zzy/cuda_learn/learning/exercises/day09
../../run python test_padding.py
```

产出：`results/results.json`、`results/benchmark.csv`、`results/bank_mapping.csv`、`results/bank_mapping.svg`、`RESULTS.md`。正确性复用 day08 的全部边界/布局/stream 检查；通用接口检查复用 day08/transpose_contract.cuh；计时复用 day07 工具。

### 为什么加一列

对于 float32 元素索引 `tile[row][col]`，bank=`(row*stride+col)%32`。转置写回前的 shared load `tile[tx][ty+j]` 中，一个 warp 的 tx=lane，列固定。

```text
lane                0   1   2   3 ... 31
stride32，col=0     0   0   0   0 ...  0    bank编号
stride33，col=0     0   1   2   3 ... 31    bank编号
```

跨度 32 时，同一个 load 的 32 lane 读取同一 bank 的不同字，形成 32 路冲突，不是同地址广播。跨度 33 后均匀落在 32 个 bank。SVG 图对应此列加载的理论 bank 映射；不是硬件计数器测量。CSV 还给出 col=1/7/31 的映射。

性能报告给出 `speedup_vs_unpadded`，小矩阵和不规则矩阵不保证 padding 更快。有效带宽不是物理显存带宽。Event 时间支持“是否变快”的观察，不能单凭时间证明硬件冲突计数下降。

### 可选硬件计数器采集

```bash
../../run ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum \
  --kernel-name 'regex:transpose_padding.*' --launch-count 2 python profile_padding.py
```

`profile_padding.py` 只运行两种 1024×1024 kernel，避免采集到边界测试的小矩阵。实际采集状态见 RESULTS.md。若 ERR_NVGPUCTRPERM，需要 [Windows 性能计数器权限](../../WINDOWS_TOOLS.md)。ncu 下的时间不作为本实验正常基准。
