# 第 6 天

阅读：同上，warp shuffle 部分

练习：将 block sum 改为 warp 内 shuffle，再归约各 warp 结果

验收：固定整 warp block，越界 lane 输入 0；解释 mask、warp=32、为何浮点结果可不同

自己的代码保存在这里；阅读入口相对项目根目录。
每日约 60 分钟，见 [完整计划](../../PLAN.md)。


## 检查与修复成果

保留原文件为 `reduce_shfl.cu.original`、`test_reduce_shfl.py.original`。
修复 `constecpr`、`tmplate`、`__forceinlince__` 三处编译错误；改用 64 位元素索引；参数检查移到空输入返回之前；拒绝非正或超过 INT_MAX 的 max_blocks；增加 CUDAGuard 和 kernel 启动错误检查。支持 block=32/64/128/256/512/1024，均为完整 warp。

原始的 warp sum → shared 中转 → warp0 sum → atomicAdd 思路保留。全 32 lane 使用同一 shuffle mask；越界 lane 的寄存器值为 0，不提前退出。只有 lane0 写 warp 部分和，全部线程经过 block 屏障后 warp0 读取。每次输出清零，避免重复调用累加旧值。

```bash
cd /home/zzy/cuda_learn/learning/exercises/day06
../../run python test_reduce_shfl.py
```

输出：`results/test.log`（一键验收时生成）、`results/results.json`、`results/benchmark.csv`、`RESULTS.md`。
测试含六种 block、三种 grid 上限、空/单元素/尾部/随机输入、非默认 stream、重复调用与非法输入拒绝。随机数参考 FP64 累加；全 1 要求零误差。性能对照为公开 API 成本，自写版本含输出分配、清零和原子累加，torch.sum 也分配输出。不要把它当成“只测 kernel 指令”。Atomic 浮点累加顺序可能改变，不能要求随机数据逐位相等。
