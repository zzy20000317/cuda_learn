# 第 7 天

阅读：复习 2–6 天

练习：从空文件重写 add；修复一个归约 bug；整理首周笔记

验收：20 分钟写 add；口述“正确性→瓶颈猜测→修改→测量”；来不及则归约停在 shared 版

自己的代码保存在这里；阅读入口相对项目根目录。
每日约 60 分钟，见 [完整计划](../../PLAN.md)。


## 当天成果与产出

- `review.cu`：独立标量 grid-stride add；可控的 row reduce 逻辑 bug；修复后的两级 shuffle 行归约。
- `test_review.py`：add 与 torch 的精确对照、尾部/偏移切片、非默认 stream、非法输入检查；归约 bug 的确定性复现与修复验收。
- `WEEK1_NOTES.md`：首周概念复盘、调试步骤和口述题。
- `results/results.json`、`results/benchmark.csv`、`RESULTS.md`：实测检查与性能记录。
- `cuda_test_utils.py`：本组练习共用的 Event 计时与结果导出工具；day06/08/09 也使用它。

```bash
cd /home/zzy/cuda_learn/learning/exercises/day07
../../run python test_review.py
```

归约 bug：2×256 全 1 输入，每行正确值为 256，buggy 版本只合并 warp0 的 32 个输入而返回 32。修复版先存 8 个 warp 的部分和，经 __syncthreads 后交给 warp0 归约，返回 256。这是安全的逻辑 bug 演示，不故意触发越界或数据竞争。

add 性能实验输出提前分配；torch.add(out=...) 与自写 add 使用相同策略。默认 20 次预热、5 组各 50 次调用中位数，不计 JIT/随机输入初始化。参考实现与自动测试已交付；“20 分钟从空文件默写”和口述能力需由学习者另行自测，不能从脚本通过推断。
