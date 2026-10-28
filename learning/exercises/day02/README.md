# 第 2 天

阅读：`kernels/elementwise/elementwise.cu` 的标量版

练习：不看答案写 FP32 add，处理 N=0/1/257/1025；与 torch 对比

验收：正确性通过；解释为何需要 `i<N`、为何 CUDA 调用异步

自己的代码保存在这里；阅读入口相对项目根目录。
每日约 60 分钟，见 [完整计划](../../PLAN.md)。
