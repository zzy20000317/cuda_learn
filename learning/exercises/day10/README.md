# 第 10 天

阅读：`kernels/softmax/softmax.cu`，只理解基础稳定公式

练习：写每 block 一行的 FP32 softmax：max → exp(x-max) → sum → 除法；行长先 128/256/1024

验收：与 torch.softmax 比较；测试 ±1000、大量相同值；说明减 max 防溢出，暂不支持 NaN/Inf

自己的代码保存在这里；阅读入口相对项目根目录。
每日约 60 分钟，见 [完整计划](../../PLAN.md)。
