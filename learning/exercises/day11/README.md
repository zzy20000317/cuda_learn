# 第 11 天

阅读：`kernels/sgemm/sgemm.cu` 中最简单 SIMT 版本

练习：写 naive FP32 GEMM，先 128×128；使用 torch.mm 作参考

验收：非方阵和非 tile 整除 shape 正确；解释 C[m,n] 的 K 维循环

自己的代码保存在这里；阅读入口相对项目根目录。
每日约 60 分钟，见 [完整计划](../../PLAN.md)。
