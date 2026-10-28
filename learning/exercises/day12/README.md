# 第 12 天

阅读：同上 shared-memory 分块思路

练习：写 16×16 shared tiled GEMM，加载越界补零，所有线程参与同步

验收：比较 naive/tiled/torch，记录 M/N/K；解释数据复用和同步，性能不要求超过 cuBLAS

自己的代码保存在这里；阅读入口相对项目根目录。
每日约 60 分钟，见 [完整计划](../../PLAN.md)。
