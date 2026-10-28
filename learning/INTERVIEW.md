# 面试速查与模拟题

- **thread/block/grid/warp？** kernel 由 grid 中的 block 执行，block 包含线程；warp 是 32 个线程的执行组织。block 可共享 shared memory 并同步；不要假定不同 block 的执行顺序。
- **global/shared/register/local？** global 是设备内存地址空间；shared 是 block 内显式管理的片上存储；register 保存线程局部值；local 是线程私有地址空间，溢出的值可能落到设备内存，不意味着片上高速。
- **合并访存？** 同一 warp 的线程访问相邻地址，通常能用更少事务完成。float4 是单线程向量化访问，需要对齐和尾部处理，不能替代合并访存。
- **同步怎么选？** `__syncthreads()` 是 block 范围，不能只让部分线程绕开同一次屏障；shuffle 参与掩码需匹配活跃线程；多个 block 的归约可用第二次 kernel launch。
- **为什么减去 max？** softmax 的分子分母同时缩放，有限输入下数学值不变，同时避免大正数 exp 溢出。
- **为什么优化没变快？** shape 太小受 launch/提交影响，瓶颈不在改动处，寄存器或同步开销增加，或者缓存/频率波动掩盖收益。
- **occupancy 越高越好？** 不一定；活跃 warp 有助于隐藏延迟，但寄存器复用、指令和访存效率同样影响性能。
- **怎么证明更快？** 同 dtype、同 shape、同语义、同精度要求，先验证，再排除编译与分配，多次 Event 测量，明确 cache 状态、硬件和局限。
- **GEMM 分块为什么快？** 将 tile 放入 shared memory，让多个输出复用加载的数据，减少重复 global load；性能仍可能远低于 cuBLAS。
- **为什么 PyTorch 快而自写慢？** 库有成熟的分块/向量化/硬件特化；Python launch、小 shape 和不合适的 tile 也会影响表现。

## 2 分钟项目讲述模板

“我在 RTX 3050 / WSL 上，参考 vitamin-cuda 实现了 ___。先用 torch 验证 ___ 输入，再用 CUDA Event 比较 ___。基于 ___ 的访存/同步分析，我改了 ___。在 ___ shape 上从 ___ ms 到 ___ ms，在 ___ 上没有收益，原因推测是 ___。目前支持 ___，不支持 ___。”

数字必须来自 results 中实际记录。追问时准备展示代码、计时方法和未获收益的实验。
