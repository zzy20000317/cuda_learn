# 第 3 天

阅读：`learning/exercises/bench_add.py`

练习：加入 CUDA Event 计时、预热；比较 128/256/512 线程，N 从 1K 到 4M

验收：保存时间和 shape；解释为什么不能只用 Python wall time、首次编译不计入

自己的代码保存在这里；阅读入口相对项目根目录。
每日约 60 分钟，见 [完整计划](../../PLAN.md)。




一次add要读a,b,写c：12字节/元素
总的字节数 = 12*n
ms -> s 要除1000，GB -> B要除1e9
结果：N=1024时kernel launch开销主导，block大小影响大
N=4M时带宽主导，向量化收益明显.
"""
rows = []
for n in [1024, 65536, 1048576, 4194304]:
    a = torch.randn(n, device="cuda")
    b = torch.randn_like(a)
    c = torch.empty_like(a)
    baseline = milliseconds(lambda: torch.add(a, b, out=c))
    for block in [128, 256, 512]:
        for vec in [False, True]:
            ms = milliseconds(lambda: lib.add(a, b, c, block, vec))
            torch.testing.assert_close(c, a+b)
            row = dict(n=n, block=block, vectorized=vec,
                       ms=ms, torch_ms=baseline,
                       speedup=baseline/ms,
                       effective_GB_s=12*n/(ms*1e6))
            rows.append(row)
            print(row)

"""


