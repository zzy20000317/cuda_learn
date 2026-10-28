"""测试 float4 add：奇数长度、偏移切片、对齐、非对齐。
运行：python exercises/test_add.py
"""
from pathlib import Path
import torch
from torch.utils.cpp_extension import load

ROOT = Path(__file__).resolve().parents[1]
lib = load(
    name="learning_add_test",
    sources=[str(Path(__file__).with_name("add.cu"))],
    extra_cuda_cflags=["-O3"],
    extra_cflags=["-O3"],
)

def run_one(a, b, block, vec):
    c = torch.empty_like(a)
    lib.add(a, b, c, block, vec)
    torch.testing.assert_close(c, a + b)

def main():
    torch.manual_seed(0)

    # 1. 奇数长度：N 不是 4 的倍数
    for n in [1, 2, 3, 5, 7, 127, 129, 257, 1025, 65537]:
        a = torch.randn(n, device="cuda")
        b = torch.randn_like(a)
        for block in [128, 256, 512]:
            for vec in [False, True]:
                run_one(a, b, block, vec)
    print("PASS: 奇数长度")

    # 2. 偏移切片：起始地址非 16 字节对齐
    for offset in [1, 2, 3, 5, 7]:
        n = 257
        a = torch.randn(n + offset, device="cuda")[offset:]
        b = torch.randn_like(a)
        # 检查对齐状态
        aligned = (a.data_ptr() % 16 == 0)
        for block in [128, 256]:
            for vec in [False, True]:
                run_one(a, b, block, vec)
        print(f"PASS: 偏移 {offset}，a.data_ptr()%16={a.data_ptr()%16}，aligned={aligned}")

    # 3. 完全对齐的基准
    for n in [4, 8, 256, 1024, 1025, 65536]:
        a = torch.randn(n, device="cuda")
        b = torch.randn_like(a)
        run_one(a, b, 256, False)
        run_one(a, b, 256, True)
    print("PASS: 对齐场景")

    # 4. 对齐但 N 不是 4 的倍数（尾部 1~3 个）
    for n in [5, 6, 7, 255, 257, 1023, 1025]:
        a = torch.randn(n, device="cuda")
        b = torch.randn_like(a)
        run_one(a, b, 256, True)
    print("PASS: 对齐 + 尾部")

    print("\nALL PASS")

if __name__ == "__main__":
    main()