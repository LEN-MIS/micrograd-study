"""
第 3 步：自动反向传播 —— 一行 y.backward() 背后发生了什么

运行：
    python 03_autograd.py

这一节用 micrograd.engine.Value：
    1. 复现 README 里的复杂表达式，看数值结果；
    2. 用数值梯度验证 autograd 算得对不对；
    3. 看“重复 backward 会累加梯度”这个坑。
"""

from micrograd.engine import Value

EPS = 1e-6


def expression(a, b):
    """README 里的那段表达式，返回输出节点 g。"""
    c = a + b
    d = a * b + b**3
    c += c + 1
    c += 1 + c + (-a)
    d += d * 2 + (b + a).relu()
    d += 3 * d + (b - a).relu()
    e = c - d
    f = e**2
    g = f / 2.0
    g += 10.0 / f
    return g


def numerical_partial(which, a0=-4.0, b0=2.0):
    """用中心差分估计 g 对 a 或 b 的偏导。"""
    if which == "a":
        gp = expression(Value(a0 + EPS), Value(b0)).data
        gm = expression(Value(a0 - EPS), Value(b0)).data
    else:
        gp = expression(Value(a0), Value(b0 + EPS)).data
        gm = expression(Value(a0), Value(b0 - EPS)).data
    return (gp - gm) / (2 * EPS)


def readme_example():
    print("=" * 62)
    print("例 1：README 里的复杂表达式")
    print("=" * 62)

    a = Value(-4.0)
    b = Value(2.0)
    g = expression(a, b)

    print(f"前向：g = {g.data:.4f}")
    g.backward()
    print(f"反向：dg/da = {a.grad:.4f}")
    print(f"      dg/db = {b.grad:.4f}")
    print()

    print("数值梯度校验（中心差分）：")
    print(f"  dg/da ≈ {numerical_partial('a'):.4f}")
    print(f"  dg/db ≈ {numerical_partial('b'):.4f}")
    print("两者一致，说明链式法则 + 拓扑排序的实现是对的。")
    print()


def diamond_example():
    print("=" * 62)
    print("例 2：自动处理多条路径")
    print("=" * 62)

    x = Value(3.0)
    y = x * x + x
    y.backward()
    print(f"y = x^2 + x，x = 3，dy/dx 应为 7，实际 = {x.grad}")
    print()


def accumulate_example():
    print("=" * 62)
    print("例 3：连续调用 backward，梯度会累加")
    print("=" * 62)

    x = Value(2.0)
    y = x * x

    y.backward()
    print(f"第 1 次 backward 后 x.grad = {x.grad}（应为 4）")
    y.backward()
    print(f"第 2 次 backward 后 x.grad = {x.grad}（应为 8，不是 4）")
    print("所以训练循环里每步必须先 zero_grad()，否则梯度会越加越大。")
    print()


def main():
    readme_example()
    diamond_example()
    accumulate_example()
    print("小结：")
    print("  - backward() 先做拓扑排序，再从输出往输入逆序调用每个节点的 _backward")
    print("  - 输出节点的 grad 被设为 1，表示 dL/dL = 1")
    print("  - 每个节点用 += 把梯度累加给子节点")


if __name__ == "__main__":
    main()
