"""
第 2 步：手工反向传播 —— 链式法则到底在算什么

运行：
    python 02_backward_manual.py

这一节不调用 y.backward()，而是自己一步一步把梯度传回去。
表达式：
    m = x * 2
    a = m + 1
    y = relu(a)

在 x = 1.0 时：
    m = 2, a = 3, y = 3
    dy/dx = dy/da * da/dm * dm/dx
          = 1 * 1 * 2
          = 2
"""

from micrograd.engine import Value


def numerical_grad(f, x0, eps=1e-6):
    """中心差分：用两个很近的点估计导数。"""
    return (f(x0 + eps) - f(x0 - eps)) / (2 * eps)


def scalar_example():
    print("=" * 60)
    print("例 1：y = relu(2x + 1)，在 x = 1 处")
    print("=" * 60)

    x = Value(1.0)
    two = Value(2.0)
    one = Value(1.0)

    m = x * two
    a = m + one
    y = a.relu()

    print(f"前向：m = {m.data}, a = {a.data}, y = {y.data}")

    # ---- 手工反向传播 ----
    # 1) 输出对自己的梯度是 1
    y.grad = 1.0

    # 2) ReLU：a > 0 时梯度直通，否则为 0
    a.grad += y.grad if a.data > 0 else 0.0

    # 3) 加法：梯度原样分给两个输入
    m.grad += a.grad
    one.grad += a.grad

    # 4) 乘法：m = x * two
    #    dm/dx = two.data, dm/dtwo = x.data
    x.grad += two.data * m.grad
    two.grad += x.data * m.grad

    print("反向：")
    print(f"  y.grad   = {y.grad}")
    print(f"  a.grad   = {a.grad}   # ReLU 直通")
    print(f"  m.grad   = {m.grad}   # 加法原样分发")
    print(f"  one.grad = {one.grad}")
    print(f"  x.grad   = {x.grad}   # 2.0")
    print(f"  two.grad = {two.grad} # 1.0")

    def f(t):
        z = 2.0 * t + 1.0
        return max(0.0, z)

    print(f"  数值梯度 dy/dx ≈ {numerical_grad(f, 1.0):.6f}")
    print("  手工梯度与数值梯度一致。")
    print()


def dead_relu_example():
    print("=" * 60)
    print("例 2：ReLU 关掉时，梯度会被“闸门”挡住")
    print("=" * 60)

    x = Value(-1.0)
    two = Value(2.0)
    one = Value(1.0)
    m = x * two
    a = m + one
    y = a.relu()

    print(f"前向：m = {m.data}, a = {a.data}, y = {y.data}")

    y.grad = 1.0
    a.grad += y.grad if a.data > 0 else 0.0
    m.grad += a.grad
    one.grad += a.grad
    x.grad += two.data * m.grad
    two.grad += x.data * m.grad

    print(f"反向：a = {a.data} < 0，所以 a.grad = {a.grad}")
    print(f"      x.grad = {x.grad}（梯度被 ReLU 截断了）")
    print()


def diamond_example():
    print("=" * 60)
    print("例 3：同一个变量被用了两次，梯度必须累加")
    print("=" * 60)

    x = Value(3.0)
    a = x * x       # x 在乘法里出现两次
    y = a + x       # x 又通过加法出现一次

    print(f"前向：a = {a.data}, y = {y.data}")
    print("解析解：y = x^2 + x，所以 dy/dx = 2x + 1 = 7")

    y.grad = 1.0

    # y = a + x
    a.grad += y.grad
    x.grad += y.grad          # 来自 y = a + x 的那条路径

    # a = x * x：两条路径都指向同一个 x
    x.grad += x.data * a.grad # 左乘数
    x.grad += x.data * a.grad # 右乘数

    print(f"x.grad = {x.grad}（3 + 3 + 1 = 7）")
    print("如果第二次写成 x.grad = ... 而不是 +=，最后只会保留一条乘法路径的 3，")
    print("或者再加 1 得到 4（取决于覆盖顺序），总之会漏掉一条路径。")
    print()


def main():
    scalar_example()
    dead_relu_example()
    diamond_example()
    print("小结：")
    print("  - 加法：梯度原样分发")
    print("  - 乘法：梯度乘以另一个乘数")
    print("  - ReLU：正数直通，负数截断")
    print("  - 多条路径：必须 += 累加")
    print("手动写这些太繁琐，下一步让每个节点自己记住怎么反向传播。")


if __name__ == "__main__":
    main()
