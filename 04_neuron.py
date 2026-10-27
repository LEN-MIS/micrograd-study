"""
第 4 步：从标量到神经元 —— y = relu(w·x + b)

运行：
    python 04_neuron.py

一个神经元只做两件事：
    1. 线性组合：z = w1*x1 + w2*x2 + b
    2. 非线性激活：a = relu(z)

它就是一个很小的计算图。backward() 会算出每个 w 和 b 的梯度。
"""

import random

from micrograd.engine import Value
from micrograd.nn import Neuron


def main():
    random.seed(1337)

    print("=" * 62)
    print("例 1：一个有 2 个输入的 ReLU 神经元")
    print("=" * 62)

    n = Neuron(2)
    # 为了手算方便，把随机初始化的参数改成固定值
    n.w[0].data = 1.5
    n.w[1].data = -0.5
    n.b.data = 0.25

    x = [Value(1.0), Value(-2.0)]
    y = n(x)

    print(f"神经元结构：{n}")
    print(f"参数：w = [{n.w[0].data}, {n.w[1].data}], b = {n.b.data}")
    print(f"输入：x = [{x[0].data}, {x[1].data}]")
    print("手算：z = 1.5*1 + (-0.5)*(-2) + 0.25 = 2.75")
    print(f"输出：y = relu(z) = {y.data}")
    print()

    y.backward()
    print("反向传播（z > 0，所以 ReLU 直通）：")
    print(f"  dy/dw1 = x1 = {n.w[0].grad}")
    print(f"  dy/dw2 = x2 = {n.w[1].grad}")
    print(f"  dy/db  = 1  = {n.b.grad}")
    print("这三个梯度告诉你：w1 增加一点，y 会按 x1 的比例增加。")
    print()

    print("=" * 62)
    print("例 2：z < 0 时，ReLU 把梯度全部截断（dead ReLU）")
    print("=" * 62)

    dead = Neuron(2)
    dead.w[0].data = -1.0
    dead.w[1].data = -1.0
    dead.b.data = -1.0

    out = dead([Value(1.0), Value(1.0)])
    out.backward()
    print(f"z = -1*1 + -1*1 + -1 = -3, y = relu(z) = {out.data}")
    print(f"梯度：w = [{dead.w[0].grad}, {dead.w[1].grad}], b = {dead.b.grad}")
    print("全是 0：这个神经元当前对损失没有任何影响，参数也就不会更新。")
    print("如果学习率太大，神经元可能长期停在负数区，这就是 dead ReLU 现象。")
    print()

    print("=" * 62)
    print("例 3：去掉 ReLU，就是一个线性神经元")
    print("=" * 62)

    lin = Neuron(2, nonlin=False)
    lin.w[0].data = 1.0
    lin.w[1].data = 1.0
    lin.b.data = 0.0
    z = lin([Value(2.0), Value(3.0)])
    print(f"线性神经元输出：{z.data}（2 + 3 = 5）")
    print("线性神经元只能表达直线/平面；要拟合弯曲边界，需要 ReLU 这样的非线性。")
    print()

    print("小结：")
    print("  一个神经元 = 一次点积 + 一次加偏置 + 一次激活")
    print("  参数个数 = 输入维度 + 1（偏置）")
    print("  这些参数就是训练时要学的东西")


if __name__ == "__main__":
    main()
