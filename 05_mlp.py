"""
第 5 步：层与 MLP —— 把神经元拼成网络

运行：
    python 05_mlp.py

MLP(2, [16, 16, 1]) 的意思是：
    输入 2 维
    -> 16 个 ReLU 神经元（第 1 个隐藏层）
    -> 16 个 ReLU 神经元（第 2 个隐藏层）
    -> 1 个线性神经元（输出层，输出一个分数 logit）

参数量：
    (2*16 + 16) + (16*16 + 16) + (16*1 + 1)
    = 48 + 272 + 17
    = 337
"""

import random

from micrograd.engine import Value
from micrograd.nn import MLP


def main():
    random.seed(1337)

    print("=" * 62)
    print("例 1：网络结构")
    print("=" * 62)

    model = MLP(2, [16, 16, 1])
    print(model)
    print()

    for i, layer in enumerate(model.layers):
        n_in = len(layer.neurons[0].w)
        n_out = len(layer.neurons)
        kind = "ReLU" if layer.neurons[0].nonlin else "Linear"
        print(f"  Layer {i}: {kind:6s}  {n_in:2d} -> {n_out:2d}  "
              f"参数 {n_out * (n_in + 1)}")
    print()

    params = model.parameters()
    print(f"参数总数：{len(params)}")
    print(f"手算：48 + 272 + 17 = {48 + 272 + 17}")
    print()

    print("最后一层为什么是 Linear？")
    print("  因为损失函数直接使用输出分数 s。如果最后加 ReLU，s >= 0，")
    print("  当真实标签 y = -1 时就无法得到 s <= -1 的正确分数。")
    print()

    print("=" * 62)
    print("例 2：一次前向传播")
    print("=" * 62)

    x = [Value(0.5), Value(-1.2)]
    score = model(x)
    print(f"输入 x = [0.5, -1.2]")
    print(f"输出 score = {score.data:.4f}")
    print("score > 0 预测 +1，score < 0 预测 -1。")
    print()

    print("=" * 62)
    print("例 3：一次处理一个 batch")
    print("=" * 62)

    batch = [
        [Value(0.5), Value(-1.2)],
        [Value(-0.3), Value(0.8)],
        [Value(1.0), Value(1.0)],
    ]
    scores = [model(row) for row in batch]
    print(f"batch 大小：{len(scores)}")
    print("scores =", [round(s.data, 4) for s in scores])
    print()

    print("=" * 62)
    print("例 4：形状是怎么流动的")
    print("=" * 62)
    print("  x: 2 维")
    print("  -> Layer(2, 16): 每个神经元看 2 个输入，输出 16 个数")
    print("  -> Layer(16, 16): 每个神经元看 16 个输入，输出 16 个数")
    print("  -> Layer(16, 1):  1 个神经元看 16 个输入，输出 1 个数")
    print()
    print("每个隐藏层同时计算多个神经元，它们共享输入、各自有独立的 w 和 b。")
    print("这种层也叫全连接层（fully connected / dense layer）。")


if __name__ == "__main__":
    main()
