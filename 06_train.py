"""
第 6 步：完整训练 —— 数据、损失、反向传播、SGD 更新

运行：
    python 06_train.py

任务：二维“双月”数据集二分类。
    输入 x = (x1, x2)
    标签 y ∈ {-1, +1}
    模型 MLP(2, [16, 16, 1])
    损失 SVM max-margin hinge loss + L2 正则

训练循环只有五步：
    1. forward：算分数和损失
    2. zero_grad：清空上一步的梯度
    3. backward：反向传播，得到每个参数的梯度
    4. update：p.data -= lr * p.grad
    5. log：记录 loss / accuracy

脚本会生成 figs/training_result.png。
"""

import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # 不弹窗，直接存图
import matplotlib.pyplot as plt
import numpy as np

from micrograd.engine import Value
from micrograd.nn import MLP

HERE = Path(__file__).resolve().parent
FIGS = HERE / "figs"
FIGS.mkdir(exist_ok=True)


def make_moons(n_samples=100, noise=0.1, seed=0):
    """不依赖 sklearn 的双月数据集，和 sklearn.datasets.make_moons 形状一致。"""
    rng = np.random.default_rng(seed)
    n_outer = n_samples // 2
    n_inner = n_samples - n_outer

    t = np.linspace(0, np.pi, n_outer)
    outer = np.column_stack([np.cos(t), np.sin(t)])
    inner = np.column_stack([1 - np.cos(t[:n_inner]), 0.5 - np.sin(t[:n_inner])])

    X = np.vstack([outer, inner])
    y = np.hstack([np.ones(n_outer), -np.ones(n_inner)])
    X = X + rng.normal(0, noise, X.shape)

    idx = rng.permutation(n_samples)
    return X[idx], y[idx]


def decision_grid(model, X, h=0.25):
    """在二维网格上逐点前向，返回网格坐标和模型分数。"""
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    Z = np.zeros_like(xx)
    for i in range(xx.shape[0]):
        for j in range(xx.shape[1]):
            Z[i, j] = model([Value(xx[i, j]), Value(yy[i, j])]).data
    return xx, yy, Z


def plot_result(X, y, losses, accs, grid_before, grid_after, path):
    xx0, yy0, Z0 = grid_before
    xx1, yy1, Z1 = grid_after

    fig, axes = plt.subplots(2, 2, figsize=(11, 9.5))
    fig.suptitle("micrograd: training a 2-layer MLP on two moons", fontsize=14)

    # (a) 数据
    ax = axes[0, 0]
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", s=35, edgecolors="k", linewidths=0.4)
    ax.set_title("dataset: y = +1 (red) / -1 (blue)")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_aspect("equal")

    # (b) 训练曲线
    ax = axes[0, 1]
    ax.plot(losses, color="crimson", lw=2, label="training loss")
    ax.set_xlabel("step")
    ax.set_ylabel("loss", color="crimson")
    ax.tick_params(axis="y", labelcolor="crimson")
    ax.grid(alpha=0.3)

    ax2 = ax.twinx()
    ax2.plot([a * 100 for a in accs], color="navy", lw=1.5, ls="--", label="accuracy %")
    ax2.set_ylabel("accuracy (%)", color="navy")
    ax2.tick_params(axis="y", labelcolor="navy")
    ax2.set_ylim(-5, 105)
    ax.set_title("loss and accuracy")

    # (c)(d) 决策边界
    for ax, (xx, yy, Z), title in [
        (axes[1, 0], grid_before, "before training"),
        (axes[1, 1], grid_after, "after 100 steps"),
    ]:
        ax.contourf(xx, yy, Z, levels=24, cmap="coolwarm", alpha=0.55)
        ax.contour(xx, yy, Z, levels=[0], colors="k", linewidths=1.5)
        ax.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", s=30,
                   edgecolors="k", linewidths=0.4)
        ax.set_title(f"decision boundary: {title}")
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.set_aspect("equal")

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n图已保存：{path}")


def main():
    random.seed(1337)
    np.random.seed(1337)

    X, y = make_moons(n_samples=100, noise=0.1, seed=1337)
    print(f"数据：X.shape = {X.shape}, y.shape = {y.shape}")
    print(f"标签取值：{np.unique(y)}")
    print()

    model = MLP(2, [16, 16, 1])
    print(f"模型：{model}")
    print(f"参数个数：{len(model.parameters())}")
    print()

    def loss_fn(batch_size=None):
        # 1) 选 batch
        if batch_size is None:
            Xb, yb = X, y
        else:
            idx = np.random.permutation(X.shape[0])[:batch_size]
            Xb, yb = X[idx], y[idx]

        # 2) 前向：numpy -> Value -> 模型 -> 分数
        inputs = [[Value(xi) for xi in row] for row in Xb]
        scores = [model(row) for row in inputs]

        # 3) SVM hinge loss：max(0, 1 - y*s)
        losses = [(1.0 + -yi * si).relu() for yi, si in zip(yb, scores)]
        data_loss = sum(losses) * (1.0 / len(losses))

        # 4) L2 正则：让权重不要太大
        alpha = 1e-4
        reg_loss = alpha * sum((p * p for p in model.parameters()))

        # 5) 总损失 + 准确率
        total_loss = data_loss + reg_loss
        acc = sum((yi > 0) == (si.data > 0) for yi, si in zip(yb, scores)) / len(yb)
        return total_loss, acc

    # 训练前的决策边界（用未训练的随机参数）
    grid_before = decision_grid(model, X)

    losses, accs = [], []
    n_steps = 100

    print("开始训练：")
    print(f"{'step':>4}  {'loss':>8}  {'acc':>7}  {'lr':>5}")
    for k in range(n_steps):
        # 1) 清空梯度（必须在 backward 前）
        model.zero_grad()

        # 2) 前向 + 损失
        total_loss, acc = loss_fn()

        # 3) 反向传播
        total_loss.backward()

        # 4) SGD 更新：w <- w - lr * dw
        lr = 1.0 - 0.9 * k / n_steps
        for p in model.parameters():
            p.data -= lr * p.grad

        losses.append(total_loss.data)
        accs.append(acc)

        if k % 10 == 0 or k == n_steps - 1:
            print(f"{k:4d}  {total_loss.data:8.4f}  {acc * 100:6.1f}%  {lr:5.2f}")

    grid_after = decision_grid(model, X)
    final_loss, final_acc = loss_fn()

    print()
    print(f"训练结束：loss = {final_loss.data:.4f}, accuracy = {final_acc * 100:.1f}%")
    print(f"参数个数：{len(model.parameters())}")
    print("注意：这是训练集准确率，不代表泛化能力。")

    plot_result(X, y, losses, accs, grid_before, grid_after,
                FIGS / "training_result.png")


if __name__ == "__main__":
    main()
