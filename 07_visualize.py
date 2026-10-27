"""
第 7 步：把计算图画出来 —— 看清前向和反向的每个节点

运行：
    python 07_visualize.py

原版教程用 graphviz 的 draw_dot，这里不依赖 graphviz，
用 matplotlib 画同样的小图：
    x = 1.0
    y = relu(2x + 1)

每个方框里：
    上排：data（前向数值）
    下排：grad（反向梯度）
    方框上方：产生这个节点的运算
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from micrograd.engine import Value

HERE = Path(__file__).resolve().parent
FIGS = HERE / "figs"
FIGS.mkdir(exist_ok=True)


def trace(root):
    """返回所有节点和所有边（child -> parent）。"""
    nodes, edges = set(), set()

    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)

    build(root)
    return nodes, edges


def depth_map(nodes):
    """节点的层号：叶子是 0，父节点比所有子节点都大。"""
    depth = {}

    def get(v):
        if v not in depth:
            depth[v] = 0 if not v._prev else 1 + max(get(c) for c in v._prev)
        return depth[v]

    for n in nodes:
        get(n)
    return depth


def draw(root, path):
    nodes, edges = trace(root)
    depth = depth_map(nodes)

    columns = {}
    for n in nodes:
        columns.setdefault(depth[n], []).append(n)

    pos = {}
    for d, col in columns.items():
        # set 的顺序不固定；用 _order 固定叶子的上下位置，避免箭头穿过方框
        col.sort(key=lambda n: (getattr(n, "_order", 99), n.data))
        for i, n in enumerate(col):
            y = (i - (len(col) - 1) / 2.0) * 1.45
            pos[n] = (d * 2.5, y)

    fig, ax = plt.subplots(figsize=(11, 6))

    for n in nodes:
        x, y = pos[n]
        ax.add_patch(FancyBboxPatch(
            (x - 0.8, y - 0.42), 1.6, 0.84,
            boxstyle="round,pad=0.04",
            facecolor="#f4f7fb", edgecolor="#2f4b7c", linewidth=1.4, zorder=2,
        ))
        label = getattr(n, "name", None) or f"{n.data:.3f}"
        ax.text(x, y + 0.14, label, ha="center", va="center",
                fontsize=12, fontweight="bold", zorder=3)
        ax.text(x, y - 0.18, f"grad {n.grad:.3f}", ha="center", va="center",
                fontsize=9, color="#b22222", zorder=3)
        if n._op:
            ax.text(x, y + 0.58, n._op, ha="center", va="bottom",
                    fontsize=11, color="#2f4b7c", zorder=3)

    for child, parent in edges:
        x1, y1 = pos[child]
        x2, y2 = pos[parent]
        ax.add_patch(FancyArrowPatch(
            (x1 + 0.82, y1), (x2 - 0.82, y2),
            arrowstyle="-|>", mutation_scale=14,
            color="#666666", linewidth=1.2,
            connectionstyle="arc3,rad=0.06", zorder=1,
        ))

    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    ax.set_xlim(min(xs) - 1.3, max(xs) + 1.3)
    ax.set_ylim(min(ys) - 1.0, max(ys) + 1.2)
    ax.axis("off")
    ax.set_title("computation graph:  y = relu(2x + 1)   (data | grad)", fontsize=13)

    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"图已保存：{path}")


def main():
    x = Value(1.0)
    two = Value(2.0)
    one = Value(1.0)

    x.name = "x = 1.000"
    two.name = "2"
    one.name = "1"
    # 固定三片叶子的上下顺序：1 在底、x 在中、2 在上
    one._order, x._order, two._order = 0, 1, 2

    y = (x * two + one).relu()
    y.backward()

    print("前向：y =", y.data)
    print("反向：x.grad =", x.grad)
    print()
    print("每个节点：")
    for v in trace(y)[0]:
        print(f"  op={v._op or 'leaf':>5}  data={v.data:6.3f}  grad={v.grad:6.3f}")

    draw(y, FIGS / "computation_graph.png")


if __name__ == "__main__":
    main()
