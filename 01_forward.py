"""
第 1 步：前向传播 —— 一个表达式如何变成一张计算图

运行：
    python 01_forward.py

这一节只看"前向"：
    x = 1.0
    y = (x * 2 + 1).relu()

重点观察：
    1. Value 把每个标量包成一个节点；
    2. 每次 +、*、relu 都会创建一个新节点，并记住它的子节点 _prev；
    3. 此时所有 grad 都是 0，因为反向传播还没有发生。
"""

from micrograd.engine import Value


def topo_sort(root):
    """DFS 后序：先访问所有子节点，再访问自己。"""
    order, visited = [], set()

    def dfs(v):
        if v in visited:
            return
        visited.add(v)
        for child in v._prev:
            dfs(child)
        order.append(v)

    dfs(root)
    return order


def describe(v):
    op = v._op if v._op else "leaf"
    return f"{op:>5} | data={v.data:7.3f} | grad={v.grad:5.2f}"


def main():
    x = Value(1.0)
    y = (x * 2 + 1).relu()

    print("y =", y)
    print("y._op =", y._op)
    print("y._prev 里有", len(y._prev), "个直接子节点")
    print()

    print("从输出 y 往上走，能看到的整张图（拓扑序：先叶子，后输出）：")
    for i, v in enumerate(topo_sort(y)):
        print(f"  {i:2d}. {describe(v)}")
    print()

    print("y 的直接子节点：")
    for v in y._prev:
        print("   ", v, " op =", v._op or "leaf")
    print()

    print("注意：所有 grad 都是 0，因为还没有调用 backward()。")
    print("x 是叶子节点，data 是 1.0；常量 2 和 1 也被包装成了 Value。")
    print()
    print("小实验：把下面的表达式改成 x * 3 - 2，再打印一次拓扑序。")
    print("       你会发现图变长了，但结构完全一样。")


if __name__ == "__main__":
    main()
