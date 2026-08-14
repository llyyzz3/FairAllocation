from .baselines import EnvyCycleElimination


class KRoundRobinWithECE:
    """
    Algorithm 2: k-RoundRobin with EnvyCycleElimination

    流程:
    1. 固定 agent 顺序 I = [0, 1, ..., n-1]
    2. 执行 k 轮 Round-Robin
    3. 对剩余 goods 运行 Envy Cycle Elimination

    理论保证:
    k / (k + 1)-EFkX allocation
    """

    def __init__(self, k=1, order=None):
        if k < 1:
            raise ValueError("k must be at least 1.")

        self.k = k
        self.order = order
        self.eval_k = k  # 用于实验评估 MaxAlphaEFkX
        self.name = f"{k}-RoundRobin with ECE"

    def run(self, instance):
        n = instance.n
        m = instance.m
        valuations = instance.valuations

        # 自动生成 agent 顺序
        if self.order is None:
            order = list(range(n))
        else:
            order = list(self.order)

        # 检查 order 是否合法
        if len(order) != n:
            raise ValueError(
                f"Invalid order length: expected {n}, got {len(order)}. "
                f"For n={n}, order should contain agents 0 to {n - 1}."
            )

        if set(order) != set(range(n)):
            raise ValueError(
                f"Invalid order: {order}. "
                f"It must contain each agent exactly once from 0 to {n - 1}."
            )

        allocation = {i: [] for i in range(n)}
        remaining_items = list(range(m))

        # -----------------------------
        # Part 1: k rounds of Round-Robin
        # -----------------------------
        for _ in range(self.k):
            if not remaining_items:
                return allocation

            for agent in order:
                if not remaining_items:
                    break

                best_item = max(
                    remaining_items,
                    key=lambda item: valuations[agent][item]
                )

                allocation[agent].append(best_item)
                remaining_items.remove(best_item)

        # -----------------------------
        # Part 2: Envy Cycle Elimination
        # -----------------------------
        ece = EnvyCycleElimination()
        allocation = ece.run_from_partial(
            instance=instance,
            allocation=allocation,
            remaining_items=remaining_items
        )

        return allocation