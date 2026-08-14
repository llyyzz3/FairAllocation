import numpy as np

class FairnessMetric:
    """
    评估指标的基类。
    所有具体的公平性或效率指标都应继承此类并实现 compute 方法。
    """

    def __init__(self, name):
        self.name = name

    def compute(self, instance, allocation):
        """
        核心计算逻辑。
        :param instance: 包含代理人数(n)、物品数(m)和估值矩阵(valuations)的对象
        :param allocation: 字典格式的分配结果 {agent_index: [item_indices]}
        :return: 指标的具体数值或布尔值
        """
        raise NotImplementedError("子类必须实现 compute 方法")


class EF1Checker(FairnessMetric):
    """
    EF1 (Envy-Free up to One Item) 检查器。
    定义：对于任何代理 i 和 j，如果移除 j 的分配包中的【至少一个】物品后，
    代理 i 不再嫉妒 j，则满足 EF1。
    """

    def __init__(self):
        super().__init__("EF1")

    def compute(self, instance, allocation):
        valuations = instance.valuations
        n = instance.n

        for i in range(n):
            for j in range(n):
                if i == j: continue

                # 计算代理 i 对自己包的价值
                val_i_own = sum(valuations[i][item] for item in allocation[i])
                # 计算代理 i 对代理 j 包的价值
                val_i_others = sum(valuations[i][item] for item in allocation[j])

                # 如果产生嫉妒 (i 对 j 的包估值更高)
                if val_i_own < val_i_others:
                    # 如果 j 的包本身就是空的但 i 仍嫉妒，则不满足（加性估值下通常不会发生）
                    if not allocation[j]:
                        return False

                    satisfied_after_removal = False
                    # 尝试从 j 的包中移除每一个物品
                    for item_in_j in allocation[j]:
                        # 检查移除该物品后，嫉妒是否消失
                        if val_i_own >= (val_i_others - valuations[i][item_in_j]):
                            satisfied_after_removal = True
                            break

                    # 如果没有任何一个物品移除后能消除嫉妒，则该分配不满足 EF1
                    if not satisfied_after_removal:
                        return False
        return True


class EFXChecker(FairnessMetric):
    """
    EFX (Envy-Free up to Any Item) 检查器。
    定义：对于任何代理 i 和 j，从 j 的分配包中移除【任意】一个被 i 视为正价值的物品后，
    代理 i 都不再嫉妒 j，则满足 EFX。
    """

    def __init__(self):
        super().__init__("EFX")

    def compute(self, instance, allocation):
        valuations = instance.valuations
        n = instance.n

        for i in range(n):
            for j in range(n):
                if i == j: continue

                val_i_own = sum(valuations[i][item] for item in allocation[i])
                val_i_others = sum(valuations[i][item] for item in allocation[j])

                # 如果产生嫉妒
                if val_i_own < val_i_others:
                    # 检查 j 包里每一个对 i 具有正价值的物品
                    for item_in_j in allocation[j]:
                        # 只有当 i 对该物品的估值 > 0 时，才要求移除后消除嫉妒
                        if valuations[i][item_in_j] > 0:
                            # 如果移除【任何一个】物品后仍存在嫉妒，则不满足 EFX
                            if val_i_own < (val_i_others - valuations[i][item_in_j]):
                                return False
        return True

import math


class EmpiricalEFXRatio(FairnessMetric):
    """
    Empirical EFX Ratio（经验 EFX 比率）。

    定义：在所有有序代理对 (i, j) 中，满足 EFX 条件的比例。
    EFX 条件：对于代理 i 和 j，移除 j 包中【任意一个】物品后，i 不再嫉妒 j。
    即：∀g ∈ alloc[j], v_i(alloc[i]) >= v_i(alloc[j] \ {g})

    返回值：float，范围 [0.0, 1.0]
        - 1.0 表示完全满足 EFX（所有代理对均满足）
        - 0.0 表示没有任何代理对满足 EFX
        - 跳过 i == j 的对角线对

    与 EFX bool checker 的区别：
        - bool checker 只返回"是否所有对都满足"
        - 本指标返回"有多少比例的对满足"，适合横向比较算法之间的差距
    """

    def __init__(self):
        super().__init__("EmpiricalEFXRatio")

    def compute(self, instance, allocation):
        valuations = instance.valuations
        n = instance.n

        total_pairs = 0
        satisfied_pairs = 0

        for i in range(n):
            val_i_own = sum(valuations[i][item] for item in allocation[i])

            for j in range(n):
                if i == j:
                    continue

                total_pairs += 1
                val_i_j = sum(valuations[i][item] for item in allocation[j])

                # i 不嫉妒 j，天然满足 EFX
                if val_i_own >= val_i_j:
                    satisfied_pairs += 1
                    continue

                # j 的包为空但 i 仍嫉妒（加性估值下极少发生），不满足
                if not allocation[j]:
                    continue

                # EFX 核心检查：移除 j 包中【每一个】物品后，i 均不再嫉妒
                # 注意：EFX 要求对"所有"物品移除后都不嫉妒，比 EF1 更强
                positive_goods = [
                    g
                    for g in allocation[j]
                    if valuations[i][g] > 0
                ]

                if not positive_goods:
                    satisfied_pairs += 1
                    continue

                efx_satisfied = all(
                    val_i_own >= val_i_j - valuations[i][g]
                    for g in positive_goods
                )

                if efx_satisfied:
                    satisfied_pairs += 1

        if total_pairs == 0:
            return 1.0  # 只有一个代理，定义为完全公平

        return satisfied_pairs / total_pairs

class MaxAlphaEFkX(FairnessMetric):
    """
    Maximum alpha-EFkX checker.

    默认 k=1，即 MaxAlphaEFX。

    对给定 allocation，计算它最大满足多少 alpha-EFkX：

        v_i(X_i) >= alpha * v_i(X_j \\ Y)

    其中：
        i != j
        Y 是 X_j 中任意 k 个 goods

    返回：
        float in [0, 1]

    解释：
        1.0 表示满足 exact EFkX
        0.667 表示满足 2/3-EFkX
        0.5 表示满足 1/2-EFkX
    """

    def __init__(self, k=1):
        if k < 1:
            raise ValueError("k must be at least 1.")

        self.k = k

        if k == 1:
            name = "MaxAlphaEFX"
        else:
            name = f"MaxAlphaEF{k}X"

        super().__init__(name)

    # def compute(self, instance, allocation):
    #     valuations = instance.valuations
    #     n = instance.n
    #     k = self.k
    #
    #     min_alpha = float("inf")
    #
    #     for i in range(n):
    #         # agent i 对自己 bundle 的估值
    #         val_i_own = sum(valuations[i][item] for item in allocation[i])
    #
    #         for j in range(n):
    #             if i == j:
    #                 continue
    #
    #             bundle_j = allocation[j]
    #
    #             # 如果 j 的 bundle 物品数 <= k，
    #             # 移除任意 k 个 goods 后，剩余 bundle 为空，
    #             # 这个 pair 自动满足 EFkX，不限制 alpha。
    #             if len(bundle_j) <= k:
    #                 continue
    #
    #             # agent i 对 j 的 bundle 的总估值
    #             val_i_j = sum(valuations[i][item] for item in bundle_j)
    #
    #             # EFkX 要求移除任意 k 个 goods 后都满足。
    #             # 对 additive non-negative valuation 来说，
    #             # 最难满足的情况是：移除 i 看来价值最小的 k 个 goods，
    #             # 因为这样 X_j \\ Y 的剩余价值最大。
    #             item_values_for_i = sorted(valuations[i][item] for item in bundle_j)
    #             removed_value = sum(item_values_for_i[:k])
    #             remaining_value = val_i_j - removed_value
    #
    #             if remaining_value <= 0:
    #                 continue
    #
    #             alpha = val_i_own / remaining_value
    #             min_alpha = min(min_alpha, alpha)
    #
    #     # 没有任何有效约束，说明没有 envy 压力，记为 1.0
    #     if min_alpha == float("inf"):
    #         return 1.0
    #
    #     # alpha 通常只关心 [0, 1]。
    #     # 如果算出来 > 1，说明 exact EFkX 已经满足，所以返回 1.0。
    #     return min(1.0, min_alpha)
    def compute(self, instance, allocation):
        valuations = instance.valuations
        n = instance.n
        k = self.k

        min_alpha = float("inf")

        for i in range(n):
            val_i_own = float(
                sum(valuations[i][item] for item in allocation[i])
            )

            for j in range(n):
                if i == j:
                    continue

                bundle_j = allocation[j]

                if len(bundle_j) <= k:
                    continue

                val_i_j = float(
                    sum(valuations[i][item] for item in bundle_j)
                )

                item_values_for_i = sorted(
                    float(valuations[i][item])
                    for item in bundle_j
                    if valuations[i][item] > 0
                )

                if len(item_values_for_i) <= k:
                    continue

                removed_value = sum(item_values_for_i[:k])
                remaining_value = val_i_j - removed_value

                if remaining_value <= 0:
                    continue

                alpha = float(val_i_own / remaining_value)
                min_alpha = min(min_alpha, alpha)

        if min_alpha == float("inf"):
            return 1.0

        return float(min(1.0, min_alpha))

class NashSocialWelfare(FairnessMetric):
    """
    Nash Social Welfare（纳什社会福利），使用对数形式（log-NSW）。

    定义：log-NSW = (1/n) * Σ log(v_i(alloc[i]))
        即各代理效用几何平均的对数，等价于 log(∏ v_i)^(1/n)

    返回值：float
        - 越大越好（效用分配越均衡且整体越高）
        - 若某代理效用为 0，返回 -inf（几何平均退化为 0）
        - 若某代理效用为负（chores 场景），返回 None 并提示不适用

    选择 log-NSW 而非原始乘积的原因：
        - n 个代理的效用连乘在 n 较大时会数值下溢（趋近于 0）
        - log 形式在数值上更稳定，且保持相同的排序关系（log 单调）
        - 便于跨实验比较，不受 n 和物品数量 m 的尺度影响
    """

    def __init__(self):
        super().__init__("NashSocialWelfare")

    def compute(self, instance, allocation):
        valuations = instance.valuations
        n = instance.n

        utilities = []
        for i in range(n):
            u = sum(valuations[i][item] for item in allocation[i])
            utilities.append(u)

        # 负效用场景（chores）：NSW 几何平均无意义
        if any(u < 0 for u in utilities):
            return None

        # 零效用：几何平均为 0，log-NSW 为 -inf
        if any(u == 0 for u in utilities):
            return float("-inf")

        # log-NSW = 均值形式，数值稳定
        log_nsw = sum(math.log(u) for u in utilities) / n
        return log_nsw
