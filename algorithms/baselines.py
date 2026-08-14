import numpy as np


class RoundRobin:
    def __init__(self):
        self.name = "Round-Robin"

    def run(self, instance):
        """
        执行轮转法分配
        输入: Instance 对象 (包含 n, m 和 valuations 矩阵)
        输出: 分配方案 (字典，key 是代理索引，value 是分配到的物品列表)
        """
        n = instance.n
        m = instance.m
        valuations = instance.valuations

        # 初始化每个代理的分配包 (bundle)
        allocation = {i: [] for i in range(n)}

        # 记录物品是否已被分配
        remaining_items = list(range(m))

        # 轮转分配过程
        # 只要还有剩余物品，就继续轮转 [cite: 622]
        for round_idx in range(m):
            # 确定当前轮到哪个代理 (i = l mod n)
            current_agent = round_idx % n

            if not remaining_items:
                break

            # 贪心策略：该代理从剩余物品中选择估值最高的物品 [cite: 620, 2492]
            # 获取剩余物品的估值索引
            best_item = -1
            best_val = -1

            for item_idx in remaining_items:
                if valuations[current_agent][item_idx] > best_val:
                    best_val = valuations[current_agent][item_idx]
                    best_item = item_idx

            # 将物品分配给该代理并从剩余列表中移除
            allocation[current_agent].append(best_item)
            remaining_items.remove(best_item)

        return allocation


import numpy as np
import networkx as nx


# class EnvyCycleElimination:
#     def __init__(self):
#         self.name = "Envy-Cycle Elimination"
#
#     def _build_envy_graph(self, n, valuations, allocation):
#         G = nx.DiGraph()
#         G.add_nodes_from(range(n))
#         for i in range(n):
#             val_i_own = sum(valuations[i][item] for item in allocation[i])
#             for j in range(n):
#                 if i == j:
#                     continue
#                 val_i_others = sum(valuations[i][item] for item in allocation[j])
#                 if val_i_others > val_i_own:
#                     G.add_edge(i, j)
#         return G
#
#     def _resolve_cycles(self, n, allocation, G):
#         """
#         消除图中一个环，返回 True 表示发生了交换。
#         注意：调用方负责在返回 True 后重建图再调用。
#         """
#         try:
#             cycle = nx.find_cycle(G, orientation="original")
#             cycle_nodes = [edge[0] for edge in cycle]
#
#             # ✅ 修复问题2：显式拷贝，避免引用陷阱
#             first_bundle = list(allocation[first_node := cycle_nodes[0]])
#
#             for k in range(len(cycle_nodes) - 1):
#                 allocation[cycle_nodes[k]] = list(allocation[cycle_nodes[k + 1]])
#
#             allocation[cycle_nodes[-1]] = first_bundle
#             return True
#         except nx.NetworkXNoCycle:
#             return False
#
#     def run_from_partial(self, instance, allocation, remaining_items):
#         """
#         从已有 partial allocation 开始运行 Envy Cycle Elimination。
#
#         参数:
#             instance: 包含 n, m, valuations 的实例
#             allocation: 当前已有的 partial allocation，格式 {agent: [items]}
#             remaining_items: 尚未分配的物品列表
#
#         返回:
#             完整 allocation
#         """
#         n = instance.n
#         valuations = instance.valuations
#
#         # 复制一份，避免修改外部传入的 allocation
#         allocation = {i: list(allocation[i]) for i in range(n)}
#         remaining_items = list(remaining_items)
#
#         for item_idx in remaining_items:
#             # 1. 每次分配新物品前，先消除所有 envy cycles
#             G = self._build_envy_graph(n, valuations, allocation)
#
#             while self._resolve_cycles(n, allocation, G):
#                 G = self._build_envy_graph(n, valuations, allocation)
#
#             # 2. 找 envy graph 中 indegree = 0 的 source agent
#             source_nodes = [node for node, in_deg in G.in_degree() if in_deg == 0]
#
#             # 消完环后理论上一定有 source；这里用 min 保证结果可复现
#             target_agent = min(source_nodes)
#
#             # 3. 把当前 item 分给 source agent
#             allocation[target_agent].append(item_idx)
#
#         # 最后再消一次 cycle
#         G = self._build_envy_graph(n, valuations, allocation)
#         while self._resolve_cycles(n, allocation, G):
#             G = self._build_envy_graph(n, valuations, allocation)
#
#         return allocation
#
#
#     def run(self, instance):
#         n = instance.n
#         m = instance.m
#
#         allocation = {i: [] for i in range(n)}
#         remaining_items = list(range(m))
#
#         return self.run_from_partial(instance, allocation, remaining_items)
class EnvyCycleElimination:
    def __init__(self):
        self.name = "Envy-Cycle Elimination"

    def _build_envy_graph(self, n, valuations, allocation):
        G = nx.DiGraph()
        G.add_nodes_from(range(n))

        for i in range(n):
            val_i_own = sum(valuations[i][item] for item in allocation[i])

            for j in range(n):
                if i == j:
                    continue

                val_i_others = sum(valuations[i][item] for item in allocation[j])

                # 如果 i 更喜欢 j 的 bundle，就加边 i -> j
                if val_i_others > val_i_own:
                    G.add_edge(i, j)

        return G

    def _resolve_cycles(self, n, allocation, G):
        """
        消除 envy graph 中的一个 cycle。
        如果发生交换，返回 True。
        如果没有 cycle，返回 False。
        """
        try:
            cycle = nx.find_cycle(G, orientation="original")

            # cycle 里的边是 (u, v, direction)
            # u envies v
            cycle_nodes = [edge[0] for edge in cycle]

            old_bundles = {
                agent: list(allocation[agent])
                for agent in cycle_nodes
            }

            # 沿着 envy edge 方向拿下一个人的 bundle
            for idx in range(len(cycle_nodes) - 1):
                allocation[cycle_nodes[idx]] = old_bundles[cycle_nodes[idx + 1]]

            allocation[cycle_nodes[-1]] = old_bundles[cycle_nodes[0]]

            return True

        except nx.NetworkXNoCycle:
            return False

    def _eliminate_all_cycles(self, n, valuations, allocation):
        """
        重复消除所有 envy cycles，直到 envy graph 无环。
        """
        G = self._build_envy_graph(n, valuations, allocation)

        while self._resolve_cycles(n, allocation, G):
            G = self._build_envy_graph(n, valuations, allocation)

        return allocation

    def run_from_partial(self, instance, allocation, remaining_items):
        """
        从已有 partial allocation 开始运行 ECE。

        用于:
        - 0.618-EFX 后半部分
        - k-RoundRobin with ECE 后半部分
        """
        n = instance.n
        valuations = instance.valuations

        # 复制，避免修改外部 allocation
        allocation = {i: list(allocation[i]) for i in range(n)}
        remaining_items = list(remaining_items)

        while remaining_items:
            # 1. 先消除所有 cycles
            allocation = self._eliminate_all_cycles(n, valuations, allocation)

            # 2. 构建当前 envy graph
            G = self._build_envy_graph(n, valuations, allocation)

            # 3. 找 source agent，也就是 indegree = 0 的 agent
            source_nodes = [node for node, in_deg in G.in_degree() if in_deg == 0]

            if not source_nodes:
                raise RuntimeError("No source node found after cycle elimination.")

            # 固定选最小编号 source，保证实验可复现
            target_agent = min(source_nodes)

            # 4. source agent 从 remaining_items 中选自己最喜欢的 good
            best_item = max(
                remaining_items,
                key=lambda g: valuations[target_agent][g]
            )

            allocation[target_agent].append(best_item)
            remaining_items.remove(best_item)

        # 5. 最后再消一次 cycle
        allocation = self._eliminate_all_cycles(n, valuations, allocation)

        return allocation

    def run(self, instance):
        n = instance.n
        m = instance.m

        allocation = {i: [] for i in range(n)}
        remaining_items = list(range(m))

        return self.run_from_partial(instance, allocation, remaining_items)

import numpy as np
import networkx as nx


# class EFX_0618_Approx:
#     """
#     0.618-EFX (φ-EFX) 算法实现
#     基于 Amanatidis et al. (2020) "Maximum Nash Welfare Implies Envy-Freeness Up to a Small Item"
#     黄金比例 φ = (√5 - 1) / 2 ≈ 0.618
#
#     核心性质：对所有代理 i, j，存在物品 g ∈ allocation[j] 使得：
#         v_i(allocation[i]) >= φ * v_i(allocation[j] \ {g})
#     """
#
#     def __init__(self):
#         self.name = "0.618-EFX (Golden Ratio EFX)"
#         self.phi = (5 ** 0.5 - 1) / 2  # ≈ 0.618
#
#     # ------------------------------------------------------------------
#     # 辅助：嫉妒图相关（与 ECE 保持一致）
#     # ------------------------------------------------------------------
#
#     def _build_envy_graph(self, n, valuations, allocation):
#         """
#         构建有向嫉妒图。
#         如果代理 i 羡慕代理 j（严格大于），则添加边 i -> j。
#         """
#         G = nx.DiGraph()
#         G.add_nodes_from(range(n))
#
#         for i in range(n):
#             val_i_own = sum(valuations[i][item] for item in allocation[i])
#             for j in range(n):
#                 if i == j:
#                     continue
#                 val_i_others = sum(valuations[i][item] for item in allocation[j])
#                 if val_i_others > val_i_own:
#                     G.add_edge(i, j)
#         return G
#
#     def _resolve_cycles(self, n, allocation, G):
#         """
#         寻找并消除嫉妒图中的一个环（与 ECE 相同的轮转交换逻辑）。
#         返回 True 表示发生了交换，False 表示图中无环。
#         """
#         while True:
#             try:
#                 cycle = nx.find_cycle(G, orientation="original")
#                 cycle_nodes = [edge[0] for edge in cycle]
#
#                 first_node = cycle_nodes[0]
#                 first_bundle = list(allocation[first_node])
#
#                 for k in range(len(cycle_nodes) - 1):
#                     curr_node = cycle_nodes[k]
#                     next_node = cycle_nodes[k + 1]
#                     allocation[curr_node] = list(allocation[next_node])
#
#                 allocation[cycle_nodes[-1]] = first_bundle
#
#                 return True
#             except nx.NetworkXNoCycle:
#                 return False
#
#     def _eliminate_all_cycles(self, n, valuations, allocation):
#         """
#         反复重建嫉妒图并消环，直到图中无环为止。
#         """
#         while True:
#             G = self._build_envy_graph(n, valuations, allocation)
#             changed = self._resolve_cycles(n, allocation, G)
#             if not changed:
#                 return G  # 返回无环的嫉妒图
#
#     # ------------------------------------------------------------------
#     # 辅助：φ-EFX 检查与修复
#     # ------------------------------------------------------------------
#
#     def _is_phi_efx_satisfied(self, i, j, valuations, allocation):
#         """
#         检查代理 i 对代理 j 的包是否满足 φ-EFX：
#         ∀g ∈ allocation[j]: v_i(allocation[i]) >= φ * v_i(allocation[j] \ {g})
#         即：存在移除某物品后，i 不再以比例 φ 羡慕 j。
#         等价于：v_i(allocation[j]) - min_g(v_i(g)) <= v_i(allocation[i]) / φ
#
#         实际判断：对 j 包中每个物品 g，移除后 i 是否还以 φ 倍羡慕 j。
#         任意 g 使得 v_i(alloc[j]\{g}) <= v_i(alloc[i]) / φ
#         """
#         if not allocation[j]:
#             return True
#
#         val_i_own = sum(valuations[i][item] for item in allocation[i])
#         val_i_j = sum(valuations[i][item] for item in allocation[j])
#
#         # 若 i 本来就不羡慕 j，直接满足
#         if val_i_j <= val_i_own:
#             return True
#
#         # 找 j 包中对 i 价值最小的物品
#         min_val_item = min(valuations[i][item] for item in allocation[j])
#         val_after_removal = val_i_j - min_val_item
#
#         # φ-EFX：移除最小物品后，i 对 j 包估值 <= i 对自己包估值 / φ
#         return val_after_removal <= val_i_own / self.phi
#
#     def _find_least_valuable_item(self, agent, bundle, valuations):
#         """返回 bundle 中对 agent 价值最小的物品。"""
#         if not bundle:
#             return None
#         return min(bundle, key=lambda item: valuations[agent][item])
#
#     def _find_most_valuable_item(self, agent, bundle, valuations):
#         """返回 bundle 中对 agent 价值最大的物品。"""
#         if not bundle:
#             return None
#         return max(bundle, key=lambda item: valuations[agent][item])
#
#     # ------------------------------------------------------------------
#     # 核心：φ-EFX 维护步骤
#     # ------------------------------------------------------------------
#
#     def _enforce_phi_efx(self, n, valuations, allocation, unallocated):
#         """
#         在当前分配中检查 φ-EFX 违规，并通过将物品移回"未分配池"来修复。
#
#         策略：找到一对 (i, j) 违反 φ-EFX，
#         将 j 包中对 i 价值最大的物品移回未分配池（保守策略保留 φ-EFX）。
#
#         返回是否进行了修复。
#         """
#         for i in range(n):
#             for j in range(n):
#                 if i == j:
#                     continue
#                 if not self._is_phi_efx_satisfied(i, j, valuations, allocation):
#                     # 找到违规对，从 j 的包中移除对 i 价值最大的物品
#                     item_to_remove = self._find_most_valuable_item(
#                         i, allocation[j], valuations
#                     )
#                     if item_to_remove is not None:
#                         allocation[j].remove(item_to_remove)
#                         unallocated.append(item_to_remove)
#                         return True
#         return False
#
#     # ------------------------------------------------------------------
#     # 主算法
#     # ------------------------------------------------------------------
#
#     def run(self, instance):
#         """
#         执行 φ(0.618)-EFX 算法。
#
#         算法流程（每轮迭代）：
#         1. 消除嫉妒图中的所有环（轮转交换保持当前 φ-EFX 性质）。
#         2. 在无环嫉妒图中找入度为 0 的"源节点"（没人羡慕它的代理）。
#         3. 将未分配物品中对该源节点价值最高的物品分配给它。
#         4. 检查并修复可能新产生的 φ-EFX 违规（将破坏性物品放回待分配池）。
#         5. 重复直到所有物品分配完毕。
#         """
#         n = instance.n
#         m = instance.m
#         valuations = instance.valuations
#
#         allocation = {i: [] for i in range(n)}
#         unallocated = list(range(m))  # 待分配物品池
#
#         max_iter = m * n * 10  # 防止极端情况下的无限循环
#         iteration = 0
#
#         while unallocated and iteration < max_iter:
#             iteration += 1
#
#             # Step 1：消环，得到无环嫉妒图
#             G = self._eliminate_all_cycles(n, valuations, allocation)
#
#             # Step 2：找源节点（入度为 0 的代理，即没有人羡慕他）
#             source_nodes = [
#                 node for node, in_deg in G.in_degree() if in_deg == 0
#             ]
#             target_agent = source_nodes[0] if source_nodes else 0
#
#             # Step 3：将未分配物品中对 target_agent 价值最高的物品分配给他
#             best_item = max(
#                 unallocated,
#                 key=lambda item: valuations[target_agent][item]
#             )
#             unallocated.remove(best_item)
#             allocation[target_agent].append(best_item)
#
#             # Step 4：修复 φ-EFX 违规（若新分配破坏了某对的 φ-EFX）
#             while self._enforce_phi_efx(n, valuations, allocation, unallocated):
#                 # 修复后重新消环，保持图的性质
#                 G = self._eliminate_all_cycles(n, valuations, allocation)
#
#         # 最终消环
#         self._eliminate_all_cycles(n, valuations, allocation)
#
#         return allocation
import itertools
import math
import numpy as np


# class EFX_0618_Approx:
#     """
#     0.618-EFX (Golden Ratio EFX)
#
#     Based on Amanatidis et al. (2020):
#     "Maximum Nash Welfare Implies Envy-Freeness up to a Small Item"
#
#     Main idea:
#         Compute a Maximum Nash Welfare (MNW) allocation.
#         The theorem states that an MNW allocation satisfies phi-EFX,
#         where phi = (sqrt(5) - 1) / 2 ~= 0.618.
#
#     Note:
#         Exact MNW is NP-hard in general, so this brute-force implementation
#         is only suitable for small experimental instances (n^m <= 10^7).
#         For n=5 agents, this means m <= 10 items.
#     """
#
#     def __init__(self):
#         self.name = "0.618-EFX via Exact MNW"
#         self.phi = (5 ** 0.5 - 1) / 2
#
#     # ---------------------------------------------------------
#     # Utility functions
#     # ---------------------------------------------------------
#
#     def _val(self, agent, items, valuations):
#         return sum(valuations[agent][g] for g in items)
#
#     def _nash_log_value(self, n, allocation, valuations):
#         """
#         Nash welfare = product of utilities.
#         We use log(product) = sum(log(utility)) for numerical stability.
#         Returns -inf if any agent gets 0 utility (Nash product is 0).
#         """
#         utilities = [self._val(i, allocation[i], valuations) for i in range(n)]
#         if any(u <= 0 for u in utilities):
#             return float("-inf")
#         return sum(math.log(u) for u in utilities)
#
#     # ---------------------------------------------------------
#     # Exact MNW search
#     # ---------------------------------------------------------
#
#     def _exact_mnw_allocation(self, n, m, valuations):
#         """
#         Enumerate all n^m possible allocations and return the one
#         that maximizes the Nash Social Welfare (product of utilities).
#
#         Fix 1: removed the broken tie-breaker on zero-utility allocations.
#                When Nash product is 0 (-inf), we simply keep the first one
#                found; it does not affect the MNW result since any allocation
#                with positive Nash product will replace it.
#
#         Fix 2: added a scale guard to fail fast on infeasible instance sizes.
#         """
#         # Scale guard: brute-force is only practical for small instances
#         if n ** m > 10_000_000:
#             raise ValueError(
#                 f"Instance too large for brute-force MNW: n^m = {n}^{m} = {n**m:,}. "
#                 f"With n={n} agents, keep m <= {int(math.log(10_000_000) / math.log(n))} items."
#             )
#
#         best_allocation = None
#         best_score = float("-inf")
#
#         for assignment in itertools.product(range(n), repeat=m):
#             allocation = {i: [] for i in range(n)}
#             for g, agent in enumerate(assignment):
#                 allocation[agent].append(g)
#
#             score = self._nash_log_value(n, allocation, valuations)
#
#             # Fix 1: simple greater-than comparison, no tie-breaker needed.
#             # MNW is well-defined as the allocation maximising Nash product;
#             # among ties we just keep the first one encountered.
#             if score > best_score:
#                 best_score = score
#                 best_allocation = allocation
#
#         return best_allocation
#
#     # ---------------------------------------------------------
#     # Main entry
#     # ---------------------------------------------------------
#
#     def run(self, instance):
#         """
#         Input:
#             instance.n          number of agents
#             instance.m          number of items
#             instance.valuations n x m valuation matrix
#
#         Output:
#             allocation: dict
#                 key   = agent index  (0 .. n-1)
#                 value = list of item indices
#         """
#         n = instance.n
#         m = instance.m
#         valuations = instance.valuations
#
#         return self._exact_mnw_allocation(n, m, valuations)
import networkx as nx


class EFX_0618_Approx:
    """
    0.618-EFX via Draft-and-Eliminate

    Based on:
    Amanatidis, Markakis, Ntokos,
    "Multiple Birds with One Stone: Beating 1/2 for EFX and GMMS via Envy Cycle Elimination"

    Implements:
      Algorithm 4: Preprocessing
      Algorithm 3: Draft-and-Eliminate

    Guarantee:
      (phi - 1)-EFX = 0.618-EFX
    """

    def __init__(self):
        self.big_phi = (1 + 5 ** 0.5) / 2      # 1.618
        self.alpha = self.big_phi - 1          # 0.618
        self.name = "0.618-EFX (Draft-and-Eliminate)"

    def _val(self, i, items, valuations):
        return sum(valuations[i][g] for g in items)

    # ---------------------------------------------------------
    # Envy graph and cycle elimination
    # ---------------------------------------------------------

    def _build_envy_graph(self, n, allocation, valuations):
        G = nx.DiGraph()
        G.add_nodes_from(range(n))

        for i in range(n):
            own = self._val(i, allocation[i], valuations)
            for j in range(n):
                if i == j:
                    continue
                other = self._val(i, allocation[j], valuations)

                # Paper uses strict envy: v_i(P_i) < v_i(P_j)
                if own < other:
                    G.add_edge(i, j)

        return G

    def _cycle_resolution(self, allocation, cycle_nodes):
        first_bundle = list(allocation[cycle_nodes[0]])

        for k in range(len(cycle_nodes) - 1):
            allocation[cycle_nodes[k]] = list(allocation[cycle_nodes[k + 1]])

        allocation[cycle_nodes[-1]] = first_bundle

    def _eliminate_cycles_until_source_exists(self, n, allocation, valuations):
        while True:
            G = self._build_envy_graph(n, allocation, valuations)
            sources = [v for v, d in G.in_degree() if d == 0]

            if sources:
                return G

            cycle = nx.find_cycle(G, orientation="original")
            cycle_nodes = [edge[0] for edge in cycle]
            self._cycle_resolution(allocation, cycle_nodes)

    def _envy_cycle_elimination(self, n, allocation, pool, valuations):
        # Algorithm 1: process goods in lexicographic order
        for g in list(pool):
            G = self._eliminate_cycles_until_source_exists(
                n, allocation, valuations
            )

            sources = sorted(v for v, d in G.in_degree() if d == 0)
            i = sources[0]

            allocation[i].append(g)
            pool.remove(g)

        return allocation

    # ---------------------------------------------------------
    # Round Robin
    # ---------------------------------------------------------

    def _round_robin(self, allocation, pool, ordering, steps, valuations):
        k = 0

        while pool and steps > 0:
            agent = ordering[k % len(ordering)]

            best_g = max(pool, key=lambda g: valuations[agent][g])
            allocation[agent].append(best_g)
            pool.remove(best_g)

            k += 1
            steps -= 1

        return allocation, pool

    # ---------------------------------------------------------
    # Algorithm 4: Preprocessing
    # ---------------------------------------------------------

    def _preprocessing(self, n, m, valuations):
        """
        Implements Algorithm 4: Preprocessing(N, M)

        Returns:
            ordering ell
            n_prime = |L|
        """
        L = []
        active = set(range(n))
        remaining_goods = set(range(m))

        h = {}
        timestamp = {}

        while active:
            i = min(active)

            # h_i = i's favourite remaining good
            hi = max(remaining_goods, key=lambda g: valuations[i][g])
            h[i] = hi
            timestamp[i] = m - len(remaining_goods) + 1

            # R = (N \ (A union L)) union {i}
            R = (set(range(n)) - (active | set(L))) | {i}

            # j = argmax_t in R v_i(h_t)
            j = max(R, key=lambda t: valuations[i][h[t]])

            if self.big_phi * valuations[i][h[i]] < valuations[i][h[j]]:
                # h_i = h_j
                h[i] = h[j]

                L.append(i)

                active.remove(i)
                active.add(j)
            else:
                active.remove(i)
                remaining_goods.remove(h[i])

        # ell = L first, then N \ L by increasing timestamp
        non_L = [i for i in range(n) if i not in L]
        non_L.sort(key=lambda x: timestamp[x])

        ell = L + non_L
        n_prime = len(L)

        return ell, n_prime

    # ---------------------------------------------------------
    # Main Algorithm 3: Draft-and-Eliminate
    # ---------------------------------------------------------

    def run(self, instance):
        n = instance.n
        m = instance.m
        valuations = instance.valuations

        # Algorithm 3, line 1
        ell, n_prime = self._preprocessing(n, m, valuations)

        allocation = {i: [] for i in range(n)}
        pool = list(range(m))

        # Algorithm 3, line 3:
        # first round-robin for n steps
        allocation, pool = self._round_robin(
            allocation, pool, ell, n, valuations
        )

        # Algorithm 3, lines 4-5:
        # reverse order, only n - n_prime steps
        ell_reverse = list(reversed(ell))
        allocation, pool = self._round_robin(
            allocation, pool, ell_reverse, n - n_prime, valuations
        )

        # Algorithm 3, line 6:
        # envy-cycle-elimination on remaining goods
        allocation = self._envy_cycle_elimination(
            n, allocation, pool, valuations
        )

        return allocation