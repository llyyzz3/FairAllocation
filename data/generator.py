import csv
import re

import numpy as np

class Instance:
    """表示一个分配问题的实例"""
    def __init__(self, n, m, valuations, dist_type):
        self.n = n
        self.m = m
        self.valuations = valuations
        self.dist_type = dist_type

class DataGenerator:
    """数据生成器基类"""
    def __init__(self, name):
        self.name = name

    def generate(self, n, m):
        raise NotImplementedError("子类必须实现 generate 方法")


class SplidditSQLDataset:
    """Load complete, non-demo Divide Goods instances from a Spliddit SQL dump."""

    def __init__(self, sql_path):
        self.sql_path = str(sql_path)
        self.name = "Spliddit-Real-World"

    @staticmethod
    def _parse_rows(sql_text, table_name):
        marker = f"INSERT INTO `{table_name}` VALUES "
        insert_starts = list(re.finditer(re.escape(marker), sql_text))
        if not insert_starts:
            raise ValueError(f"SQL dump does not contain table data for {table_name!r}")

        for insert_match in insert_starts:
            start = insert_match.end()
            end = sql_text.find(";\n", start)
            if end < 0:
                end = sql_text.find(";\r\n", start)
            if end < 0:
                raise ValueError(
                    f"Could not find the end of an INSERT for {table_name!r}"
                )

            values_text = sql_text[start:end]
            for match in re.finditer(r"\(([^()]*)\)", values_text):
                yield next(
                    csv.reader(
                        [match.group(1)],
                        delimiter=",",
                        quotechar="'",
                        escapechar="\\",
                    )
                )

    def load_instances(
            self,
            require_m_at_least_n=True,
            require_strictly_more_items=False,
            require_positive_valuations=False,
            include_demo=False,
    ):
        with open(self.sql_path, "r", encoding="utf-8", errors="replace") as sql_file:
            sql_text = sql_file.read()

        # instances columns: id, type, rent, application_id, status,
        # currency, total_fare, isDemo.  The dump uses 0 for false.
        instance_demo_flags = {
            int(row[0]): row[7].strip() != "0"
            for row in self._parse_rows(sql_text, "instances")
            if len(row) >= 8
        }
        selected_instance_ids = {
            instance_id
            for instance_id, is_demo in instance_demo_flags.items()
            if include_demo or not is_demo
        }

        agents_by_instance = {}
        for row in self._parse_rows(sql_text, "agents"):
            if len(row) < 2 or row[1] == "NULL":
                continue
            instance_id = int(row[1])
            if instance_id in selected_instance_ids:
                agents_by_instance.setdefault(instance_id, []).append(int(row[0]))

        resources_by_instance = {}
        for row in self._parse_rows(sql_text, "resources"):
            if len(row) < 3 or row[2] == "NULL":
                continue
            instance_id = int(row[2])
            if instance_id in selected_instance_ids:
                resources_by_instance.setdefault(instance_id, []).append(int(row[0]))

        values_by_instance = {}
        for row in self._parse_rows(sql_text, "valuations"):
            if len(row) < 5 or "NULL" in (row[1], row[2], row[3], row[4]):
                continue
            instance_id = int(row[3])
            if instance_id in selected_instance_ids:
                values_by_instance.setdefault(instance_id, {})[
                    (int(row[1]), int(row[2]))
                ] = float(row[4])

        instances = []
        for instance_id in sorted(selected_instance_ids):
            agent_ids = sorted(set(agents_by_instance.get(instance_id, [])))
            resource_ids = sorted(set(resources_by_instance.get(instance_id, [])))
            n, m = len(agent_ids), len(resource_ids)
            if n == 0 or m == 0:
                continue
            if require_strictly_more_items and m <= n:
                continue
            if require_m_at_least_n and not require_strictly_more_items and m < n:
                continue

            values = values_by_instance.get(instance_id, {})
            if len(values) != n * m:
                continue

            try:
                matrix = np.array(
                    [[values[(agent_id, resource_id)] for resource_id in resource_ids]
                     for agent_id in agent_ids],
                    dtype=float,
                )
            except KeyError:
                continue

            if not np.all(np.isfinite(matrix)):
                continue
            if require_positive_valuations:
                if np.any(matrix <= 0):
                    continue
            elif np.any(matrix < 0):
                continue

            instance = Instance(n, m, matrix, self.name)
            instance.source_instance_id = instance_id
            instance.source_is_demo = instance_demo_flags[instance_id]
            instances.append(instance)

        return instances

class UniformGenerator(DataGenerator):
    """均匀分布生成器"""
    def __init__(self, v_min=0, v_max=1000):
        super().__init__("Uniform")
        self.v_min = v_min
        self.v_max = v_max

    def generate(self, n, m):
        matrix = np.random.uniform(self.v_min, self.v_max, size=(n, m))
        return Instance(n, m, matrix, self.name)

class NormalGenerator(DataGenerator):
    """正态分布生成器"""
    def __init__(self, mean=500, std=100):
        super().__init__("Normal")
        self.mean = mean
        self.std = std

    def generate(self, n, m):
        matrix = np.random.normal(self.mean, self.std, size=(n, m))
        matrix = np.clip(matrix, 0, None)
        return Instance(n, m, matrix, self.name)


class SparseZeroHeavyGenerator(DataGenerator):
    """Non-negative sparse valuations in which most entries are exactly zero."""

    def __init__(self, zero_probability=0.8, v_min=1, v_max=1000):
        if not 0 <= zero_probability <= 1:
            raise ValueError("zero_probability must be between 0 and 1.")
        if v_min < 0 or v_max < v_min:
            raise ValueError("Require 0 <= v_min <= v_max.")
        super().__init__(
            f"SparseZeroHeavy(p_zero={zero_probability:g},"
            f"range={v_min:g}-{v_max:g})"
        )
        self.zero_probability = float(zero_probability)
        self.v_min = float(v_min)
        self.v_max = float(v_max)

    def generate(self, n, m):
        if n <= 0 or m <= 0:
            raise ValueError("n and m must be positive.")
        matrix = np.random.uniform(self.v_min, self.v_max, size=(n, m))
        zero_mask = np.random.random(size=(n, m)) < self.zero_probability
        matrix[zero_mask] = 0.0
        return Instance(n, m, matrix, self.name)


class SharedTopGoodsGenerator(DataGenerator):
    """All agents share the same set of especially high-valued goods."""

    def __init__(
            self,
            shared_top_fraction=0.2,
            regular_min=0,
            regular_max=500,
            top_min=750,
            top_max=1000
    ):
        if not 0 < shared_top_fraction <= 1:
            raise ValueError("shared_top_fraction must be in (0, 1].")
        if not 0 <= regular_min <= regular_max < top_min <= top_max:
            raise ValueError(
                "Require 0 <= regular_min <= regular_max < top_min <= top_max."
            )
        super().__init__(
            f"SharedTopGoods(fraction={shared_top_fraction:g})"
        )
        self.shared_top_fraction = float(shared_top_fraction)
        self.regular_min = float(regular_min)
        self.regular_max = float(regular_max)
        self.top_min = float(top_min)
        self.top_max = float(top_max)

    def generate(self, n, m):
        if n <= 0 or m <= 0:
            raise ValueError("n and m must be positive.")

        top_count = max(1, min(m, int(np.ceil(m * self.shared_top_fraction))))
        shared_top_goods = sorted(
            int(good)
            for good in np.random.choice(m, size=top_count, replace=False)
        )
        matrix = np.random.uniform(
            self.regular_min, self.regular_max, size=(n, m)
        )
        matrix[:, shared_top_goods] = np.random.uniform(
            self.top_min, self.top_max, size=(n, top_count)
        )

        result = Instance(n, m, matrix, self.name)
        result.shared_top_goods = shared_top_goods
        return result


import numpy as np
from scipy.stats import rankdata



# class MallowsGenerator(DataGenerator):
#     """
#     Mallows 模型生成器
#     模拟具有相关偏好的 agents，所有人围绕一个中心排名进行波动。
#     """
#
#     def __init__(self, phi=0.5, v_max=1000):
#         """
#         :param phi: 分散参数。越接近 0，偏好越趋同；越接近 1，越随机。
#         :param v_max: 最高估值基准
#         """
#         super().__init__(f"Mallows(phi={phi})")
#         self.phi = phi
#         self.v_max = v_max
#
#     def generate(self, n, m):
#         reference_ranking = np.arange(m)
#         np.random.shuffle(reference_ranking)
#
#         valuations = np.zeros((n, m))
#
#         for i in range(n):
#             noise = np.random.exponential(scale=self.phi, size=m)
#             agent_ranking = rankdata(reference_ranking + noise)
#
#             for item_idx in range(m):
#                 rank = agent_ranking[item_idx]
#                 valuations[i][item_idx] = (m - rank + 1) * (self.v_max / m)
#
#         valuations += np.random.uniform(0, self.v_max / (2 * m), size=(n, m))
#
#         return Instance(n, m, valuations, self.name)

# 根据你项目中的实际路径修改
# from data.generator import DataGenerator


class MallowsGenerator(DataGenerator):
    """
    基于 Kendall-Tau 距离的标准 Mallows 模型。

    对每个 agent 生成一个围绕中心排名 reference_ranking 的排列：

        P(pi) ∝ phi ^ d_K(pi, reference_ranking)

    参数：
        phi = 0:
            所有 agent 的排名都与中心排名完全相同。

        0 < phi < 1:
            agent 的排名围绕中心排名波动。

        phi = 1:
            每个排名都是均匀随机排列。
    """

    def __init__(self, phi=0.5, v_max=1000):
        if not 0 <= phi <= 1:
            raise ValueError("phi must be between 0 and 1.")

        if v_max <= 0:
            raise ValueError("v_max must be positive.")

        super().__init__(f"Mallows(phi={phi})")

        self.phi = float(phi)
        self.v_max = float(v_max)

    def _sample_inversion_count(self, max_inversions):
        """
        从截断几何分布中采样逆序数量 v：

            v ∈ {0, 1, ..., max_inversions}
            P(v) ∝ phi^v

        v 越大，新加入的 item 越可能被插入到前面，
        从而相对于中心排名产生更多逆序。
        """
        if max_inversions == 0:
            return 0

        # phi = 0 时，只可能选择 0 个逆序
        if self.phi == 0:
            return 0

        # phi = 1 时，每个插入位置等概率
        if self.phi == 1:
            return np.random.randint(0, max_inversions + 1)

        inversion_counts = np.arange(max_inversions + 1)

        probabilities = self.phi ** inversion_counts
        probabilities = probabilities / probabilities.sum()

        return int(
            np.random.choice(
                inversion_counts,
                p=probabilities
            )
        )

    def _sample_mallows_ranking(self, reference_ranking):
        """
        使用 Repeated Insertion Model 从 Mallows 分布中采样排列。

        reference_ranking 按“最喜欢到最不喜欢”的顺序存储 item。

        例如：
            reference_ranking = [2, 0, 3, 1]

        表示中心排名：
            item 2 > item 0 > item 3 > item 1
        """
        sampled_ranking = []

        for item in reference_ranking:
            current_length = len(sampled_ranking)

            inversion_count = self._sample_inversion_count(
                max_inversions=current_length
            )

            # inversion_count = 0:
            # 新 item 放在末尾，保持中心排名顺序
            #
            # inversion_count = current_length:
            # 新 item 放在最前面
            insertion_position = (
                current_length - inversion_count
            )

            sampled_ranking.insert(
                insertion_position,
                int(item)
            )

        return sampled_ranking

    def _ranking_to_valuations(self, ranking, m):
        """
        将 ordinal ranking 转换为 additive cardinal valuations。

        ranking[0] 是最喜欢的 good，
        ranking[-1] 是最不喜欢的 good。

        基础价值按排名线性下降，并加入不会改变排名的小扰动。
        """
        valuations = np.zeros(m, dtype=float)

        value_gap = self.v_max / m

        for position, item in enumerate(ranking):
            # position = 0 时价值最高
            # position = m-1 时仍保持正价值
            base_value = (
                m - position
            ) * value_gap

            # 扰动最大为相邻价值间距的 20%，
            # 因此不会改变 ranking
            jitter = np.random.uniform(
                0,
                0.2 * value_gap
            )

            valuations[item] = base_value + jitter

        return valuations

    def generate(self, n, m):
        if n <= 0:
            raise ValueError("n must be positive.")

        if m <= 0:
            raise ValueError("m must be positive.")

        # 中心排名：按最喜欢到最不喜欢排列
        reference_ranking = np.random.permutation(m)

        valuations = np.zeros((n, m), dtype=float)

        for agent in range(n):
            agent_ranking = self._sample_mallows_ranking(
                reference_ranking
            )

            valuations[agent] = self._ranking_to_valuations(
                ranking=agent_ranking,
                m=m
            )

        return Instance(
            n=n,
            m=m,
            valuations=valuations,
            dist_type=self.name
        )

#所有人偏好一样
class IdenticalGenerator(DataGenerator):
    """
    Identical valuations generator.
    所有 agents 对 goods 的估值完全相同。
    """

    def __init__(self, v_min=0, v_max=1000):
        super().__init__("Identical")
        self.v_min = v_min
        self.v_max = v_max

    def generate(self, n, m):
        base_values = np.random.uniform(self.v_min, self.v_max, size=m)
        matrix = np.tile(base_values, (n, 1))
        return Instance(n, m, matrix, self.name)

#所有 agents 对 goods 的排序一样
class OrderedGenerator(DataGenerator):
    """
    Ordered valuations generator.
    所有 agents 对 goods 的偏好顺序相同，但具体数值不同。
    """

    def __init__(self, v_min=0, v_max=1000):
        super().__init__("Ordered")
        self.v_min = v_min
        self.v_max = v_max

    def generate(self, n, m):
        # 生成一个共同的 goods 排名
        common_order = np.arange(m)
        np.random.shuffle(common_order)

        matrix = np.zeros((n, m))

        for i in range(n):
            # 为每个 agent 生成一组递增值
            values = np.sort(np.random.uniform(self.v_min, self.v_max, size=m))

            # 最大值给 common_order 里排第一的 good
            # 最小值给排最后的 good
            for rank, item in enumerate(common_order):
                matrix[i][item] = values[m - rank - 1]

        return Instance(n, m, matrix, self.name)

class FixedGenerator(DataGenerator):
    """
    使用手动给定的估值矩阵生成固定实例。
    """

    def __init__(self, valuations, name="Fixed"):
        super().__init__(name)

        self.valuations = np.array(
            valuations,
            dtype=float
        )

        if self.valuations.ndim != 2:
            raise ValueError(
                "valuations must be a 2D matrix"
            )

    def generate(self, n, m):
        fixed_n, fixed_m = self.valuations.shape

        if n != fixed_n or m != fixed_m:
            raise ValueError(
                f"{self.name} requires n={fixed_n}, m={fixed_m}, "
                f"but got n={n}, m={m}"
            )

        return Instance(
            n=fixed_n,
            m=fixed_m,
            valuations=self.valuations.copy(),
            dist_type=self.name
        )
