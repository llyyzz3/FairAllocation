from dataclasses import dataclass
from enum import Enum
from itertools import product

from .ef2x_utils import (
    Bundle,
    PRAlgorithm,
    PRResult,
    PRStep,
    Partition,
    ValuationMatrix,
    best_k_value,
    bundle_value,
    copy_partition,
    ef2x_best_bundle_indices,
    efx_best_bundle_indices,
    find_ef2x_assignment,
    find_efx_assignment,
    is_ef2x_feasible_bundle,
    is_efx_feasible_bundle,
    is_swap_optimal,
    least_good,
    swap_optimize_with_trace,
    validate_partition,
    validate_valuations,
)


class EF2XStage(Enum):
    """
    å››äºº EF2X ç®—æ³•ä¸­çš„é˜¶æ®µåç§°ã€‚
    """

    FINISHED = "finished"
    STAGE_A = "stage_a"
    STAGE_B = "stage_b"
    STAGE_B1 = "stage_b1"
    STAGE_B2 = "stage_b2"
    STAGE_B2I = "stage_b2i"
    STAGE_B2II = "stage_b2ii"
    UNKNOWN = "unknown"


ALLOWED_TRANSITIONS = {
    EF2XStage.STAGE_A: {EF2XStage.FINISHED, EF2XStage.STAGE_A, EF2XStage.STAGE_B},
    EF2XStage.STAGE_B: {EF2XStage.FINISHED, EF2XStage.STAGE_B1},
    EF2XStage.STAGE_B1: {
        EF2XStage.FINISHED,
        EF2XStage.STAGE_A,
        EF2XStage.STAGE_B1,
        EF2XStage.STAGE_B2,
    },
    EF2XStage.STAGE_B2: {
        EF2XStage.FINISHED,
        EF2XStage.STAGE_A,
        EF2XStage.STAGE_B2I,
        EF2XStage.STAGE_B2II,
    },
    EF2XStage.STAGE_B2I: {
        EF2XStage.FINISHED,
        EF2XStage.STAGE_A,
        EF2XStage.STAGE_B1,
        EF2XStage.STAGE_B2,
        EF2XStage.STAGE_B2I,
        EF2XStage.STAGE_B2II,
    },
    EF2XStage.STAGE_B2II: {
        EF2XStage.FINISHED,
        EF2XStage.STAGE_A,
        EF2XStage.STAGE_B,
        EF2XStage.STAGE_B1,
        EF2XStage.STAGE_B2,
        EF2XStage.STAGE_B2I,
        EF2XStage.STAGE_B2II,
    },
}


@dataclass
class StageInfo:
    """
    ä¿å­˜å½“å‰ partition æ‰€å¤„é˜¶æ®µä»¥åŠç›¸å…³ agent ä¿¡æ¯ã€‚
    """

    stage: EF2XStage
    partition: Partition
    assignment: list[int] | None = None
    special_agent: int = 0
    agent_i: int | None = None
    agent_j: int | None = None
    agent_u: int | None = None
    shared_best_bundle_index: int | None = None
    stage_a_type: str | None = None
    stage_a_agent_j: int | None = None
    stage_a_shared_agents: list[int] | None = None
    stage_b1_agent_j: int | None = None
    stage_b2_transfer_goods: list[int] | None = None
    previous_x3_size: int | None = None
    current_x3_size: int | None = None
    iteration: int | None = None
    notes: str = ""

    def __post_init__(self) -> None:
        """
        æ·±æ‹·è´ partition å’Œ assignmentï¼Œé¿å…å¤–éƒ¨ä¿®æ”¹çŠ¶æ€å¯¹è±¡ã€‚
        """
        self.partition = copy_partition(self.partition)
        if self.assignment is not None:
            self.assignment = list(self.assignment)
        if self.stage_a_shared_agents is not None:
            self.stage_a_shared_agents = list(self.stage_a_shared_agents)
        if self.stage_b2_transfer_goods is not None:
            self.stage_b2_transfer_goods = list(self.stage_b2_transfer_goods)


@dataclass
class InitializationResult:
    """
    ä¿å­˜å››äºº EF2X åˆå§‹åŒ–é˜¶æ®µçš„ç»“æœã€‚
    """

    partition: Partition
    stage_info: StageInfo
    pr_iterations: int
    pr_steps: list[PRStep]
    original_partition: Partition

    def __post_init__(self) -> None:
        """
        æ·±æ‹·è´ partitionï¼Œä¿è¯ç»“æœå¯¹è±¡ä¸å…±äº«è°ƒç”¨è€…è¾“å…¥ã€‚
        """
        self.partition = copy_partition(self.partition)
        self.original_partition = copy_partition(self.original_partition)
        self.pr_steps = list(self.pr_steps)


@dataclass
class EF2XFourAgentsResult:
    """
    ä¿å­˜å®Œæ•´å››äºº EF2X ç®—æ³•çš„è¿è¡Œç»“æœã€‚
    """

    allocation: Partition
    assignment: list[int]
    final_partition: Partition
    final_stage: EF2XStage
    iterations: int
    stage_trace: list[StageInfo]
    initial_partition: Partition
    terminated: bool
    notes: str = ""

    def __post_init__(self) -> None:
        """
        æ·±æ‹·è´æ‰€æœ‰å¯å˜å­—æ®µï¼Œé¿å…è°ƒç”¨æ–¹ä¿®æ”¹ç»“æœæ—¶æ±¡æŸ“ traceã€‚
        """
        self.allocation = copy_partition(self.allocation)
        self.assignment = list(self.assignment)
        self.final_partition = copy_partition(self.final_partition)
        self.stage_trace = [StageInfo(**info.__dict__) for info in self.stage_trace]
        self.initial_partition = copy_partition(self.initial_partition)


class EF2XFourAgents:
    """
    å®ç°è®ºæ–‡ã€ŠEF2X Exists For Four Agentsã€‹ä¸­çš„å››äºº EF2X ç®—æ³•ã€‚
    """

    def __init__(
        self,
        valuations: ValuationMatrix,
        eps: float = 1e-9,
        special_agent: int = 0,
    ):
        """
        åˆå§‹åŒ–å››äºº EF2X ç®—æ³•ã€‚

        valuations å¿…é¡»æ°å¥½åŒ…å« 4 ä¸ª agentsã€‚å½“å‰å®ç°ä¸ä¼šä¿®æ”¹è¾“å…¥ä¼°å€¼çŸ©é˜µã€‚
        """
        self.valuations = self._normalize_valuations(valuations)
        validate_valuations(self.valuations)

        if len(self.valuations) != 4:
            raise ValueError("å››äºº EF2X ç®—æ³•è¦æ±‚ valuations æ°å¥½åŒ…å« 4 ä¸ª agentsã€‚")

        if special_agent < 0 or special_agent >= 4:
            raise ValueError(f"special_agent index è¶Šç•Œï¼š{special_agent}ã€‚")

        self.eps = eps
        self.special_agent = special_agent
        self.num_goods = len(self.valuations[0])
        self.name = "EF2X Four Agents"
        self._valuations_snapshot = [row[:] for row in self.valuations]
        self._good_rank = {good: good for good in range(self.num_goods)}

    def get_name(self) -> str:
        """
        è¿”å›å®éªŒè¾“å‡ºä¸­ä½¿ç”¨çš„ç¨³å®šç®—æ³•åç§°ã€‚
        """
        return self.name

    def get_metadata(self) -> dict:
        """
        è¿”å›å¯ JSON åºåˆ—åŒ–çš„ç®—æ³• metadataï¼Œæ˜ç¡®å†…éƒ¨å…¬å¹³æ€§å®šä¹‰ã€‚
        """
        return {
            "algorithm_name": self.name,
            "supported_agents": 4,
            "fairness_guarantee": "exact EF2X",
            "internal_definition": "all_goods",
            "source_paper": "EF2X Exists For Four Agents",
            "deterministic": True,
            "valuation_model": "non-negative additive valuations",
            "degeneracy_handling": {
                "enabled": True,
                "special_agent": self.special_agent,
                "method": "exact_lexicographic_perturbation",
                "scope": "internal_working_comparisons_only",
            },
        }

    def _build_initial_partition(self) -> Partition:
        """
        æ„é€ å®Œæ•´ç®—æ³•ä½¿ç”¨çš„ç¡®å®šæ€§å›› bundle åˆå§‹ partitionã€‚
        """
        partition = [set() for _ in range(4)]
        for good in range(self.num_goods):
            partition[good % 4].add(good)
        validate_partition(partition, self.num_goods, require_complete=True)
        return partition

    def _copy_stage_info(self, info: StageInfo) -> StageInfo:
        """
        æ·±æ‹·è´ StageInfoï¼Œé˜²æ­¢ trace è¢«åç»­ä¿®æ”¹ã€‚
        """
        return StageInfo(
            stage=info.stage,
            partition=copy_partition(info.partition),
            assignment=None if info.assignment is None else list(info.assignment),
            special_agent=info.special_agent,
            agent_i=info.agent_i,
            agent_j=info.agent_j,
            agent_u=info.agent_u,
            shared_best_bundle_index=info.shared_best_bundle_index,
            stage_a_type=info.stage_a_type,
            stage_a_agent_j=info.stage_a_agent_j,
            stage_a_shared_agents=None if info.stage_a_shared_agents is None else list(info.stage_a_shared_agents),
            stage_b1_agent_j=info.stage_b1_agent_j,
            stage_b2_transfer_goods=None if info.stage_b2_transfer_goods is None else list(info.stage_b2_transfer_goods),
            previous_x3_size=info.previous_x3_size,
            current_x3_size=info.current_x3_size,
            iteration=info.iteration,
            notes=info.notes,
        )

    def _state_key(self, info: StageInfo) -> tuple:
        """
        ä¸ºå¾ªç¯æ£€æµ‹ç”Ÿæˆç¡®å®šæ€§çš„çŠ¶æ€ keyã€‚
        """
        return (
            info.stage.value,
            tuple(tuple(sorted(bundle)) for bundle in info.partition),
            info.special_agent,
            info.agent_i,
            info.agent_j,
            info.agent_u,
            info.shared_best_bundle_index,
            info.stage_a_type,
            info.stage_a_agent_j,
            tuple(info.stage_a_shared_agents or []),
            info.stage_b1_agent_j,
            tuple(info.stage_b2_transfer_goods or []),
            info.previous_x3_size,
            info.current_x3_size,
        )

    def _validate_stage_info(self, info: StageInfo) -> None:
        """
        éªŒè¯ StageInfo ä¸å…¶ stage ç±»å‹ä¸€è‡´ã€‚
        """
        if info.stage == EF2XStage.FINISHED:
            if info.assignment is None:
                raise RuntimeError("FINISHED çŠ¶æ€å¿…é¡»åŒ…å« assignmentã€‚")
            self._validate_assignment(info.partition, info.assignment, require_ef2x=True)
            return
        if info.stage == EF2XStage.STAGE_A:
            valid = self.classify_stage_a(info.partition) is not None
        elif info.stage == EF2XStage.STAGE_B:
            valid = self.classify_stage_b(info.partition) is not None
        elif info.stage == EF2XStage.STAGE_B1:
            valid = self.classify_stage_b1(info.partition) is not None
        elif info.stage == EF2XStage.STAGE_B2:
            valid = self.classify_stage_b2(info.partition) is not None
        elif info.stage == EF2XStage.STAGE_B2I:
            valid = self.classify_stage_b2i(info.partition, info) is not None
        elif info.stage == EF2XStage.STAGE_B2II:
            valid = self.classify_stage_b2ii(info.partition, info) is not None
        else:
            raise RuntimeError("å®Œæ•´ EF2X ä¸»å¾ªç¯ä¸­ä¸å…è®¸å‡ºç° UNKNOWN çŠ¶æ€ã€‚")
        if not valid:
            raise RuntimeError(f"StageInfo éªŒè¯å¤±è´¥ï¼šstage={info.stage}, partition={info.partition}ã€‚")

    def _dispatch_stage(self, info: StageInfo) -> StageInfo:
        """
        æ ¹æ®å½“å‰ stage è°ƒç”¨å¯¹åº”å¤„ç†å‡½æ•°ã€‚
        """
        if info.stage == EF2XStage.FINISHED:
            return self._copy_stage_info(info)
        if info.stage == EF2XStage.STAGE_A:
            return self.handle_stage_a(info.partition)
        if info.stage == EF2XStage.STAGE_B:
            return self.handle_stage_b(info.partition)
        if info.stage == EF2XStage.STAGE_B1:
            return self.handle_stage_b1(info.partition)
        if info.stage == EF2XStage.STAGE_B2:
            return self.handle_stage_b2(info.partition)
        if info.stage == EF2XStage.STAGE_B2I:
            return self.handle_stage_b2i(info.partition)
        if info.stage == EF2XStage.STAGE_B2II:
            return self.handle_stage_b2ii(info.partition, info)
        raise RuntimeError("æ— æ³• dispatch UNKNOWN çŠ¶æ€ã€‚")

    def _build_algorithm_diagnostics(
        self,
        current_info: StageInfo,
        stage_trace: list[StageInfo],
    ) -> str:
        """
        æ„é€ å®Œæ•´ç®—æ³•å¤±è´¥æ—¶çš„ä¸­æ–‡è¯Šæ–­æ–‡æœ¬ã€‚
        """
        stage_sequence = [info.stage.value for info in stage_trace]
        potentials = [self.potential(info.partition) for info in stage_trace]
        cardinalities = [[len(bundle) for bundle in info.partition] for info in stage_trace]
        return (
            f"valuations={self.valuations}, current_partition={current_info.partition}, "
            f"current_stage={current_info.stage}, agents=(i={current_info.agent_i}, "
            f"j={current_info.agent_j}, u={current_info.agent_u}), "
            f"stage_sequence={stage_sequence}, potentials={potentials}, "
            f"cardinalities={cardinalities}, state_key={self._state_key(current_info)}, "
            f"diagnostics={self._stage_b2_diagnostics(current_info.partition)}"
        )

    def _validate_transition_progress(self, old_info: StageInfo, new_info: StageInfo) -> None:
        """
        éªŒè¯ä¸€æ¬¡ stage transformation çš„åŸºæœ¬è¿›å±•æ€§ã€‚
        """
        allowed = ALLOWED_TRANSITIONS.get(old_info.stage, set())
        if new_info.stage not in allowed:
            raise RuntimeError(
                "stage transition ä¸è¢«è®ºæ–‡æµç¨‹å…è®¸ï¼š"
                f"old={old_info.stage}, new={new_info.stage}ã€‚"
            )
        if self._state_key(old_info) == self._state_key(new_info):
            raise RuntimeError(f"stage transition æ²¡æœ‰æ”¹å˜çŠ¶æ€ï¼š{self._state_key(old_info)}ã€‚")
        if new_info.stage == EF2XStage.FINISHED:
            self._validate_stage_info(new_info)
            return
        self._validate_stage_info(new_info)

        old_potential = self.working_potential_key(old_info.partition)
        new_potential = self.working_potential_key(new_info.partition)
        if new_potential > old_potential:
            return
        if (
            old_info.stage == EF2XStage.STAGE_A
            and new_info.stage == EF2XStage.STAGE_A
            and new_potential > old_potential
        ):
            return
        if (
            new_info.current_x3_size is not None
            and new_info.previous_x3_size is not None
            and new_info.current_x3_size > new_info.previous_x3_size
        ):
            return
        if (
            old_info.stage == EF2XStage.STAGE_B
            and new_info.stage == EF2XStage.STAGE_B1
            and new_potential >= old_potential
        ):
            return
        if old_info.stage in {EF2XStage.STAGE_B1, EF2XStage.STAGE_B2} and new_info.stage != old_info.stage:
            return
        raise RuntimeError(
            "stage transition ç¼ºå°‘å¯éªŒè¯è¿›å±•ï¼š"
            f"old={old_info.stage}, new={new_info.stage}, "
            f"old_working_potential={old_potential}, new_working_potential={new_potential}, "
            f"old_original_potential={self.potential(old_info.partition)}, "
            f"new_original_potential={self.potential(new_info.partition)}ã€‚"
        )

    def _build_final_allocation(self, final_info: StageInfo) -> Partition:
        """
        æ ¹æ® final_partition å’Œ assignment æ„é€ æŒ‰ agent é¡ºåºæ’åˆ—çš„ allocationã€‚
        """
        if final_info.assignment is None:
            raise RuntimeError("æ„é€ æœ€ç»ˆ allocation æ—¶ç¼ºå°‘ assignmentã€‚")
        allocation = [
            set(final_info.partition[final_info.assignment[agent]])
            for agent in range(4)
        ]
        validate_partition(allocation, self.num_goods, require_complete=True)
        for agent in range(4):
            if not is_ef2x_feasible_bundle(self.valuations, agent, agent, allocation, self.eps):
                raise RuntimeError(f"æœ€ç»ˆ allocation ä¸­ agent {agent} æœªæ»¡è¶³ EF2Xã€‚")
        return allocation

    def validate_final_result(self, result: EF2XFourAgentsResult) -> None:
        """
        éªŒè¯å®Œæ•´ç®—æ³•è¿”å›ç»“æœçš„ç»“æ„å’Œ all-goods EF2X æ­£ç¡®æ€§ã€‚
        """
        if result.terminated is not True:
            raise RuntimeErroã¯:ÖÚ$z{-®éÜj×2‚¢–bÆVâ†vVçG2’ãÒ ¢Ğ ¢–bæ÷B6æF–FFW3 ¢&—6R'VçF–ÖTW'&÷"†b.iÊ®h›îX‹X[Kª²Te‚Ö&W7B'VæFÆ^ûÈÇ&WVW7G3×·&WVW7G7Ş8"" ¢6æF–FFW2ç6÷'B†¶W“ÖÆÖ&F—FVÓ¢‚ÖÆVâ†—FVÕ³Ò’Â—FVÕ³Ò’¢&WGW&â6æF–FFW5³Ğ ¢FVb÷&VæÖU÷Fõ÷7FvUö€¢6VÆbÀ¢'F—F–öã¢'F—F–öâÀ¢6†&VEö'VæFÆUö–æFWƒ¢–çBÀ¢’ÓâGWÆUµ'F—F–öâÂF–7E¶–çBÂ–çEÕÓ ¢"" ¢[b'F—F–öâ˜xŞYŞYŞK‹¢7FvR{¹>ièN8  ¢ikKØŞ{ÚâÃÃ"Ã2XˆnXŠ¾Zû[©NŠë®ih~KŠŞy¨BƒÅƒ"Åƒ2ÅƒN8 ¢"" ¢6VÆbå÷fÆ–FFUöf÷W%ö'VæFÆU÷'F—F–öâ‡'F—F–öâ¢–b6†&VEö'VæFÆUö–æFW‚Â÷"6†&VEö'VæFÆUö–æFW‚ãÒC ¢&—6RfÇVTW'&÷"†b'6†&VEö'VæFÆUö–æFW‚‹h®yXÎûÉ§·6†&VEö'VæFÆUö–æFW‡Ş8"" ¢&VÖ–æ–æuööÆEö–æF–6W2Ò°¢–æFW€¢f÷"–æFW‚–â&ævRƒB¢–b–æFW‚Ò6†&VEö'VæFÆUö–æFW€¢Ğ¢ƒööÆEö–æFW‚ÒÖ–â€¢&VÖ–æ–æuööÆEö–æF–6W2À¢¶W“ÖÆÖ&F–æFWƒ¢€¢6VÆbçv÷&¶–æu÷fÇVUö¶W’‡6VÆbç7V6–ÅövVçBÂ'F—F–öå¶–æFW…Ò’À¢–æFW‚À¢’À¢¢ƒ%÷ƒ5ööÆEö–æF–6W2Ò6÷'FVB€¢–æFW€¢f÷"–æFW‚–â&VÖ–æ–æuööÆEö–æF–6W0¢–b–æFW‚ÒƒööÆEö–æFW€¢ ¢æWuö÷&FW"Ò·ƒööÆEö–æFW…Ò²ƒ%÷ƒ5ööÆEö–æF–6W2²·6†&VEö'VæFÆUö–æFW…Ğ¢öÆE÷FõöæWrÒ°¢öÆEö–æFWƒ¢æWuö–æFW€¢f÷"æWuö–æFW‚ÂöÆEö–æFW‚–âVçVÖW&FR†æWuö÷&FW"¢Ğ¢&VæÖVBÒ·6WB‡'F—F–öå¶öÆEö–æFW…Ò’f÷"öÆEö–æFW‚–âæWuö÷&FW%Ğ¢fÆ–FFU÷'F—F–öâ‡&VæÖVBÂ6VÆbæçVÕövööG2Â&WV—&Uö6ö×ÆWFSÕG'VR¢&WGW&â&VæÖVBÂöÆE÷FõöæWp ¢FVb÷W6VEövööG2‡6VÆbÂ'F—F–öã¢'F—F–öâ’Óâ6WE¶–çEÓ ¢"" ¢‹ùNY¹â'F—F–öâKŠŞ[{.{¸şKÛşyJy¨BvööG2™¸nYûÈÎyJK¨îj8iúRG&ç6f÷&ÖF–öâiŠşY
nKùŞhÈvööG2Zèh.8 ¢"" ¢vööG3¢6WE¶–çEÒÒ6WB‚¢f÷"'VæFÆR–â'F—F–öã ¢vööG2çWFFR†'VæFÆR¢&WGW&âvööG0 ¢FVb÷fÆ–FFU÷7FvUö–’€¢6VÆbÀ¢'F—F–öã¢'F—F–öâÀ¢6†&VEövVçG3¢Æ—7E¶–çEÒÀ¢’ÓâæöæS ¢"" ¢š¨ÎŠøX‰ŞZx¾XÉnyIşh‰y¨B'F—F–öâkº‹k27FvR–8 ¢"" ¢6VÆbå÷fÆ–FFUöf÷W%ö'VæFÆU÷'F—F–öâ‡'F—F–öâ ¢F–væ÷7F–72Ò6VÆbå÷7FvUö–•öF–væ÷7F–72‡'F—F–öâÂ6†&VEövVçG2 ¢–bÆVâ‡6WB‡6†&VEövVçG2’’Â# ¢&—6R'VçF–ÖTW'&÷"†b%7FvR–’š¨ÎŠøZK‹J^ûÉ§6†&VEövVçG2[	K¨îKŠNKŠ®8'¶F–væ÷7F–77Ò" ¢–bç’†vVçBÓÒ6VÆbç7V6–ÅövVçBf÷"vVçB–â6†&VEövVçG2“ ¢&—6R'VçF–ÖTW'&÷"†b%7FvR–’š¨ÎŠøZK‹J^ûÉ§6†&VEövVçG2XÈ^Y
¾x›jè¢vVçN8'¶F–væ÷7F–77Ò" ¢f÷"'VæFÆUö–æFW‚–âƒÂÂ"“ ¢–bæ÷B6VÆbåö—5÷v÷&¶–æuöVg…öfV6–&ÆUö'VæFÆR€¢6VÆbç7V6–ÅövVçBÀ¢'VæFÆUö–æFW‚À¢'F—F–öâÀ¢“ ¢&—6R'VçF–ÖTW'&÷"€¢b%7FvR–’š¨ÎŠøZK‹J^ûÉ¥‡¶'VæFÆUö–æFW‚²Ò ¢b.Zû’7V6–ÅövVçBKˆŞiŠòTe‚ÖfV6–&Æ^8'¶F–væ÷7F–77Ò ¢ ¢ƒ÷fÇVRÒ6VÆbçv÷&¶–æu÷fÇVUö¶W’‡6VÆbç7V6–ÅövVçBÂ'F—F–öå³Ò¢f÷"'VæFÆUö–æFW‚–âƒÂ"“ ¢÷F†W%÷fÇVRÒ6VÆbçv÷&¶–æu÷fÇVUö¶W’€¢6VÆbç7V6–ÅövVçBÀ¢'F—F–öå¶'VæFÆUö–æFW…ÒÀ¢¢–bƒ÷fÇVRâ÷F†W%÷fÇVS ¢&—6R'VçF–ÖTW'&÷"€¢%7FvR–’š¨ÎŠøZK‹J^ûÉ¥ƒKˆŞiŠò7V6–ÅövVçBYÊX˜ŞKˆKŠ¢ ¢b&'VæFÆW2KŠŞy¨BÆV7B'VæFÆ^8'¶F–væ÷7F–77Ò ¢ ¢f÷"vVçB–â6†&VEövVçG3 ¢&W7Eö–æF–6W2ÒVg…ö&W7Eö'VæFÆUö–æF–6W2€¢6VÆbçfÇVF–öç2À¢vVçBÀ¢'F—F–öâÀ¢6VÆbæW2À¢¢–b2æ÷B–â&W7Eö–æF–6W3 ¢&—6R'VçF–ÖTW'&÷"€¢b%7FvR–’š¨ÎŠøZK‹J^ûÉ¥ƒBKˆŞiŠòvVçB¶vVçGÒy¨BTe‚Ö&W7N8" ¢b'¶F–væ÷7F–77Ò ¢ ¢FVb÷7FvUö–•öF–væ÷7F–72€¢6VÆbÀ¢'F—F–öã¢'F—F–öâÀ¢6†&VEövVçG3¢Æ—7E¶–çEÒÀ¢’Óâ7G# ¢"" ¢yIşh‰7FvR–’š¨ÎŠøZK‹J^i{ny¨NKŠŞih~Šø®ijŞKúhş8 ¢"" ¢fV6–&–Æ—G’Ò°¢°¢—5öVg…öfV6–&ÆUö'VæFÆR€¢6VÆbçfÇVF–öç2À¢vVçBÀ¢'VæFÆUö–æFW‚À¢'F—F–öâÀ¢6VÆbæW2À¢¢f÷"'VæFÆUö–æFW‚–â&ævRƒB¢Ğ¢f÷"vVçB–â&ævRƒB¢Ğ¢&W7E÷6WG2Ò°¢vVçC¢Vg…ö&W7Eö'VæFÆUö–æF–6W2€¢6VÆbçfÇVF–öç2À¢vVçBÀ¢'F—F–öâÀ¢6VÆbæW2À¢¢f÷"vVçB–â&ævRƒB¢Ğ¢&WGW&â€¢b'fÇVF–öç3×·6VÆbçfÇVF–öç7ÒÂ'F—F–öã×·'F—F–öçÒÂ ¢b$Te‚fV6–&–Æ—G“×¶fV6–&–Æ—G—ÒÂTe‚Ö&W7B6WG3×¶&W7E÷6WG7ÒÂ ¢b'6†&VEövVçG3×·6†&VEövVçG7Ş8" ¢ ¢FVb÷fÆ–FFU÷7V6–ÅövVçEöVg‚‡6VÆbÂ'F—F–öã¢'F—F–öâ’ÓâæöæS ¢"" ¢š¨ÎŠø"‹é>X{®KŠŞh˜iÈ’'VæFÆW2˜;ŞZûx›jè¢vVçBTe‚ÖfV6–&Æ^8 ¢"" ¢f÷"'VæFÆUö–æFW‚–â&ævRƒB“ ¢–bæ÷B6VÆbåö—5÷v÷&¶–æuöVg…öfV6–&ÆUö'VæFÆR€¢6VÆbç7V6–ÅövVçBÀ¢'VæFÆUö–æFW‚À¢'F—F–öâÀ¢“ ¢&—6R'VçF–ÖTW'&÷"€¢b%"‹é>X{®š¨ÎŠøZK‹J^ûÉ¦'VæFÆR¶'VæFÆUö–æFW‡Ò ¢.Zû’7V6–ÅövVçBYÊ‚v÷&¶–ærfÇVF–öâKˆ¾KˆŞiŠòTe‚ÖfV6–&Æ^8" ¢ ¢FVb÷fÆ–FFU÷7FvUö#ö–æfò‡6VÆbÂ7FvUö#ö–æfó¢7FvT–æfò’ÓâæöæS ¢"" ¢š¨ÎŠø7FvR#KúhşXÈ^Y
¾ZèÎi[NK‰NK©.KˆŞy»YÎy¨NY¹¾KŠ¢vVçBŠy.ˆ›.8 ¢"" ¢–b7FvUö#ö–æfòç7FvRÒTc%…7FvRå5DtUô# ¢&—6RfÇVTW'&÷"‚%7FvR#7vZÙzˆ¾[¨şŠhk"7FvUö#ö–æfòç7FvRK‹¢5DtUô#8""¢–b€¢7FvUö#ö–æfòævVçEö’—2æöæP¢÷"7FvUö#ö–æfòævVçEö¢—2æöæP¢÷"7FvUö#ö–æfòævVçE÷R—2æöæP¢“ ¢&—6RfÇVTW'&÷"‚%7FvR#Kúhş{Ë®[	vVçEö8vVçEö¢h‰bvVçE÷^8"" ¢&öÆW2Ò°¢6VÆbç7V6–ÅövVçBÀ¢7FvUö#ö–æfòævVçEö’À¢7FvUö#ö–æfòævVçEö¢À¢7FvUö#ö–æfòævVçE÷RÀ¢Ğ¢–bÆVâ‡&öÆW2’ÒC ¢&—6RfÇVTW'&÷"€¢%7FvR#y¨B7V6–ÅövVçN8vVçEö8vVçEö®8vVçE÷R[ø^š¾K©.KˆŞy»YÎ8" ¢b&–æfó×·7FvUö#ö–æf÷Ş8" ¢ ¢FVb÷fÆ–FFU÷7FvUö#%ö–æfò‡6VÆbÂ7FvUö#%ö–æfó¢7FvT–æfò’ÓâæöæS ¢"" ¢š¨ÎŠø7FvR#"KúhşXÈ^Y
¾ZèÎi[NK‰NK©.KˆŞy»YÎy¨NY¹¾KŠ¢vVçBŠy.ˆ›.8 ¢"" ¢–b7FvUö#%ö–æfòç7FvRæ÷B–â°¢Tc%…7FvRå5DtUô#"À¢Tc%…7FvRå5DtUô#$’À¢Tc%…7FvRå5DtUô#$”’À¢Ó ¢&—6RfÇVTW'&÷"‚%7FvR#"y»X[>X{Şi[Šhk"7FvUö#%ö–æfòç7FvRK‹¢5DtUô#"ô#$’ô#$”8""¢–b€¢7FvUö#%ö–æfòævVçEö’—2æöæP¢÷"7FvUö#%ö–æfòævVçEö¢—2æöæP¢÷"7FvUö#%ö–æfòævVçE÷R—2æöæP¢“ ¢&—6RfÇVTW'&÷"‚%7FvR#"Kúhş{Ë®[	vVçEö8vVçEö¢h‰bvVçE÷^8""¢&öÆW2Ò°¢6VÆbç7V6–ÅövVçBÀ¢7FvUö#%ö–æfòævVçEö’À¢7FvUö#%ö–æfòævVçEö¢À¢7FvUö#%ö–æfòævVçE÷RÀ¢Ğ¢–bÆVâ‡&öÆW2’ÒC ¢&—6RfÇVTW'&÷"€¢%7FvR#"y¨B7V6–ÅövVçN8vVçEö8vVçEö®8vVçE÷R[ø^š¾K©.KˆŞy»YÎ8" ¢b&–æfó×·7FvUö#%ö–æf÷Ş8" ¢ ¢FVb÷fÆ–FFU÷7FvUö#&•ö–æfò‡6VÆbÂ7FvUö#&•ö–æfó¢7FvT–æfò’ÓâæöæS ¢"" ¢š¨ÎŠø7FvR#&’KúhşXÈ^Y
¾ZèÎi[NK‰NK©.KˆŞy»YÎy¨NY¹¾KŠ¢vVçBŠy.ˆ›.8 ¢"" ¢–b7FvUö#&•ö–æfòç7FvRÒTc%…7FvRå5DtUô#$“ ¢&—6RfÇVTW'&÷"‚%7FvR#&’y»X[>X{Şi[Šhk"7FvUö#&•ö–æfòç7FvRK‹¢5DtUô#$8""¢–b€¢7FvUö#&•ö–æfòævVçEö’—2æöæP¢÷"7FvUö#&•ö–æfòævVçEö¢—2æöæP¢÷"7FvUö#&•ö–æfòævVçE÷R—2æöæP¢“ ¢&—6RfÇVTW'&÷"‚%7FvR#&’Kúhş{Ë®[	vVçEö8vVçEö¢h‰bvVçE÷^8""¢&öÆW2Ò°¢6VÆbç7V6–ÅövVçBÀ¢7FvUö#&•ö–æfòævVçEö’À¢7FvUö#&•ö–æfòævVçEö¢À¢7FvUö#&•ö–æfòævVçE÷RÀ¢Ğ¢–bÆVâ‡&öÆW2’ÒC ¢&—6RfÇVTW'&÷"€¢%7FvR#&’y¨B7V6–ÅövVçN8vVçEö8vVçEö®8vVçE÷R[ø^š¾K©.KˆŞy»YÎ8" ¢b&–æfó×·7FvUö#&•ö–æf÷Ş8" ¢ ¢FVb÷fÆ–FFU÷7FvUö#&–•ö–æfò‡6VÆbÂ7FvUö#&–•ö–æfó¢7FvT–æfò’ÓâæöæS ¢"" ¢š¨ÎŠø7FvR#&–’KúhşXÈ^Y
¾ZèÎi[NK‰NK©.KˆŞy»YÎy¨NY¹¾KŠ¢vVçBŠy.ˆ›.8 ¢"" ¢–b7FvUö#&–•ö–æfòç7FvRÒTc%…7FvRå5DtUô#$”“ ¢&—6RfÇVTW'&÷"‚%7FvR#&–’y»X[>X{Şi[Šhk"7FvUö#&–•ö–æfòç7FvRK‹¢5DtUô#$”8""¢–b€¢7FvUö#&–•ö–æfòævVçEö’—2æöæP¢÷"7FvUö#&–•ö–æfòævVçEö¢—2æöæP¢÷"7FvUö#&–•ö–æfòævVçE÷R—2æöæP¢“ ¢&—6RfÇVTW'&÷"‚%7FvR#&–’Kúhş{Ë®[	vVçEö8vVçEö¢h‰bvVçE÷^8""¢&öÆW2Ò°¢6VÆbç7V6–ÅövVçBÀ¢7FvUö#&–•ö–æfòævVçEö’À¢7FvUö#&–•ö–æfòævVçEö¢À¢7FvUö#&–•ö–æfòævVçE÷RÀ¢Ğ¢–bÆVâ‡&öÆW2’ÒC ¢&—6RfÇVTW'&÷"€¢%7FvR#&–’y¨B7V6–ÅövVçN8vVçEö8vVçEö®8vVçE÷R[ø^š¾K©.KˆŞy»YÎ8" ¢b&–æfó×·7FvUö#&–•ö–æf÷Ş8" ¢ ¢FVb÷&VÖ÷fUöµöÆV7EövööG2‡6VÆbÂ'VæFÆS¢'VæFÆRÂvVçC¢–çBÂ³¢–çB’Óâ'VæFÆS ¢"" ¢‹ùNY¹îXŠ™šBvVçByÈ¾iÚR²KŠ®iÈKØîK»~XÂvööG2Yîy¨B'VæFÆRXšşiÊÎ8 ¢"" ¢fÆ–FFU÷'F—F–öâ…·6WB†'VæFÆR•ÒÂ6VÆbæçVÕövööG2Â&WV—&Uö6ö×ÆWFSÔfÇ6R¢–b²Â ¢&—6RfÇVTW'&÷"†b&²KˆŞˆ;ŞK‹®‹IşûÉ§¶·Ş8""¢&VÖ–æ–ærÒ6WB†'VæFÆR¢÷&FW&VBÒ6÷'FVB‡&VÖ–æ–ærÂ¶W“ÖÆÖ&FvööC¢‡6VÆbçfÇVF–öç5¶vVçEÕ¶vööEÒÂvööB’¢f÷"vööB–â÷&FW&VE³¦µÓ ¢&VÖ–æ–ærç&VÖ÷fR†vööB¢&WGW&â&VÖ–æ–æp ¢FVb÷&VÖ–æ–æuövVçB‡6VÆbÂvVçEö“¢–çBÂvVçEö£¢–çB’Óâ–çC ¢"" ¢‹ùNY¹î™šB7V6–ÅövVçN8vVçEö8vVçEö¢K˜¾ZInYJşKˆXšKÙy¨BvVçN8 ¢"" ¢&VÖ–æ–ærÒ°¢vVç@¢f÷"vVçB–â&ævRƒB¢–bvVçBæ÷B–â·6VÆbç7V6–ÅövVçBÂvVçEö’ÂvVçEö§Ğ¢Ğ¢–bÆVâ‡&VÖ–æ–ær’Ò ¢&—6RfÇVTW'&÷"€¢.izk9^zîZé®YJşKˆXšKÙ’vVçNûÉ¢ ¢b'7V6–Ã×·6VÆbç7V6–ÅövVçGÒÂ“×¶vVçEö—ÒÂ£×¶vVçEö§Ş8" ¢¢&WGW&â&VÖ–æ–æu³Ğ ¢FVb÷fÆ–FFUö76–væÖVçB€¢6VÆbÀ¢'F—F–öã¢'F—F–öâÀ¢76–væÖVçC¢Æ—7E¶–çEÒÀ¢&WV—&UöVgƒ¢&ööÂÒfÇ6RÀ¢&WV—&UöVc'ƒ¢&ööÂÒG'VRÀ¢’ÓâæöæS ¢"" ¢š¨ÎŠø76–væÖVçBiŠşY¹¾KŠ¢'VæFÆW2y¨NKˆKŠ®hé.X‰~ûÈÎ[›nj8iú^Zû[©NXZÎ[›>h
~iÚK»n8 ¢"" ¢6VÆbå÷fÆ–FFUöf÷W%ö'VæFÆU÷'F—F–öâ‡'F—F–öâ ¢–bÆVâ†76–væÖVçB’ÒC ¢&—6RfÇVTW'&÷"†b&76–væÖVçB™[ş[ªn[ø^š¾K‹¢NûÈÎ[Ù>X˜ŞK‹¢¶ÆVâ†76–væÖVçB—Ş8""¢–b6÷'FVB†76–væÖVçB’ÒÆ—7B‡&ævRƒB’“ ¢&—6RfÇVTW'&÷"†b&76–væÖVçB[ø^š¾iŠò³ÂÂ"Â5Òy¨Nhé.X‰~ûÉ§¶76–væÖVçGŞ8"" ¢f÷"vVçBÂ'VæFÆUö–æFW‚–âVçVÖW&FR†76–væÖVçB“ ¢–b&WV—&UöVg‚æBæ÷B—5öVg…öfV6–&ÆUö'VæFÆR€¢6VÆbçfÇVF–öç2À¢vVçBÀ¢'VæFÆUö–æFW‚À¢'F—F–öâÀ¢6VÆbæW2À¢“ ¢&—6R'VçF–ÖTW'&÷"€¢&76–væÖVçBiÊ®kº‹k2TeûÉ¢ ¢b&vVçC×¶vVçGÒÂ'VæFÆS×¶'VæFÆUö–æFW‡ÒÂ76–væÖVçC×¶76–væÖVçGŞ8" ¢ ¢–b&WV—&UöVc'‚æBæ÷B—5öVc'…öfV6–&ÆUö'VæFÆR€¢6VÆbçfÇVF–öç2À¢vVçBÀ¢'VæFÆUö–æFW‚À¢'F—F–öâÀ¢6VÆbæW2À¢“ ¢&—6R'VçF–ÖTW'&÷"€¢&76–væÖVçBiÊ®kº‹k2Tc%ûÉ¢ ¢b&vVçC×¶vVçGÒÂ'VæFÆS×¶'VæFÆUö–æFW‡ÒÂ76–væÖVçC×¶76–væÖVçGŞ8" ¢ ¢FVb÷7FvUö%öF–væ÷7F–72‡6VÆbÂ'F—F–öã¢'F—F–öâ’Óâ7G# ¢"" ¢yIşh‰7FvR"ô#š¨ÎŠøZK‹J^i{nKÛşyJy¨NKŠŞih~Šø®ijŞKúhş8 ¢"" ¢fV6–&–Æ—G’Ò°¢°¢—5öVg…öfV6–&ÆUö'VæFÆR€¢6VÆbçfÇVF–öç2À¢vVçBÀ¢'VæFÆUö–æFW‚À¢'F—F–öâÀ¢6VÆbæW2À¢¢f÷"'VæFÆUö–æFW‚–â&ævRƒB¢Ğ¢f÷"vVçB–â&ævRƒB¢Ğ¢&W7E÷6WG2Ò°¢vVçC¢Vg…ö&W7Eö'VæFÆUö–æF–6W2€¢6VÆbçfÇVF–öç2À¢vVçBÀ¢'F—F–öâÀ¢6VÆbæW2À¢¢f÷"vVçB–â&ævRƒB¢Ğ¢&WGW&â€¢b'fÇVF–öç3×·6VÆbçfÇVF–öç7ÒÂ'F—F–öã×·'F—F–öçÒÂ ¢b$Te‚ÖfV6–&ÆR'VæFÆW3×¶fV6–&–Æ—G—ÒÂTe‚Ö&W7B'VæFÆW3×¶&W7E÷6WG7Ş8" ¢ ¢FVb÷7FvUö#%öF–væ÷7F–72‡6VÆbÂ'F—F–öã¢'F—F–öâ’Óâ7G# ¢"" ¢yIşh‰7FvR#"G&ç6fW"Xˆn{¾ZK‹J^i{nKÛşyJy¨NKŠŞih~Šø®ijŞKúhş8 ¢"" ¢Vg…öfV6–&ÆRÒ°¢°¢—5öVg…öfV6–&ÆUö'VæFÆR‡6VÆbçfÇVF–öç2ÂvVçBÂ'VæFÆUö–æFW‚Â'F—F–öâÂ6VÆbæW2¢f÷"'VæFÆUö–æFW‚–â&ævRƒB¢Ğ¢f÷"vVçB–â&ævRƒB¢Ğ¢Vc'…öfV6–&ÆRÒ°¢°¢—5öVc'…öfV6–&ÆUö'VæFÆR‡6VÆbçfÇVF–öç2ÂvVçBÂ'VæFÆUö–æFW‚Â'F—F–öâÂ6VÆbæW2¢f÷"'VæFÆUö–æFW‚–â&ævRƒB¢Ğ¢f÷"vVçB–â&ævRƒB¢Ğ¢Vg…ö&W7BÒ°¢vVçC¢Vg…ö&W7Eö'VæFÆUö–æF–6W2‡6VÆbçfÇVF–öç2ÂvVçBÂ'F—F–öâÂ6VÆbæW2¢f÷"vVçB–â&ævRƒB¢Ğ¢Vc'…ö&W7BÒ°¢vVçC¢Vc'…ö&W7Eö'VæFÆUö–æF–6W2‡6VÆbçfÇVF–öç2ÂvVçBÂ'F—F–öâÂ6VÆbæW2¢f÷"vVçB–â&ævRƒB¢Ğ¢fÇVW2Ò°¢vVçC¢°¢'VæFÆU÷fÇVR‡6VÆbçfÇVF–öç2ÂvVçBÂ'F—F–öå¶'VæFÆUö–æFW…Ò¢f÷"'VæFÆUö–æFW‚–â&ævRƒB¢Ğ¢f÷"vVçB–â&ævRƒB¢Ğ¢ÆV7EövööG2Ò°¢†vVçBÂ'VæFÆUö–æFW‚“¢ÆV7EövööB‡6VÆbçfÇVF–öç2ÂvVçBÂ'F—F–öå¶'VæFÆUö–æFW…Ò¢f÷"vVçB–â&ævRƒB¢f÷"'VæFÆUö–æFW‚–â&ævRƒB¢Ğ¢&WGW&â€¢b$Te‚ÖfV6–&ÆS×¶Vg…öfV6–&ÆWÒÂTc%‚ÖfV6–&ÆS×¶Vc'…öfV6–&ÆWÒÂ ¢b$Te‚Ö&W7C×¶Vg…ö&W7GÒÂTc%‚Ö&W7C×¶Vc'…ö&W7GÒÂ ¢b'fÇVW3×·fÇVW7ÒÂÆV7EövööG3×¶ÆV7EövööG7Ş8" ¢ ¢FVböæöå÷7V6–ÅövVçG2‡6VÆb’ÓâÆ—7E¶–çEÓ ¢"" ¢‹ùNY¹îhÈ’–æFW‚XØ~[¨şhé.X‰~y¨NKˆKŠ®™Ùîx›jè¢vVçG>8 ¢"" ¢&WGW&â°¢vVç@¢f÷"vVçB–â&ævRƒB¢–bvVçBÒ6VÆbç7V6–ÅövVç@¢Ğ