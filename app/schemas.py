"""请求/响应模型与输入校验。"""

from __future__ import annotations

import math

from pydantic import BaseModel, Field, field_validator, model_validator

MAX_TOTAL_TOKENS = 512
MAX_CHUNK_SIZE = 32
MIN_CHUNK_SIZE = 1
MAX_S = 1000.0
HEAD_DIM = 16

# /repair-value 专用约束
REPAIR_MAX_TOTAL_TOKENS = 32
REPAIR_MAX_S = 10.0
REPAIR_BOX = 8.0  # 四组张量各分量的取值范围 [-8, 8]
REPAIR_OUTPUT_TOL = 1e-8  # 输出误差容差
REPAIR_WIDTH_TOL = 1e-3  # 每维可行值范围宽度上限


class VerifyRequest(BaseModel):
    """/verify 请求体。"""

    seed: int = Field(..., ge=0, description="随机种子（非负整数）")
    doc_lengths: list[int] = Field(
        ..., min_length=1, description="各文档长度（正整数），总长不超过 512"
    )
    chunk_size: int = Field(..., ge=MIN_CHUNK_SIZE, le=MAX_CHUNK_SIZE)
    s: float = Field(..., gt=0.0, le=MAX_S, description="注意力缩放系数")

    @field_validator("doc_lengths")
    @classmethod
    def _check_doc_lengths(cls, value: list[int]) -> list[int]:
        if any(not isinstance(x, int) or x <= 0 for x in value):
            raise ValueError("doc_lengths 必须为正整数列表")
        total = sum(value)
        if total == 0 or total > MAX_TOTAL_TOKENS:
            raise ValueError(
                f"文档总长度必须在 1..{MAX_TOTAL_TOKENS} 之间，当前为 {total}"
            )
        return value

    @field_validator("s")
    @classmethod
    def _check_s_finite(cls, value: float) -> float:
        if value != value or value in (float("inf"), float("-inf")):
            raise ValueError("s 必须为有限正数")
        return value


class ErrorMetrics(BaseModel):
    output_max_abs_err: float
    grad_q_max_abs_err: float
    grad_k_max_abs_err: float
    grad_v_max_abs_err: float


class VerifyResponse(BaseModel):
    passed: bool
    tolerance: float
    errors: ErrorMetrics
    scale_used: float = Field(..., description="实际进入 softmax 的系数 s/4")
    total_tokens: int
    num_docs: int
    chunk_size: int
    s: float
    seed: int
    finite_ok: bool = Field(..., description="所有输出与梯度是否均为有限值")
    failure_reasons: list[str]
    output: list[list[float]] | None = None
    grad_q: list[list[float]] | None = None
    grad_k: list[list[float]] | None = None
    grad_v: list[list[float]] | None = None


class ScenarioSummary(BaseModel):
    name: str
    description: str
    request: VerifyRequest


class RepairValueRequest(BaseModel):
    """/repair-value 请求体：同批次的 Q/K、现存 V 与归档输出。"""

    doc_lengths: list[int] = Field(
        ..., min_length=1, description="各文档长度（正整数），总长不超过 32"
    )
    chunk_size: int = Field(..., ge=MIN_CHUNK_SIZE, le=MAX_CHUNK_SIZE)
    s: float = Field(..., gt=0.0, le=REPAIR_MAX_S, description="注意力缩放系数")
    q: list[list[float]] = Field(..., description="N×16 的 Q，各分量 ∈ [-8, 8]")
    k: list[list[float]] = Field(..., description="N×16 的 K，各分量 ∈ [-8, 8]")
    v: list[list[float]] = Field(
        ..., description="N×16 的现存 V（至多一行被误写），各分量 ∈ [-8, 8]"
    )
    output: list[list[float]] = Field(
        ..., description="N×16 的归档输出，各分量 ∈ [-8, 8]"
    )

    @field_validator("doc_lengths")
    @classmethod
    def _check_doc_lengths(cls, value: list[int]) -> list[int]:
        if any(not isinstance(x, int) or isinstance(x, bool) or x <= 0 for x in value):
            raise ValueError("doc_lengths 必须为正整数列表")
        total = sum(value)
        if total == 0 or total > REPAIR_MAX_TOTAL_TOKENS:
            raise ValueError(
                f"文档总长度必须在 1..{REPAIR_MAX_TOTAL_TOKENS} 之间，当前为 {total}"
            )
        return value

    @field_validator("s")
    @classmethod
    def _check_s_finite(cls, value: float) -> float:
        if value != value or value in (float("inf"), float("-inf")):
            raise ValueError("s 必须为有限正数")
        return value

    @field_validator("q", "k", "v", "output")
    @classmethod
    def _check_tensor(cls, value: list[list[float]], info) -> list[list[float]]:
        name = info.field_name
        if len(value) == 0:
            raise ValueError(f"{name} 至少需要一行")
        for row in value:
            if len(row) != HEAD_DIM:
                raise ValueError(
                    f"{name} 每行必须为 {HEAD_DIM} 维，当前为 {len(row)}"
                )
            for x in row:
                fx = float(x)
                if not math.isfinite(fx):
                    raise ValueError(f"{name} 分量必须为有限值")
                if abs(fx) > REPAIR_BOX:
                    raise ValueError(
                        f"{name} 分量须在 [-{REPAIR_BOX:.0f}, {REPAIR_BOX:.0f}] 内，"
                        f"当前为 {fx}"
                    )
        return value

    @model_validator(mode="after")
    def _check_row_counts(self) -> "RepairValueRequest":
        n = sum(self.doc_lengths)
        for name in ("q", "k", "v", "output"):
            if len(getattr(self, name)) != n:
                raise ValueError(
                    f"{name} 行数须等于文档总长 {n}，当前为 {len(getattr(self, name))}"
                )
        return self


class RepairValueResponse(BaseModel):
    """/repair-value 响应体。

    仅当候选行唯一且分块算子、独立稠密参考回算误差均 ≤ 1e-8 时，
    ``repaired=True`` 并返回 ``row_index`` 与 ``recovered_value``；
    其余情形只说明原因，不给出可用修复。
    """

    status: str = Field(
        ...,
        description=(
            "repaired / no_difference / no_candidate / "
            "multiple_candidates / verification_failed"
        ),
    )
    repaired: bool
    reason: str = Field(..., description="结论说明（含无法修复时的原因）")
    row_index: int | None = Field(None, description="被误写行号（仅 repaired 时给出）")
    recovered_value: list[float] | None = Field(
        None, description="恢复出的 V 行（16 维，仅 repaired 时给出）"
    )
    recovered_widths: list[float] | None = Field(
        None, description="唯一候选行每维可行值范围宽度（均 ≤ 1e-3）"
    )
    candidate_rows: list[int] = Field(..., description="所有可行候选行")
    residual_max_abs: float = Field(
        ..., description="现存 V 回算输出与归档输出的最大绝对误差"
    )
    chunked_max_abs_err: float | None = Field(
        None, description="恢复值经分块算子回算与归档输出的最大绝对误差"
    )
    dense_max_abs_err: float | None = Field(
        None, description="恢复值经独立稠密参考回算与归档输出的最大绝对误差"
    )
    output_tol: float = REPAIR_OUTPUT_TOL
    width_tol: float = REPAIR_WIDTH_TOL
    scale_used: float = Field(..., description="实际进入 softmax 的系数 s/4")
    total_tokens: int
