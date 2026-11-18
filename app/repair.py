"""单行 V 误写修复：定位误写行并恢复训练快照。

原理
----
文档内因果注意力输出对 V 是线性的：``O = P V``，概率矩阵 P 仅由
Q/K、文档边界与缩放系数决定（Q/K 未变，故 P 可精确回算）。
若 V 的第 r 行被误写，则

    O_archived - O_current = P[:, r] ⊗ (V_orig[r] - V_current[r])

即残差只落在 P 第 r 列非零的那些 query 行上，且与列值成比例。
于是逐行假设“被误写的是第 r 行”，用区间交会检验该假设：

* P[i, r] = 0 的 query 行要求残差为 0（绝对值 ≤ 1e-8）；
* P[i, r] > 0 的每个 query 行对恢复值给出区间
  ``x ∈ V_current[r, d] + (R[i, d] ± 1e-8) / P[i, r]``；
* 所有区间交会非空、每维可行宽度 ≤ 1e-3、且与 [-8, 8] 有交，
  该行才是一个候选。

仅当候选行唯一时，用分块算子与独立稠密参考分别回算恢复后的输出，
两者与归档输出的最大误差均 ≤ 1e-8 才返回行号与恢复值。
"""

from __future__ import annotations

import math

import torch

from .attention import HEAD_DIM, NEG_INF, chunked_causal_attention
from .reference import dense_reference
from .schemas import (
    REPAIR_BOX,
    REPAIR_OUTPUT_TOL,
    REPAIR_WIDTH_TOL,
    RepairValueRequest,
    RepairValueResponse,
)
from .verification import make_doc_ids

# 候选行被拒绝的原因（用于无候选时的说明）
_REJECT_INVISIBLE = "该行对所有输出无影响"
_REJECT_RESIDUAL = "残差结构与该行不成比例"
_REJECT_BOX = "恢复值超出 [-8, 8]"
_REJECT_WIDTH = "可行值范围宽度超过 1e-3"


def _attention_probs(
    q: torch.Tensor, k: torch.Tensor, doc_ids: torch.Tensor, scale: float
) -> torch.Tensor:
    """文档内因果注意力的稠密概率矩阵（N ≤ 32，仅用于修复分析）。

    与分块算子、独立稠密参考使用同一掩码语义：
    ``valid(i, j) = (doc_i == doc_j) and (pos_j <= pos_i)``，
    手工最大值平移 softmax，不调用框架内置注意力/softmax 原语。
    """
    n = q.shape[0]
    pos = torch.arange(n, dtype=torch.int64)
    valid = (doc_ids[:, None] == doc_ids[None, :]) & (pos[None, :] <= pos[:, None])
    scores = scale * (q @ k.T)
    scores = torch.where(valid, scores, NEG_INF)
    # 每行至少含自身一个有效位置，行最大值有限，分母 ≥ exp(0) = 1。
    row_max = torch.amax(scores, dim=1, keepdim=True)
    exp_scores = torch.where(valid, torch.exp(scores - row_max), 0.0)
    return exp_scores / torch.sum(exp_scores, dim=1, keepdim=True)


def _check_row(
    col: list[float], resid: list[list[float]], v_row: list[float]
) -> tuple[bool, str, list[float] | None, list[float] | None]:
    """检验“仅第 r 行 V 被误写”的假设。

    参数
    ----
    col : P[:, r]（长度 N）
    resid : N×16 残差 ``O_archived - O_current``
    v_row : 现存 V 的第 r 行（16 维）

    返回 ``(是否候选, 拒绝原因, 每维恢复值, 每维可行宽度)``。
    """
    n = len(col)
    idx = [i for i in range(n) if col[i] > 0.0]
    if not idx:
        # 该行对任何输出都没有贡献（概率列全零，如数值下溢），
        # 无法从输出反推其取值，不能解释非零残差。
        return False, _REJECT_INVISIBLE, None, None
    # P[i, r] = 0 的 query 行：残差必须为 0，否则残差不可能只由第 r 行造成。
    for i in range(n):
        if col[i] == 0.0 and max(abs(x) for x in resid[i]) > REPAIR_OUTPUT_TOL:
            return False, _REJECT_RESIDUAL, None, None

    mids: list[float] = []
    widths: list[float] = []
    for d in range(HEAD_DIM):
        lo, hi = -math.inf, math.inf
        for i in idx:
            # 约束 |col[i] * (x - v_row[d]) - resid[i][d]| ≤ output_tol
            ratio = resid[i][d] / col[i]
            half = REPAIR_OUTPUT_TOL / col[i]
            lo_i = v_row[d] + ratio - half
            hi_i = v_row[d] + ratio + half
            if math.isnan(lo_i) or math.isnan(hi_i):
                # inf ∓ inf（概率列下溢到极小、约束半宽为 inf）：
                # 该 query 行不提供有效约束，跳过。
                continue
            lo = max(lo, lo_i)
            hi = min(hi, hi_i)
        if not lo <= hi:  # 区间交会为空（或出现 NaN）：残差与该行不成比例
            return False, _REJECT_RESIDUAL, None, None
        widths.append(hi - lo)
        lo_c = max(lo, -REPAIR_BOX)
        hi_c = min(hi, REPAIR_BOX)
        if lo_c > hi_c:  # 可行区间与 [-8, 8] 无交
            return False, _REJECT_BOX, None, None
        mids.append(0.5 * (lo_c + hi_c))
    if any(w > REPAIR_WIDTH_TOL for w in widths):
        return False, _REJECT_WIDTH, None, None
    return True, "", mids, widths


def run_repair(req: RepairValueRequest) -> RepairValueResponse:
    """定位被误写的 V 行并给出恢复值；无法唯一确定时只说明原因。"""
    doc_ids = make_doc_ids(req.doc_lengths)
    scale = float(req.s) / 4.0
    n = sum(req.doc_lengths)

    q = torch.tensor(req.q, dtype=torch.float64)
    k = torch.tensor(req.k, dtype=torch.float64)
    v = torch.tensor(req.v, dtype=torch.float64)
    o_arch = torch.tensor(req.output, dtype=torch.float64)

    # 用与训练相同的分块算子回算现存 V 的输出，得到残差。
    o_current = chunked_causal_attention(q, k, v, doc_ids, scale, req.chunk_size)
    resid_t = o_arch - o_current
    residual_max = float(torch.max(torch.abs(resid_t)).item())

    common = dict(
        scale_used=scale,
        total_tokens=n,
        residual_max_abs=residual_max,
        output_tol=REPAIR_OUTPUT_TOL,
        width_tol=REPAIR_WIDTH_TOL,
    )

    if residual_max <= REPAIR_OUTPUT_TOL:
        return RepairValueResponse(
            status="no_difference",
            repaired=False,
            reason=(
                f"现存 V 经分块算子回算与归档输出的最大绝对误差为 "
                f"{residual_max:.3e}（≤ {REPAIR_OUTPUT_TOL:.0e}），"
                "原输出无差异，无需修复，不给出修复值。"
            ),
            row_index=None,
            recovered_value=None,
            candidate_rows=[],
            **common,
        )

    probs = _attention_probs(q, k, doc_ids, scale).tolist()
    resid = resid_t.tolist()

    candidates: list[int] = []
    recovered: dict[int, tuple[list[float], list[float]]] = {}
    rejects: dict[str, int] = {}
    for r in range(n):
        col = [probs[i][r] for i in range(n)]
        ok, why, mids, widths = _check_row(col, resid, req.v[r])
        if ok:
            candidates.append(r)
            recovered[r] = (mids, widths)
        else:
            rejects[why] = rejects.get(why, 0) + 1

    if not candidates:
        detail = "；".join(f"{why} {cnt} 行" for why, cnt in sorted(rejects.items()))
        return RepairValueResponse(
            status="no_candidate",
            repaired=False,
            reason=(
                f"残差最大绝对误差 {residual_max:.3e} 超过 {REPAIR_OUTPUT_TOL:.0e}，"
                f"但没有任何单行 V 修改能在输出误差 {REPAIR_OUTPUT_TOL:.0e}、"
                f"每维可行宽度 ≤ {REPAIR_WIDTH_TOL:.0e}、恢复值 ∈ "
                f"[-{REPAIR_BOX:.0f}, {REPAIR_BOX:.0f}] 的约束下解释全部归档输出"
                f"（逐行检验：{detail}），不给修复。"
            ),
            row_index=None,
            recovered_value=None,
            candidate_rows=[],
            **common,
        )

    if len(candidates) > 1:
        return RepairValueResponse(
            status="multiple_candidates",
            repaired=False,
            reason=(
                f"第 {candidates} 行各自单独修改均可在容差内解释全部归档输出，"
                "候选行不唯一，无法确定误写位置，不给修复。"
            ),
            row_index=None,
            recovered_value=None,
            candidate_rows=candidates,
            **common,
        )

    row = candidates[0]
    mids, widths = recovered[row]
    v_fixed = v.clone()
    v_fixed[row] = torch.tensor(mids, dtype=torch.float64)

    # 分别用分块算子与独立稠密参考回算恢复后的输出，双重核对。
    o_chunk = chunked_causal_attention(q, k, v_fixed, doc_ids, scale, req.chunk_size)
    o_dense = dense_reference(q, k, v_fixed, doc_ids, scale)
    err_chunk = float(torch.max(torch.abs(o_chunk - o_arch)).item())
    err_dense = float(torch.max(torch.abs(o_dense - o_arch)).item())

    if err_chunk <= REPAIR_OUTPUT_TOL and err_dense <= REPAIR_OUTPUT_TOL:
        return RepairValueResponse(
            status="repaired",
            repaired=True,
            reason=(
                f"第 {row} 行为唯一候选；恢复值经分块算子与独立稠密参考回算，"
                f"与归档输出的最大误差分别为 {err_chunk:.3e}、{err_dense:.3e}"
                f"（均 ≤ {REPAIR_OUTPUT_TOL:.0e}）。"
            ),
            row_index=row,
            recovered_value=mids,
            recovered_widths=widths,
            candidate_rows=candidates,
            chunked_max_abs_err=err_chunk,
            dense_max_abs_err=err_dense,
            **common,
        )

    return RepairValueResponse(
        status="verification_failed",
        repaired=False,
        reason=(
            f"唯一候选行 {row} 的恢复值回算未达标：分块算子误差 "
            f"{err_chunk:.3e}、独立稠密参考误差 {err_dense:.3e}"
            f"（要求均 ≤ {REPAIR_OUTPUT_TOL:.0e}），不给修复。"
        ),
        row_index=None,
        recovered_value=None,
        candidate_rows=candidates,
        chunked_max_abs_err=err_chunk,
        dense_max_abs_err=err_dense,
        **common,
    )
