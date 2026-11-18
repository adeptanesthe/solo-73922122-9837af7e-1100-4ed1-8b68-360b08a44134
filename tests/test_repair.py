"""POST /repair-value 的端到端测试：定位误写行、恢复取值与各类拒绝情形。"""

from __future__ import annotations

import copy

import torch
from fastapi.testclient import TestClient

from app.attention import HEAD_DIM, chunked_causal_attention
from app.main import app
from app.verification import make_doc_ids

client = TestClient(app)


def _rand_box(n: int, seed: int, bound: float = 6.0) -> torch.Tensor:
    """生成各分量在 [-bound, bound] 内的 N×16 张量（留出误写余量）。"""
    g = torch.Generator().manual_seed(seed)
    return (torch.rand(n, HEAD_DIM, generator=g, dtype=torch.float64) * 2 - 1) * bound


def _archive(q, k, v, doc_lengths, chunk_size, s) -> torch.Tensor:
    """用分块算子计算归档输出（模拟训练时的前向）。"""
    doc_ids = make_doc_ids(doc_lengths)
    return chunked_causal_attention(q, k, v, doc_ids, s / 4.0, chunk_size)


def _payload(doc_lengths, chunk_size, s, q, k, v, out) -> dict:
    return {
        "doc_lengths": doc_lengths,
        "chunk_size": chunk_size,
        "s": s,
        "q": q.tolist(),
        "k": k.tolist(),
        "v": v.tolist(),
        "output": out.tolist(),
    }


def _post(payload: dict) -> dict:
    resp = client.post("/repair-value", json=payload)
    assert resp.status_code == 200, resp.text
    return resp.json()


# ---------------------------------------------------------------- 成功修复


def test_repair_unique_row_success():
    """多文档批次中误写一行：唯一定位并恢复，双路回算误差 ≤ 1e-8。"""
    doc_lengths = [5, 7, 3]
    n = sum(doc_lengths)
    q, k, v = _rand_box(n, 0), _rand_box(n, 1), _rand_box(n, 2)
    out = _archive(q, k, v, doc_lengths, 4, 2.0)

    bad_row = 8  # 落在第二篇文档内部
    v_bad = v.clone()
    v_bad[bad_row] = torch.linspace(-3.0, 3.0, HEAD_DIM, dtype=torch.float64)

    body = _post(_payload(doc_lengths, 4, 2.0, q, k, v_bad, out))
    assert body["status"] == "repaired"
    assert body["repaired"] is True
    assert body["row_index"] == bad_row
    assert body["candidate_rows"] == [bad_row]
    recovered = torch.tensor(body["recovered_value"], dtype=torch.float64)
    assert torch.max(torch.abs(recovered - v[bad_row])).item() < 1e-6
    assert all(0.0 <= w <= 1e-3 for w in body["recovered_widths"])
    assert body["chunked_max_abs_err"] <= 1e-8
    assert body["dense_max_abs_err"] <= 1e-8
    assert body["residual_max_abs"] > 1e-8


def test_repair_first_token_of_doc():
    """误写行为某篇文档的第一个 token（该行自注意力概率为 1）。"""
    doc_lengths = [4, 6]
    n = sum(doc_lengths)
    q, k, v = _rand_box(n, 10), _rand_box(n, 11), _rand_box(n, 12)
    out = _archive(q, k, v, doc_lengths, 3, 1.5)

    bad_row = 4  # 第二篇文档开头
    v_bad = v.clone()
    v_bad[bad_row] = -v_bad[bad_row]

    body = _post(_payload(doc_lengths, 3, 1.5, q, k, v_bad, out))
    assert body["status"] == "repaired"
    assert body["row_index"] == bad_row
    recovered = torch.tensor(body["recovered_value"], dtype=torch.float64)
    assert torch.max(torch.abs(recovered - v[bad_row])).item() < 1e-6


def test_repair_last_row_and_chunk_boundaries():
    """误写最后一行；同一数据在不同块长下都能修复。

    注意：文档末行只被自身 query 观测（P 列仅一个非零元），
    需要注意力分布不太尖锐才能满足可行宽度 ≤ 1e-3，
    因此这里 Q/K 取较小量程（[-1, 1]）。
    """
    doc_lengths = [9, 7]
    n = sum(doc_lengths)
    q, k, v = _rand_box(n, 20, 1.0), _rand_box(n, 21, 1.0), _rand_box(n, 22)
    bad_row = n - 1
    v_bad = v.clone()
    v_bad[bad_row] = 0.25

    for chunk_size in (1, 5, 32):
        out = _archive(q, k, v, doc_lengths, chunk_size, 0.5)
        body = _post(_payload(doc_lengths, chunk_size, 0.5, q, k, v_bad, out))
        assert body["status"] == "repaired", (chunk_size, body["reason"])
        assert body["row_index"] == bad_row
        recovered = torch.tensor(body["recovered_value"], dtype=torch.float64)
        assert torch.max(torch.abs(recovered - v[bad_row])).item() < 1e-6


def test_repair_single_token_batch():
    """N=1 的退化批次：唯一的行被误写。"""
    q, k, v = _rand_box(1, 30), _rand_box(1, 31), _rand_box(1, 32)
    out = _archive(q, k, v, [1], 1, 10.0)
    v_bad = torch.zeros_like(v)

    body = _post(_payload([1], 1, 10.0, q, k, v_bad, out))
    assert body["status"] == "repaired"
    assert body["row_index"] == 0
    recovered = torch.tensor(body["recovered_value"], dtype=torch.float64)
    assert torch.max(torch.abs(recovered - v[0])).item() < 1e-6


# ---------------------------------------------------------------- 拒绝情形


def test_no_difference_when_v_unchanged():
    """现存 V 与归档输出一致：报告无差异，不给修复。"""
    doc_lengths = [3, 4]
    n = sum(doc_lengths)
    q, k, v = _rand_box(n, 40), _rand_box(n, 41), _rand_box(n, 42)
    out = _archive(q, k, v, doc_lengths, 2, 2.0)

    body = _post(_payload(doc_lengths, 2, 2.0, q, k, v, out))
    assert body["status"] == "no_difference"
    assert body["repaired"] is False
    assert body["row_index"] is None
    assert body["recovered_value"] is None
    assert body["residual_max_abs"] <= 1e-8


def test_no_candidate_when_two_rows_differ():
    """残差由两行变化造成：任何单行假设都不成立，报告无候选。"""
    doc_lengths = [5, 7, 3]
    n = sum(doc_lengths)
    q, k, v = _rand_box(n, 50), _rand_box(n, 51), _rand_box(n, 52)
    out = _archive(q, k, v, doc_lengths, 4, 2.0)

    v_bad = v.clone()
    v_bad[3] += 0.5  # 第一篇文档内
    v_bad[8] -= 0.75  # 第二篇文档内

    body = _post(_payload(doc_lengths, 4, 2.0, q, k, v_bad, out))
    assert body["status"] == "no_candidate"
    assert body["repaired"] is False
    assert body["row_index"] is None
    assert body["recovered_value"] is None
    assert body["candidate_rows"] == []


def test_no_candidate_when_output_tampered():
    """归档输出本身被改动（非单行 V 可解释）：报告无候选。"""
    doc_lengths = [6, 4]
    n = sum(doc_lengths)
    q, k, v = _rand_box(n, 60), _rand_box(n, 61), _rand_box(n, 62)
    out = _archive(q, k, v, doc_lengths, 4, 2.0)

    out_bad = out.clone()
    out_bad[2, 0] += 0.1
    out_bad[7, 3] -= 0.2  # 两处独立扰动，单行 V 无法解释

    body = _post(_payload(doc_lengths, 4, 2.0, q, k, v, out_bad))
    assert body["status"] == "no_candidate"
    assert body["repaired"] is False


def test_multiple_candidates_not_unique():
    """两行概率列完全相同（数值下溢导致）：候选不唯一，不给修复。

    构造：q1·k0 = +1024、q1·k1 = -1024，s=10 时分数 ±2560，
    exp(-5120) 下溢为 0，使 P[1, 1] = 0；q2 = 0 使 P[2, :] 均匀。
    于是 P 的第 1、2 列完全相同，误写第 1 行与第 2 行都能解释输出。
    """
    doc_lengths = [3]
    q = torch.zeros(3, HEAD_DIM, dtype=torch.float64)
    q[1] = 8.0
    k = torch.zeros(3, HEAD_DIM, dtype=torch.float64)
    k[0] = 8.0
    k[1] = -8.0
    k[2] = 0.25
    v = _rand_box(3, 70, bound=5.0)
    s = 10.0
    out = _archive(q, k, v, doc_lengths, 2, s)

    v_bad = v.clone()
    v_bad[1] += 1.0  # 误写第 1 行（第 2 行同样能解释残差）

    body = _post(_payload(doc_lengths, 2, s, q, k, v_bad, out))
    assert body["status"] == "multiple_candidates"
    assert body["repaired"] is False
    assert body["row_index"] is None
    assert body["recovered_value"] is None
    assert set(body["candidate_rows"]) == {1, 2}


def test_no_candidate_when_width_too_wide():
    """误写行的概率列过小：可行值范围宽度超过 1e-3，报告无候选。

    q1·k0 = 5.6、q1·k1 = 0，s=10 时 P[1,1] ≈ 8.3e-7，
    可行半宽 1e-8 / 8.3e-7 ≈ 0.012，宽度 ≈ 0.024 > 1e-3。
    """
    doc_lengths = [2]
    q = torch.zeros(2, HEAD_DIM, dtype=torch.float64)
    q[1, 0] = 1.4
    k = torch.zeros(2, HEAD_DIM, dtype=torch.float64)
    k[0, 0] = 4.0
    v = _rand_box(2, 80, bound=5.0)
    s = 10.0
    out = _archive(q, k, v, doc_lengths, 2, s)

    v_bad = v.clone()
    v_bad[1] += 0.75

    body = _post(_payload(doc_lengths, 2, s, q, k, v_bad, out))
    assert body["status"] == "no_candidate"
    assert body["repaired"] is False
    assert body["candidate_rows"] == []
    assert "宽度" in body["reason"]


def test_no_candidate_when_recovered_out_of_box():
    """恢复值落在 [-8, 8] 之外：该行不是合法候选，报告无候选。"""
    doc_lengths = [2]
    q = torch.zeros(2, HEAD_DIM, dtype=torch.float64)
    k = torch.full((2, HEAD_DIM), 0.3, dtype=torch.float64)  # P[1] = (0.5, 0.5)
    v_orig = torch.zeros(2, HEAD_DIM, dtype=torch.float64)
    v_orig[1, 0] = 8.5  # 归档时的真实值越界（现存 V 无法合法恢复）
    out = _archive(q, k, v_orig, doc_lengths, 2, 1.0)
    assert out.abs().max().item() <= 8.0  # 归档输出本身仍在 [-8, 8] 内

    v_curr = torch.zeros_like(v_orig)  # 第 1 行被误写为 0

    body = _post(_payload(doc_lengths, 2, 1.0, q, k, v_curr, out))
    assert body["status"] == "no_candidate"
    assert body["repaired"] is False
    assert "[-8, 8]" in body["reason"]


# ---------------------------------------------------------------- 输入校验


def _valid_payload() -> dict:
    doc_lengths = [3, 2]
    n = sum(doc_lengths)
    q, k, v = _rand_box(n, 90), _rand_box(n, 91), _rand_box(n, 92)
    out = _archive(q, k, v, doc_lengths, 2, 2.0)
    return _payload(doc_lengths, 2, 2.0, q, k, v, out)


def test_validation_rejects_bad_requests():
    base = _valid_payload()

    def expect_422(mutate):
        payload = copy.deepcopy(base)
        mutate(payload)
        resp = client.post("/repair-value", json=payload)
        assert resp.status_code == 422, (payload, resp.text)

    # 总长超过 32
    expect_422(lambda p: p.update(doc_lengths=[20, 13]))
    # 非正文档长度
    expect_422(lambda p: p.update(doc_lengths=[3, 0]))
    # 块长越界
    expect_422(lambda p: p.update(chunk_size=0))
    expect_422(lambda p: p.update(chunk_size=33))
    # s 越界（0 与 >10）
    expect_422(lambda p: p.update(s=0.0))
    expect_422(lambda p: p.update(s=10.5))
    # 张量分量超出 [-8, 8]
    expect_422(lambda p: p["q"][0].__setitem__(0, 8.5))
    expect_422(lambda p: p["v"][1].__setitem__(0, -9.0))
    expect_422(lambda p: p["output"][2].__setitem__(0, 8.1))
    # NaN / Inf 分量
    expect_422(lambda p: p["k"][0].__setitem__(0, float("nan")))
    expect_422(lambda p: p["output"][0].__setitem__(0, float("inf")))
    # 维度不是 16
    expect_422(lambda p: p["q"][0].pop())
    # 行数与文档总长不符
    expect_422(lambda p: p["v"].pop())
    expect_422(lambda p: p.update(doc_lengths=[3, 3]))


def test_repair_endpoint_in_openapi():
    openapi = client.get("/openapi.json").json()
    assert "/repair-value" in openapi["paths"]
    assert "/verify" in openapi["paths"]
