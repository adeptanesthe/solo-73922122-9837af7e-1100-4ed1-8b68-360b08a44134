# 分块文档内因果注意力核对 API

把多篇文档拼入同一批次（总长 N ≤ 512），在 **CPU + float64** 下核对一个
自定义分块注意力 autograd 算子与独立稠密参考实现的前向输出及 Q/K/V 梯度。
另提供 **单行 V 误写修复** 接口：在 Q/K 未变、V 恰有一行被误写的前提下，
利用归档输出定位误写行并恢复其取值（详见 `POST /repair-value`）。

- 每个 token 只能读取**同一篇文档中位置不晚于自己**的 token
- 输出：`O = softmax(s · Q Kᵀ / 4) · V`
- Q/K/V 与上游梯度均为 16 维，float64
- 自定义算子前向、后向**均分块**计算，块长 `chunk_size ∈ [1, 32]`
- 不调用任何框架内置注意力算子，也不依赖内置 softmax 原语（online softmax 手写）
- 工作张量与跨前后向保存量规模 **O(N + C²)**，C 为块长；独立稠密参考不受此限制
- 容差：输出与 Q/K/V 梯度的最大绝对误差均 **≤ 1e-8**，否则 `passed=false` 并给出原因

## 目录结构

```
app/
  attention.py    # 自定义分块 autograd 算子（online softmax 前向 + 分块反向）
  reference.py    # 独立稠密参考（完整 N×N 掩码与注意力矩阵，不受内存约束）
  verification.py # 数据生成、双路计算与误差核对
  repair.py       # 单行 V 误写定位与恢复（/repair-value 核心逻辑）
  schemas.py      # 请求/响应模型与输入校验
  scenarios.py    # 内置边界情形样例
  main.py         # FastAPI 入口
tests/            # pytest：正确性、gradcheck、边界情形、内存与源码审计、API、修复
Dockerfile
docker-compose.yml
```

## 启动（Docker Compose）

```bash
docker compose up --build
# 服务监听 http://localhost:8000
# Swagger 文档：http://localhost:8000/docs
# OpenAPI：     http://localhost:8000/openapi.json
```

## 本地（无 Docker）运行

```bash
pip install -r requirements.txt   # torch 使用 CPU wheel 源
uvicorn app.main:app --host 0.0.0.0 --port 8000
pytest -q                         # 运行自测
```

## 接口

### `POST /verify`

请求体：

| 字段 | 类型 | 约束 |
| --- | --- | --- |
| `seed` | int | ≥ 0，决定 Q/K/V 与上游梯度的确定性生成 |
| `doc_lengths` | int[] | 正整数列表，总长 1..512 |
| `chunk_size` | int | 1..32 |
| `s` | float | 0 < s ≤ 1000 |

查询参数 `include_tensors`（默认 `true`）：是否回传完整输出/梯度张量。

curl 示例：

```bash
curl -X POST "http://localhost:8000/verify?include_tensors=false" \
  -H "Content-Type: application/json" \
  -d '{"seed":42,"doc_lengths":[7,11,5],"chunk_size":8,"s":1.0}'
```

响应关键字段：`passed`、`tolerance=1e-8`、`errors`（四项最大绝对误差）、
`finite_ok`、`failure_reasons`。核对失败是业务结果而非 HTTP 错误，仍返回 200，
通过 `passed=false` 与 `failure_reasons` 明确表达；参数非法返回 422。

### `POST /repair-value`

训练快照修复：同批次的 Q/K 未变，V 恰有一行被误写，手头保留了误写前的
归档输出。接口定位被误写的行并恢复其取值。

请求体：

| 字段 | 类型 | 约束 |
| --- | --- | --- |
| `doc_lengths` | int[] | 正整数列表，总长 1..32 |
| `chunk_size` | int | 1..32 |
| `s` | float | 0 < s ≤ 10 |
| `q` / `k` / `v` / `output` | float[][] | 均为 N×16（N 为文档总长），各分量 ∈ [-8, 8]；`v` 为现存（可能被误写一行）的 V，`output` 为归档输出 |

判定规则：

1. 用分块算子回算现存 V 的输出，与归档输出求残差；残差 ≤ 1e-8 视为
   **原输出无差异**（`status=no_difference`），不给修复。
2. 注意力输出对 V 是线性的（`O = P V`，P 仅由 Q/K 与掩码决定）。逐行假设
   “第 r 行被误写”：P 第 r 列为零的 query 行残差须 ≤ 1e-8；非零行给出
   恢复值区间 `V[r,d] + (R[i,d] ± 1e-8) / P[i,r]`。所有区间交会非空、
   每维可行宽度 ≤ 1e-3、且与 [-8, 8] 有交，该行才是候选。
3. **仅当候选行唯一**，且恢复值经**分块算子与独立稠密参考**回算、与归档
   输出的最大误差均 ≤ 1e-8 时，返回 `status=repaired`、`row_index` 与
   `recovered_value`（取可行区间中点）。
4. 无候选（`no_candidate`）、多候选（`multiple_candidates`）、回算未达标
   （`verification_failed`）都在 `reason` 中说明原因，不给可用修复。

curl 示例：

```bash
curl -X POST "http://localhost:8000/repair-value" \
  -H "Content-Type: application/json" \
  -d '{"doc_lengths":[3],"chunk_size":2,"s":2.0,
       "q":[[...16维...],...],"k":[[...],...],
       "v":[[...],...],"output":[[...],...]}'
```

响应关键字段：`status`、`repaired`、`reason`、`row_index`、
`recovered_value`、`recovered_widths`（每维可行宽度）、`candidate_rows`、
`residual_max_abs`、`chunked_max_abs_err`、`dense_max_abs_err`。

### `GET /scenarios` / `POST /scenarios/run`

列出或运行内置边界情形；`POST /scenarios/run?name=<名称>` 只跑单个样例。

## 内置边界情形（题目要求）

| 名称 | 配置 | 覆盖点 |
| --- | --- | --- |
| `doc_cross_chunk` | 文档 7/11/5，块长 8，s=1 | 文档边界落在块内部，切块不改变归一化 |
| `fully_masked_block` | 单文档 16，块长 4，s=2.5 | 整块被因果屏蔽时安全跳过、无 NaN |
| `large_score` | 文档 33/31，块长 32，s=1000 | 大分数下 online softmax 前后向无溢出 |
| `combined_stress` | 文档 100×4+112（N=512），块长 7，s=999 | 三类情形叠加 + 块长不整除 |
| `minimal_single_token` | 两篇各 1 token，块长 32 | 最小批次与退化分块循环 |

```bash
curl -X POST "http://localhost:8000/scenarios/run"
```

## 实现要点

1. **掩码**：`valid(i,j) = (doc_i == doc_j) ∧ (pos_j ≤ pos_i)`，按块即时构造 C×C 布尔掩码。
2. **前向 online softmax**：逐 key 块维护每行的运行最大值 `m`、归一化分母 `d`
   与归一化输出 `O`。遇到新块时以 `exp(m_old − m_new)` 修正旧累积量，
   全程只做最大值平移，不保存 N×N 矩阵。
3. **整块屏蔽**：块内无有效位置时直接跳过；行级 `-inf` 通过 `isfinite` 判定，
   以 0 参与指数运算，杜绝 `−inf − −inf`、`0/0` 产生的 NaN。
4. **后向**：用保存的行最大值/分母在每个块上**重算**块内概率 P（仅 C×C），
   利用 `dP = P ⊙ (dO·Vᵀ − Σ_j P_ij dO_iᵀV_j)` 与 `dS = scale·dP` 分块累加
   `dQ/dK/dV`；保存量仅为 Q/K/V、doc_ids、行最大值、行分母、输出，均为 O(N)。
5. **数值稳定**：float64 全程；s=1000 时原始分数很大，最大值平移保证
   `exp` 输入 ≤ 0，前后向均为有限值。
6. **确定性**：同一 `seed` 生成完全相同的输入与结果。

7. **V 行修复（/repair-value）**：利用输出对 V 的线性性，把“仅第 r 行被
   误写”的假设转化为逐维区间交会问题；候选须满足输出误差 ≤ 1e-8、
   每维可行宽度 ≤ 1e-3、恢复值 ∈ [-8, 8]，且唯一候选还需通过分块算子与
   独立稠密参考的双重回算（均 ≤ 1e-8）。对输出无影响的行（概率列下溢为
   零）无法定位，按“原输出无差异”处理；概率列过小的行可行宽度超限，
   按规格不给修复。

## 自测结果

`pytest -q` 共 39 项全部通过，包含：11 组参数化前向+梯度核对（误差均 ≤ 1e-8，
实测在 1e-12 ~ 1e-16 量级）、`torch.autograd.gradcheck` 有限差分核对、
块长不变性、确定性、掩码语义、大分数有限性、保存张量线性规模审计、
禁用算子源码审计、全部 API 端点与参数校验测试，以及 12 项 /repair-value
测试（唯一行修复、文档首/末 token、不同块长、单 token 批次、无差异、
双行误写无候选、输出被篡改无候选、概率列相同导致多候选、可行宽度超限、
恢复值越界、非法输入 422）。

另用 300 组随机批次（随机文档结构/块长/s/量程）做了单行误写压力测试：
可修复情形恢复值最大误差 2.5e-11；所有拒绝情形（无差异/无候选/多候选）
均复核为规格内的合理判定（误写行不可见或病态、候选确实不唯一）。
