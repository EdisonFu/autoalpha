# AutoAlpha Web 控制台

分步可视化界面：研究方向 → 论文检索 → 因子假设 → 代码实现 → IS/OOS 双层回测。

## 启动

```bash
export LLM_API_KEY=...                     # 你的 LLM 密钥
export LLM_MODEL=<model-id>                # 模型 ID
# 若用 OpenAI 兼容端点，再加：
# export LLM_BASE_URL=https://.../v1
# 可选：按百万 token 计价，用于成本统计
# export LLM_PRICE_INPUT=3.0
# export LLM_PRICE_OUTPUT=15.0

.venv/bin/python main.py --web             # 默认 http://127.0.0.1:8000
.venv/bin/python main.py --web --port 8765 # 指定端口
```

数据源默认指向 `factor-signal/data/klines`，可用环境变量覆盖：

```bash
export AUTOALPHA_KLINES=/path/to/klines    # 需含 etf-daily/ 子目录
```

## 两种模式

| 模式 | 行为 |
|---|---|
| **分步运行** | 每阶段暂停等你确认：选论文 → 选因子假设 → 回测 |
| **全自动** | 自动取第 1 篇论文，一路跑到回测完成 |

## 成本

单次完整运行（10 篇论文 → 3 假设 → 2 因子）实测约 **$0.15**（中等模型计价）。

页面有「Token 用量」卡片实时显示，也可查：

```bash
curl -s http://127.0.0.1:8765/api/usage | python3 -m json.tool
```

成本构成与省钱开关见 `docs/TOKEN_COST.md`。

## 灵感池（RSS）

量化博客与机构的最新文章，回答「最近业界在研究什么」。

```bash
curl -X POST http://127.0.0.1:8765/api/feeds/refresh      # 抓取入库
curl -X POST http://127.0.0.1:8765/api/feeds \
     -H 'Content-Type: application/json' \
     -d '{"keywords":["momentum","etf"],"limit":10}'      # 本地筛选
```

实测源（均带摘要）：AlphaArchitect 5 · Quantpedia 10 · QuantInsti 15 · NBER 35。

与 Step 2 检索的区别：RSS **不支持按关键词查**，只给最新 N 条。
所以这里是「先抓取入库 → 本地筛选」，不是检索源。页面上可按当前研究意图过滤。

## 界面展示什么

- **数据与样本切分** — 标的数、截面宽度、IS/OOS 区间与天数、由宽度推导的 IC/t 门槛
- **论文列表** — 每篇带**中文概述**（这篇做了什么）与**相关性**（对因子挖掘有何用），
  可多选，或跳过直接用原始想法生成假设
- **因子假设** — 每个带**中文说明**（衡量什么 / 经济逻辑 / 计算要点）+ 公式、参数，
  可勾选要实现哪些
- **Token 用量** — 调用次数、输入/输出 token、估算花费、按用途分解
- **灵感池** — 最新业界文章，可按研究意图筛选
- **回测结果** — 每个因子一张卡片：
  - 多空组合净值曲线（SVG，标注 1.0 基线）
  - IS / OOS 并排对比：IC、t 值、ICIR（日/年双口径）、月度胜率、覆盖率
  - 分层单调性 + 各分位年化收益柱状图
  - 多空 Sharpe（含扣费）、年化、回撤、换手
  - 未通过时显示**具体哪道闸门没过**

## 双层门控

```
IS 层   t 值 > 推导门槛 · 单调性 > 0.6 · 多空Sharpe > 0.5 · 覆盖率 > 0.7 · 换手 < 0.3
OOS 层  扣费Sharpe > 0.3 · Sharpe衰减 > -50% · 单调性 > 0.4 · 净值相关 < 0.7
```

IS 未过的因子**不会**访问 OOS 数据 —— 切分由 `engine.SplitEngine` 强制，
不依赖调用方自觉。OOS 重复读取会打警告。

门槛不是硬编码：`ic_mean_floor = t/√n_obs/√width`，随截面宽度自动调整。

## API

| 端点 | 说明 |
|---|---|
| `POST /api/start` | `{idea, mode}` → `{session_id}` |
| `POST /api/select` | `{session_id, paper_indices}` |
| `POST /api/run` | `{session_id, spec_indices}` |
| `POST /api/extend` | 为通过的因子生成延伸假设 |
| `GET /api/events/{sid}` | SSE 事件流 |
| `GET /api/session/{sid}` | 会话快照（含全部评估结果） |
| `GET /api/library` | 已入库因子 |

## 注意

- 改动 `autoalpha/` 下任何文件后**必须重启服务**，否则仍跑旧代码
- LLM 调用较慢，单个因子从生成到回测约需 10-40 秒。
  代码生成阶段默认 3 路并发（`FlowOrchestrator(max_workers=3)`）——
  实测单因子回测仅 1.9s，耗时几乎全在 LLM 调用与沙箱执行上。
  评估阶段保持串行：去重基线需按确定顺序累积。
- 大部分因子会 FAIL_IS，这是门控在工作 —— 弱信号区采纳率低是预期行为
