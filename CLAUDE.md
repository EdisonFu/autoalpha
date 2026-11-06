# AutoAlpha 开发指引

输入研究方向 → 检索文献 → 提取因子假设 → 生成代码 → IS/OOS 双层回测 → 因子库。

本文件只写**动手前必须知道的事**。设计论证在 `docs/`，不在这里重复。

| 文档 | 内容 |
|---|---|
| `docs/P0_BACKTEST_DESIGN.md` | 回测层设计；附录 A 记录实施中发现的问题 |
| `docs/E2E_REVIEW.md` | 端到端复盘；附录 B-E 是历次实测结论 |
| `docs/TOKEN_COST.md` | 成本实测与省钱开关 |
| `docs/OPTIMIZATION_PLAN_2026Q4.md` | 文献调研与优化路线 |
| `README_WEB.md` | Web 控制台用法 |

---

## 跑起来

```bash
.venv/bin/python main.py --web --port 8765     # 有 .env 时直接起
```

首次配置：`cp .env.example .env` 填入 `LLM_API_KEY` / `LLM_MODEL`
（OpenAI 兼容端点再加 `LLM_BASE_URL`），或在页面「LLM 配置」里填并勾选
「记住配置」。**进程环境变量 > 项目 `.env` > `~/.autoalpha/.env`**，
临时换模型用 export，不改文件。
`.env` 已 gitignore、写入权限 600。推理模型的输出上限见 `llm.max_tokens_for`。

数据默认取 `factor-signal/data/klines`（`AUTOALPHA_KLINES` 可覆盖）。
计时或冒烟测试用 `scripts/drive_pipeline.py`，别手写 curl 循环（理由见 §踩过的坑）。

---

## 流水线

```
Step 1 intent     意图识别：用户表述 → 规范学术检索式
Step 2 search     多源召回 + 跨源去重 + 回填摘要 + 滤非研究条目
       data       数据准备：选源取数、按资产类别筛池、建 IS/OOS 切分
       extract    提取因子假设（arXiv 读全文，其余读摘要并标注）
       implement  LLM 按骨架生成代码 → 沙箱执行（3 路并发）
       evaluate   IS/OOS 双层门控（串行）
       reflect    反思本轮 → 产出下一轮方向 → 回到 Step 1（loop）
```

分步模式每阶段暂停等确认；`mode=auto` 一路跑完。

### 模块职责

| 模块 | 职责 |
|---|---|
| `agents/intent_agent.py` | Step 1，输出 `SearchIntent` |
| `sources/` | 各检索源适配 + 跨源去重，统一 schema |
| `sources/fulltext.py` | arXiv 全文抓取 + 方法论章节抽取（可选 pypdf） |
| `sources/backfill.py` | 跨库补摘要（OpenAlex↔Crossref）+ 滤非研究条目 |
| `sources/feeds.py` | RSS 灵感池（独立于主链路） |
| `metrics.py` | 指标容器，字段名自带口径 |
| `backtest.py` | 分层组合、净值、换手、扣费 |
| `engine.py` | 持有 IS/OOS 切分、门槛推导、双层门控 |
| `universe.py` | 资产分类与截面池构建 |
| `data_config.py` | 数据源注册与选路（不配置走免费，配了走用户的） |
| `orchestrator.py` | 分阶段编排，每阶段 yield `FlowEvent` |
| `json_repair.py` | 容错解析 LLM 输出 |
| `flow.py` / `server.py` | 事件模型 / FastAPI + SSE |

---

## 不可动摇的约定

违反这些会产出看起来正常、实际错误的结果。

**1. 指标名必须带口径。**
不允许裸 `icir`。日频是 `icir_daily`，年化是 `icir_annual`，显著性看 `ic_tstat`。
历史教训：README 写 `icir > 0.5`（月频口径）而代码算日频，13 个「通过」的
因子实际 ICIR 只有 0.075–0.106。

**2. 门槛由数据推导，不写死。**
`ic_mean_floor = t / sqrt(n_obs) / sqrt(width)`。
写死的 `0.02` 是按 24 个标的算的；换成 117 个标的后它过严 2.3 倍，
而代码里没有任何地方记录这个耦合。换手门槛同理，按调仓频率推导。

**3. 搜索/修复阶段不得访问 OOS。**
切分由 `engine.SplitEngine` 持有，不靠调用方自觉。OOS 每因子只读一次，
重复读会打警告。防泄漏必须是结构性的——泄漏扛得住任何统计修正。

**4. 时序纪律：`weights.shift(1)`。**
t 日收盘定的权重吃 t+1 日收益。`backtest.compute_nav` 的契约。

**5. 评估阶段保持串行。**
`passed_navs` 是去重基线，必须按确定顺序累积，否则判重结果随线程调度变化。
并发只作用于 implement 阶段。

**6. 兜底源的优先级必须排在最后。**
`data_config.PROVIDERS` 里 `local` 的 priority 是 999。曾设成 5（最优先），
结果用户配了 Tushare 账号也永远走不到——配置形同虚设。
凭据与 LLM 密钥同样只存内存、不落盘、不回传明文。

**7. 检索必须区分市场。**
不同市场的制度与微观结构差异大（涨跌停、散户占比、做空限制、结算），
一个市场成立的结论常常不能平移。`SearchIntent.market` 从输入推断市场，
并额外生成一条**市场词领头的短查询**。

市场词不能缀在长句尾部——实测 Crossref 在长查询上排序很差：
`"cross-sectional momentum in exchange-traded funds Chinese A-share"`
召回的是 *Exchange-traded notes*，而 `"Chinese A-share ETF momentum"`
召回的是 *ETF Momentum / How to Improve ETF Sector Momentum*。

界面示例一律带市场（「A股ETF截面动量轮动」），未识别到市场时在意图面板提示。

**8. 参数来源必须如实标注。**
只读摘要时 LLM 产出的 `lookback=20` 之类是它自己定的 —— 实测摘要中位
833 字符，7 项复现细节命中 0–1 项，而 arXiv 全文命中 4–6 项。
`FactorSpec.param_sources` 把每个参数标成 `paper`/`inferred`/`default`，
且**非 fulltext 路径一律降级为 default** —— 实测 LLM 会把自己定的窗口
谎报成 `paper`。把编的说成读的，比不标更糟。

**9. 补资料优先查库，不要爬网。**
缺摘要多半不是「被墙了」，而是「这个源的这条记录里没有」——
同一 DOI 在另一家库里往往就有。实测 OpenAlex 与 Crossref 零重叠、
完全互补，批量回查补全 59% 仅需 4s、零依赖；而抓出版商网页
（urllib / curl_cffi / Scrapling 三者**数字完全相同**）只有 6%，
最慢要 1147s。原因是这些 URL 几乎全是 doi.org，卡在连不通与登录墙上，
指纹伪装对这两种失败都无效。引入爬虫框架前先确认卡点是不是反爬。

**10. 代码生成要给可运行骨架，不能只给签名。**
实测一次真实运行：prompt 只描述了 `data: dict[股票代码, DataFrame]`，
LLM 却写成按字段名取 `data["close"]` → KeyError。它被归成 DATA，
拿到 dropna()/min_periods 的提示，而 repair prompt 又要求「最小修改」，
于是三次重试全在原地打转。
→ `_CODEGEN_SYSTEM` 现在给出**经真实数据验证过**的完整骨架
（数据访问 + 截面 rank + stack + main 块），LLM 只填核心逻辑。
结构性误解单列为 `STRUCTURE`，明确允许重写数据访问层；
`base_parts` 附真实数据形状（具体 key、shape、列名），不让 LLM 靠猜。
改骨架后必须跑 `test_implement_structure.py` —— 里面有条测试直接
把 prompt 里的骨架抽出来在真实数据上执行，模板错了会当场炸。

**11. 研究循环必须记账。**
auto research 是 loop：每轮结束由 `research_loop_agent` 读失败原因，
产出下一轮检索式，回到 Step 1。但每多跑一轮就是在同一批数据上多搜一次，
选择性偏差累积 —— 4 轮 40 个因子选出 3 个，和首轮选出 3 个含义完全不同。
`sess.rounds` / `n_factor_trials` 必须跨轮保留，反思失败也要记账；
到 `MAX_ROUNDS` 要明确提示偏差，不能无限循环。
新一轮的 IS/OOS 切分必须复用（`o._engine` 不重置），否则跨轮不可比。

**12. 失败必须可见。**
静默降级是本项目最贵的教训（见 §踩过的坑）。解析失败要留下可读提示，
不要返回空值。`factor_judge` 解析失败时 **REJECT 而非 APPROVE**——
审查缺失是扣留因子的理由，不是放行的理由。

---

## 踩过的坑

每条都真实发生过，且不止一次浪费时间。

### 静默降级 → 用户看到的是「功能没了」

中文概述消失过两次。根因是 LLM 在中文里给术语加英文引号（`衡量"动量"强度`），
`json.loads` 抛错，而降级逻辑返回空串。用户看到的是「中文又没了」，
而不是「翻译失败了」。
→ 所有 LLM 输出走 `json_repair.loads_lenient`，失败填提示文字。

同类：prompt 里曾指示「摘要信息不足时写『需查看全文』」，模型照做，
而检索结果里空摘要不少，于是这句占位话频繁出现。
→ 不要给模型准备占位话术，让它就现有信息写能写的。

### 改了代码要重启服务

补丁写进文件但服务是补丁前启动的，跑的仍是旧代码——曾误判「修复无效」。

### 判断阶段是否结束：看真实计数，不看 `stage`

分步模式下 `stage` 停在 `"intent"` 既表示进行中也表示已完成，无法区分。
要看 `n_specs`、`len(evaluations)`。`/api/run` 返回 409 表示上一阶段在跑，重试即可。

### 后台诊断脚本必须无缓冲输出

`print` 被全缓冲时看不到任何进度，会把「看不见」误判成「卡死」，
然后去查服务端、查线程、查端口——全是白费。

### 起对照实例要验证它真的起来了

端口被占时 `bind` 失败，脚本静默连到残留的旧实例，
**产出的数字看起来完全正常但毫无意义**。曾因此两轮 A/B 数据作废。

### 确认代码路径是否执行，要看它写到哪里

曾因服务端日志里没有某条消息就断定「并发没触发」。
那是 `FlowEvent`（走 SSE），本来就不落在 stdout。证据来源用错了。

### 布局 bug 逃得过逻辑测试

前端测试原来只在 Node 里跑逻辑（构造假 DOM、断言 innerHTML），
数据源弹窗在矮视口下内容被裁掉且无法滚动——15 条逻辑断言全绿，
用户一打开就撞上。

凡涉及**尺寸、滚动、可见性、溢出**的断言，必须在真实浏览器里做
（`tests/test_e2e_ui.py`，playwright）。假 DOM 没有布局引擎，
`scrollHeight` / `bounding_box` 这类值在那里没有意义。

E2E 用系统已装的 Chrome（`channel="chrome"`）。本环境网络白名单拦掉了
`cdn.playwright.dev`（实测 307 且零字节），装不了 playwright 自带 chromium。

### 测试不可靠时先怀疑测试

A/B 脚本连错七次。坏脚本确实撞出过一个真 bug（`_spawn` 缺重入保护），
但那是运气不是方法。

---

## 开发

```bash
.venv/bin/python -m pytest tests/ -q          # 单元 + 集成
.venv/bin/python -m pytest tests/test_e2e_ui.py -q   # 真实浏览器
.venv/bin/python scripts/drive_pipeline.py --port 8765   # 冒烟
```

测试里有几条是能力证明，不是形式：泄漏因子必须被抓出（Sharpe>3、单调性>0.9）、
随机因子必须无边际、`shift` 与不 `shift` 结果必须不同。改回测层后这些必须仍通过。

**提交不加任何 AI 署名。** 代码与文档里不写具体模型型号，由 `LLM_MODEL` 指定。

---

## 现状与下一步

已完成：回测层、Web 控制台、意图识别、多源检索、RSS 灵感池、并发实现。

实测数据：单次运行约 $0.15–0.22；并发端到端约 1.3x（受控上限 3x）；
多源召回 115 条→去重 110（原单源 25 条）。

未做：
- **P2-2 内层 GP + 外层 LLM** —— 真正的提速杠杆，预期降一个数量级 LLM 调用
- 评估阶段并行 —— 需重新设计去重基线累积方式
- 搜索记账 + deflation —— 自用场景是加分项，但 77 次试验选 13 个的选择性偏差真实存在
