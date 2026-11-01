# 参与贡献

先读 `CLAUDE.md` —— 它写了动手前必须知道的约定，以及踩过的坑。

## 本地跑起来

```bash
git clone https://github.com/EdisonFu/autoalpha.git
cd autoalpha
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

export LLM_API_KEY=...        # 你的 LLM 密钥
export LLM_MODEL=<model-id>   # 模型 ID
python main.py --web
```

不配置数据平台账号也能跑 —— 此时走公共数据查询接口。

## 测试

```bash
pytest tests/ --ignore=tests/test_e2e_ui.py   # 单元 + 集成，无需网络
pytest tests/test_e2e_ui.py                   # 真实浏览器，需本地 Chrome
```

CI 只跑前者。E2E 需要浏览器和运行中的服务。

### 有几条测试是能力证明，不是形式

改动回测层后它们必须仍然通过：

- 用未来收益构造的泄漏因子**必须被抓出**（Sharpe > 3、单调性 > 0.9）
- 纯随机因子**必须显示无边际**
- 权重 `shift(1)` 与不 shift 的结果**必须不同**

这三条失效，意味着回测不再能区分真信号与噪声。

## 提 PR 之前

- [ ] `pytest tests/ --ignore=tests/test_e2e_ui.py` 通过
- [ ] 涉及尺寸/滚动/可见性的前端改动，在 `tests/test_e2e_ui.py` 里加断言
      （假 DOM 没有布局引擎，逻辑测试验证不了这些）
- [ ] 不提交密钥、不提交具体模型型号（由 `LLM_MODEL` 指定）
- [ ] commit message 说清**为什么**，而不只是改了什么

## 容易踩的地方

`CLAUDE.md` 的「不可动摇的约定」和「踩过的坑」两节值得一读。几条高频的：

- **指标名必须带口径**。不允许裸 `icir`，要写 `icir_daily` / `icir_annual`。
- **门槛由数据推导，不写死**。写死的阈值换个数据集就失效。
- **搜索/修复阶段不得访问 OOS**。切分由引擎持有，不靠调用方自觉。
- **失败必须可见**。静默降级会让用户以为"功能没了"而不是"出错了"。

## 报 Bug

请附上：复现步骤、预期与实际行为、`pytest` 输出或服务端日志片段。
涉及因子结果的问题，带上因子名与具体指标值会快很多。
