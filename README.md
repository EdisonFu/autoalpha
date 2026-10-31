# AutoAlpha — 多Agent量化因子挖掘流水线

> 输入一个研究方向，自动检索相关论文、提取因子假设、实现并评估因子质量，输出因子库与 HTML 质量报告。

---

## 功能概览

```
用户输入研究方向（如"动量因子"）
    │
    ▼
┌─────────────────┐
│  论文检索         │  搜索 arXiv q-fin 领域相关论文，LLM 语义打分，综合排序
└────────┬────────┘
         ▼
┌─────────────────┐
│  因子提取         │  LLM 从论文摘要中提取 2-4 个可量化因子假设
└────────┬────────┘
         ▼
┌─────────────────┐
│  因子实现         │  LLM 生成 Python 代码，沙箱（subprocess）执行，输出因子值 CSV
└────────┬────────┘
         ▼
┌─────────────────┐
│  因子评估         │  计算 IC/ICIR/衰减比/覆盖率/月胜率，与质量门控对比
└────────┬────────┘
         ▼
┌─────────────────┐
│  因子延伸         │  对通过评估的因子生成 3-5 个变体，自动评估
└────────┬────────┘
         ▼
┌─────────────────┐
│  生成报告         │  输出 HTML 因子质量报告 + 写入 SQLite 因子库
└─────────────────┘
```

## 因子质量门控

| 指标 | 门槛 | 说明 |
|------|------|------|
| IC 均值 | > 0.03 | 截面预测能力 |
| ICIR | > 0.5 | 信号稳定性 |
| IC 衰减比（5d/1d） | > 0.5 | 信号半衰期 |
| 因子覆盖率 | > 0.8 | 避免稀疏 |
| 月度胜率 | > 0.55 | IC > 0 的月份占比 |
| 与已有因子最大相关 | < 0.7 | 去相关 |

---

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置 LLM API Key

```bash
export ANTHROPIC_API_KEY=sk-ant-...   # 推荐，使用 Claude
# 或
export DEEPSEEK_API_KEY=sk-...        # DeepSeek（性价比高）
# 或
export OPENAI_API_KEY=sk-...          # OpenAI GPT-4o
```

### 3. 运行

```bash
python main.py
```

输入研究方向，例如：
```
❯ 动量因子选股策略
❯ 基于换手率的流动性因子
❯ 机器学习多因子模型
```

---

## 交互命令

| 命令 | 说明 |
|------|------|
| `<研究方向>` | 搜索论文并运行完整流水线 |
| `/library` | 查看已通过评估的因子库 |
| `/data <路径>` | 注册本地数据文件（CSV/Parquet/Excel） |
| `/source add tushare --token <TOKEN>` | 添加 VIP 数据源 |
| `/help` | 查看完整帮助 |
| `q` | 退出 |

---

## 项目结构

```
autoalpha/
├── main.py                          # 交互式 CLI 入口
├── requirements.txt
└── autoalpha/
    ├── llm.py                       # LLM 客户端（支持 多家 LLM 供应商）
    ├── pipeline.py                  # 主流水线编排
    ├── pipeline_state.py            # 数据结构：FactorSpec / FactorResult / FactorReport
    ├── db.py                        # SQLite 因子库
    ├── data_source.py               # 数据源适配器（AKShare/Tushare/Wind/聚宽/米筐）
    ├── local_data.py                # 本地数据自动识别加载
    ├── utils.py                     # 公共工具函数
    └── agents/
        ├── input_classifier.py      # 输入分类（idea/url/pdf/text）
        ├── paper_search_agent.py    # arXiv 论文检索 + LLM 语义排序
        ├── factor_extract_agent.py  # 从论文摘要提取因子假设
        ├── factor_implement_agent.py # LLM 生成因子代码 + 沙箱执行
        ├── factor_eval_agent.py     # IC/ICIR/衰减/覆盖率评估
        ├── factor_extension_agent.py # 生成因子延伸变体
        └── report_agent.py          # 生成 HTML 报告
```

---

## 数据源

默认使用 **AKShare**（免费，腾讯财经源）获取 CSI300 股票池日线数据。

支持接入 VIP 数据源：

```bash
# Tushare Pro
/source add tushare --token YOUR_TOKEN

# Wind（需本地安装 Wind 终端）
/source add wind

# 聚宽
/source add joinquant --username 手机号 --password 密码

# 米筐
/source add ricequant --username 账号 --password 密码

# 本地 CSV/Parquet 文件
/data ./my_factor_data.csv
```

---

## 输出

- **因子代码**：`./output/factors/<factor_name>_<timestamp>.py`
- **因子值 CSV**：`./output/factors/<factor_name>_<timestamp>.csv`
- **HTML 报告**：`./output/reports/factor_report_<timestamp>.html`
- **因子库**：`autoalpha.db`（SQLite）

---

## 技术栈

- **LLM**：Anthropic Claude / DeepSeek / OpenAI
- **数据**：AKShare（腾讯财经源）
- **IC 计算**：scipy.stats.spearmanr
- **代码沙箱**：subprocess 隔离执行（超时 120s）
- **存储**：SQLite
