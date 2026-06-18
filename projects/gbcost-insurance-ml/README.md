# GHB 团体保险赔付分析系统

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/LangGraph-1.1-green.svg)](https://github.com/langchain-ai/langgraph)
[![Tests](https://img.shields.io/badge/Tests-4475%20passing-brightgreen.svg)]()
[![Pyright](https://img.shields.io/badge/Pyright-0%20error-brightgreen.svg)]()

基于德勤14步下钻分析法的团体保险智能控费系统。双模式架构：规则引擎 + LLM增强（三提供商路由，模型见config.yaml）。独立ML预测子系统（LightGBM Tweedie）。Doris批量分析管线。TUI实时仪表盘。4475测试覆盖。v1.9.71。

## 功能特性

- **自动化分析**: 将2周人工分析缩短至10秒
- **分层异常检测**: 4级下钻（整体→责任→疾病→医院/会员）
- **双模式架构**: 规则引擎（Phase 1 编排器）+ LLM增强（Phase 2 LangGraph Agent）
- **LLM-Primary**: LLM分析为主，规则引擎验证为辅(4个Agent支持: anomaly/ibnr/health_score/cost_control)，Provider C故障转移
- **15个Prompt模板**: PT-001~PT-015覆盖分析/报告/OCR/交叉维度/预授权/FWA评估 + PT-016执行摘要增强
- **智能控费建议**: 基于异常模式自动生成量化控费措施
- **双模式对比**: `compare` 命令一键对比规则引擎 vs LLM分析结果
- **LangGraph工作流**: 36个Agent文件(36个接入workflow)，37节点StateGraph，fan-out/fan-in并行执行
- **OCR理赔文档分析**: 双引擎提取（规则+LLM），OCR-理赔关联匹配，独立调查报告
- **ETL智能管线**: 经过保费动态匹配（默认12个月）、医院名称跨年回填（81.5%覆盖率）、7项数据映射修复
- **TUI实时仪表盘**: Rich终端界面，实时Agent进度、Token统计、报告生成追踪。10组86项两级分组查询菜单 + 风险地图4维度23项 + emoji图标 + Esc返回上级 + 主菜单系统管理一级入口 + Doris数仓批量ETL(下载/扫描/提取/分析/列表)
- **量化控费指标**: HHI/CR4/CR10医院集中度、Gini系数/Lorenz曲线、Top30会员排名、同比/环比趋势
- **描述字段解析**: DescriptionFieldParser提取既往症/等待期/免赔额，per-claim结构化存储
- **完整测试覆盖**: 4475个测试用例，覆盖核心分析逻辑
- **FWA规则引擎**: 61条反欺诈/浪费/滥用规则(22欺诈+18浪费+21滥用)，多维索引加速，加法风险评分
- **DRG/DIP分组**: CHS-DRG 2.0标准，ICD-10精确+前缀匹配，地区/医院基准费用对比
- **DEG诊断分组**: ICD-10三维度Episode分组（解剖系统/病理类型/严重度）
- **NDC病程校验**: 8种慢病治疗约束，时长/频率/剂量合理性检查
- **IBNR精算分段**: 5年龄段+5疾病类别+4 LOB分段精算，95%置信区间
- **EDMP早期疾病管理**: 4种慢病识别，3因素风险评估，5干预建议模板
- **交互式仪表盘**: Chart.js钻取/数据过滤/表格排序/CSV导出
- **费用明细分项**: 7项费用结构分析(门诊/住院/药品/检查/手术/材料/其他)，分类占比+趋势+异常标记
- **标杆匹配画像**: 12维度相似度匹配(8基础+4扩展)，25个标杆保单基准，定制化控费建议
- **OCR闭环质量**: 置信度评估+人工复核标记+质量分级，OCR提取结果自动校验
- **医院库扩展**: 52家医院费用标准(18->52，覆盖20+城市)，模糊匹配+分级+区域查询
- **控费追踪闭环**: 措施生命周期管理(PLANNED->COMPLETED)，效果监控+统计检验(t-test)+告警服务(逾期/效果未达预期/异常回升)
- **知识库管理**: 独立更新工具，支持从互联网搜索真实数据源
- **总结与控费建议**: 规则+LLM双模式AI分析文字，执行摘要报告顶部展示，历史同比贯穿保单/责任/疾病层
- **历史同比分析**: ResponsibilityData/DiseaseData YoY对比，Excel责任/疾病Sheet历史列，TUI同比对比表
- **7类报告+三格式导出**: 分析报告 + OCR调查报告 + 控费报告 + 双报告对比 + Excel(42 Sheet含总结) + Chart.js交互式HTML仪表盘 + 综合可视化HTML报告(32章节,TOC导航+品牌横幅)
- **pip可安装**: 支持 `pip install .` 或 `.whl` 分发，提供 `ghb-cost-control` 全局命令
- **Doris数仓集成**: MySQL协议直连Doris FE(9030端口)，TUI一站式批量ETL(下载→扫描→提取→分析→报告)，4线程并行分析，24h扫描缓存
- **医疗合理性引擎**: ICD-10×4类规则(不合理用药/过度检查/不合理住院/诊断治疗不匹配)，规则引擎无LLM依赖
- **医院收费标准异常**: 同等级医院费用对比+增长趋势检测，cost_standards.yaml 52家医院5级分类(公立特需/公立国际部/公立普通)
- **24月走势预测**: 线性回归+季节分解，3源fallback(ramp_trajectory/ramp_analysis/claims) + loss_ratio自动计算
- **风险地图**: 4维度(疾病/医院/会员/投保单位)×23项交叉视图，TUI G10菜单+综合报表32章节全覆盖
- **职业疾病不实告知**: 12职业组×25规则匹配，欺诈深度调查建议(4风险点×4路径)
- **ML理赔预测**: 独立LightGBM子系统，L1-A案件级金额预测(Tweedie回归)，验证集R²=0.91/Gini=0.75，TUI G11菜单+综合报表Ch33集成

## 快速开始

### 方式一：pip安装（推荐）

```bash
# 1. 创建虚拟环境
python -m venv .venv
.venv\Scripts\activate              # Windows
# source .venv/bin/activate         # Linux/Mac

# 2a. 从源码安装（开发模式）
git clone <repo-url> && cd gbcost-analysys
pip install -e .

# 2b. 或从 whl 文件安装（部署模式）
pip install ghb_cost_control-1.5.1-py3-none-any.whl

# 3. 准备工作目录 (whl不包含数据文件, 需从源码仓库复制)
mkdir workdir && cd workdir
xcopy /E /I <源码路径>\knowledge knowledge    # Windows
# cp -r <源码路径>/knowledge knowledge        # Linux/Mac
copy <源码路径>\config.yaml .                  # 配置文件
copy <源码路径>\.env .                         # API密钥(可选, 不配则LLM降级)
xcopy /E /I <源码路径>\data data              # 理赔数据(按需)

# 4. 运行 (从包含 knowledge/ 和 config.yaml 的目录启动)
ghb-cost-control tui                # TUI交互模式（推荐）
```

安装后自动获得 `ghb-cost-control` 全局命令。whl部署模式下，需从包含 `knowledge/`、`config.yaml` 的工作目录运行，系统会自动定位数据文件。

### 方式二：源码运行

```bash
# 安装依赖
pip install -r requirements.txt

# TUI交互模式（推荐）
python -m src.cli tui

# 运行分析（Phase 1 轻量编排，14步顺序分析）
python -m src.orchestrator --policy deloitte

# 运行分析（Phase 2 LangGraph Agent，21个Agent并行）
python -m src.cli analyze --policy deloitte --agent

# 运行LLM增强分析（需配置API Key）
python -m src.cli analyze --policy deloitte --agent --verbose

# 双模式对比（规则引擎 vs LLM）
python -m src.cli compare --policy deloitte --verbose

# ETL管线
python -m src.cli etl --verbose

# 运行测试
python -m pytest tests/ -v
```

### 方式三：Docker运行

```bash
# 构建镜像
docker compose build

# 运行分析
docker compose up

# 查看报告
cat output/analysis_report_*.md
```

## 命令行接口

8个子命令: analyze / validate / test / compare / etl / tui / logs / download。独立ML预测管线: `python -m ml.pipeline.train / predict / evaluate`。

```bash
# 查看帮助
ghb-cost-control --help
ghb-cost-control --version

# TUI交互模式（Phase 1/2/Compare三模式）
ghb-cost-control tui
ghb-cost-control tui --skip-input              # 跳过参数输入

# Phase 1 分析（规则引擎）
ghb-cost-control analyze --policy deloitte

# Phase 2 分析（Agent模式）
ghb-cost-control analyze --policy deloitte --agent --verbose

# 双模式对比
ghb-cost-control compare --policy deloitte --verbose

# ETL管线
ghb-cost-control etl --verbose

# 验证数据
ghb-cost-control validate --policy deloitte

# 运行测试
ghb-cost-control test

# 知识库管理（源码模式下可用）
python -m src.tools.knowledge_base_updater status        # 查看状态
python -m src.tools.knowledge_base_updater search        # 搜索数据源
python -m src.tools.knowledge_base_updater report        # 生成版本报告
python -m src.tools.knowledge_base_updater full          # 完整流程

# 从Doris数仓下载CSV
python -m src.cli download --password XXX                # 下载
python -m src.cli download --test --password XXX         # 测试连接

# 执行日志查询
python -m src.cli logs                                  # 最近日志
python -m src.cli logs --date 2026-05-20                # 按日期查询
python -m src.cli logs --policy deloitte --limit 10      # 按保单查询

# ML预测管线（独立子系统）
python -m ml.pipeline.train --config ml/config_ml.yaml    # 训练
python -m ml.pipeline.predict --csv data/doris/c001...csv  # 预测
python -m ml.pipeline.evaluate --output reports/ml/       # 评估报告
python ml/pipeline/run_backtest.py                        # 回测
```

> **注**: 所有 `ghb-cost-control` 命令均可使用 `python -m src.cli` 等效替代。

## Doris 批量分析管线 (v1.9.70+)

完整自动化的保险批量分析：扫描保单 → Phase 2 Agent 分析 → 生成 agent_state + 综合报告 → 风险地图 Q&A 报告。

### 快速开始

```bash
# 1. 查看可用保单
ls data/doris/policies/wide_*.csv | sed 's/.*wide_//;s/\.csv//'

# 2. 单张保单完整分析（分析 + Q&A 提示词）
python skills/ghb-cost-control/scripts/batch_doris_pipeline.py GP2202301028502 --qa --llm-mode llm_primary

# 3. 多张保单批量分析
python skills/ghb-cost-control/scripts/batch_doris_pipeline.py PID1 PID2 PID3 --qa --llm-mode llm_primary

# 4. 仅对已完成保单生成 Q&A（跳过 Phase 2 分析）
python skills/ghb-cost-control/scripts/batch_doris_pipeline.py PID1 PID2 --qa-only

# 5. 增量运行（跳过已有 agent_state 的保单）
python skills/ghb-cost-control/scripts/batch_doris_pipeline.py PID1 PID2 PID3 --skip-existing --qa --llm-mode llm_primary
```

### 脚本参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `policies` | (必填) | 一个或多个保单 ID |
| `--qa` | false | 分析完成后保存 Q&A 委托提示词到输出目录，后续用 writing 子代理生成报告 |
| `--qa-only` | false | 跳过 Phase 2 分析，仅对已有 agent_state.json 生成 Q&A 提示词 |
| `--skip-existing` | false | 跳过已存在 agent_state.json 的保单目录 |
| `--llm-mode` | `llm_primary` | LLM 模式: `llm_primary`(LLM增强+综合HTML) / `rule_only`(纯规则) / `hybrid` |
| `--timeout` | 900 | 每保单分析超时秒数（大保单用 1200） |
| `--doris-dir` | `data/doris/policies` | Doris 宽表 CSV 目录 |
| `--output-dir` | `output` | 输出根目录 |

### 完整管线流程

```
┌─────────────────────────────────────────────────────────┐
│ Step 1: batch_doris_pipeline.py PID1 PID2 --qa          │
│                                                         │
│  ├─ Phase 2 Agent (37节点, ~7min/保单)                  │
│  │   ├─ agent_state.json (~8MB, 114 keys)               │
│  │   ├─ comprehensive_report_*.html (~250KB, 32章节)    │
│  │   ├─ analysis_report_*.md + cost_control_report_*.md │
│  │   └─ llm_analysis_report_*.txt                       │
│  │                                                      │
│  └─ Q&A 提示词保存 → qa_prompt_{pid}.txt                │
│                                                         │
│ Step 2: writing 子代理 (DeepSeek V4 Pro, ~12min)        │
│  │                                                      │
│  └─ task(category="writing",                            │
│          load_skills=["risk-map-analyzer"],              │
│          prompt=open("qa_prompt_{pid}.txt").read())      │
│       → risk_map_qa_{pid}.md (~76KB, 32问答)            │
└─────────────────────────────────────────────────────────┘
```

### Q&A 委托调用

```python
# 脚本会输出完整的子代理调用代码，直接复制使用：
task(
    category="writing",                    # DeepSeek V4 Pro
    load_skills=["risk-map-analyzer"],      # v2.3.0 技能
    prompt=open("output/{pid}_*/qa_prompt_{pid}.txt", encoding="utf-8").read()
)
# → 产出: risk_map_qa_{pid}.md (~76KB, 4维度×18场景×32问答)
```

### LLM 模式选择

| 模式 | Token | 速度 | 适用场景 |
|------|-------|------|----------|
| `llm_primary` | ~121K | ~7min | **批量分析推荐** — PT-001~016 全部 LLM 增强 |
| `rule_only` | 0 | ~5min | 快速数据质量验证 |
| `hybrid` | ~50K | ~6min | 规则为主 + 部分 LLM 补充 |

### 性能数据 (10 保单实测)

| 指标 | 值 |
|------|-----|
| 总保单 | 10 |
| 总理赔 | 10,058 件 |
| agent_state 总计 | 80MB |
| Q&A 报告总计 | 763KB |
| 每保单分析 | ~7min |
| 每保单 Q&A | ~12min |
| 总耗时 | ~3.5h |
| 成功率 | 10/10 (0 错误) |
| 脚本位置 | `skills/ghb-cost-control/scripts/batch_doris_pipeline.py` |

### 输出目录结构

```
output/GP2202301028502_20260609_185651/
├── agent_state.json                    # Phase 2 完整状态 (114 keys)
├── analysis_report_*.md                # 主分析报告 (40+节)
├── analysis_report_*.html              # 主分析 HTML
├── comprehensive_report_*.html         # 综合可视化 HTML (32章节)
├── cost_control_report_*.md            # 控费报告 (13章)
├── ocr_investigation_*.md              # OCR 调查报告
├── report_comparison_*.md              # 双报告对比
├── llm_analysis_report_*.txt           # LLM 增强分析
├── qa_prompt_GP2202301028502.txt       # Q&A 委托提示词
└── risk_map_qa_GP2202301028502.md      # 风险地图 Q&A 报告 (32问答)
```

## TUI交互模式

TUI（Terminal User Interface）是系统的推荐交互方式，提供从数据加载到分析执行的完整工作流。

### 启动

```bash
ghb-cost-control tui                # 完整交互（参数输入+分析+查询）
ghb-cost-control tui --skip-input   # 跳过参数输入，使用默认值
python -m src.cli tui               # 等效命令
```

### 工作流程

TUI包含四个阶段：

| 阶段 | 说明 |
|------|------|
| **参数收集** | 选择分析模式（Phase 1规则引擎 / Phase 2 Agent / 双模式对比）、保单数据文件、输出目录 |
| **分析执行** | Rich Live面板实时显示Agent进度、耗时、Token统计、报告生成状态 |
| **结果查询** | 10组86项两级分组菜单，选择类别→具体项→查看分析结果 |
| **报告导出** | 自动生成7类报告（Markdown + Excel + HTML三格式） |

### 查询菜单（10组86项）

分析完成后进入交互式查询菜单，按**类别→具体项**两级选择：

| # | 类别 | 项数 | 包含内容 |
|---|------|------|----------|
| 1 | **核心摘要** | 6 | 总结与控费建议、执行摘要、关键指标总览、评分结果、报告文件列表、Agent执行详情 |
| 2 | **保单与数据** | 8 | 数据对账、前提条件检查、保单属性对比、投保单位分析、出险城市分析、月度爬坡轨迹、大额理赔分析、核保LR对比 |
| 3 | **异常检测与下钻** | 6 | 4级分层异常检测、异常等级分级概览、责任同比对比、环比异常分析、路径A/B下钻分析 |
| 4 | **疾病与费用** | 9 | 疾病-治疗匹配、ICD分析明细、DRG/DIP分组、DEG诊断分组、费用明细分项、费用细项异常、EDMP早期管理、NDC病程校验、医疗合理性审核 |
| 5 | **医院分析** | 6 | 集中度分析(HHI/CR4)、类型分布、多维分析、Top医院费用、集中就医分析、收费标准异常检测 |
| 6 | **会员与画像** | 5 | 会员画像、Pareto分析、Top理赔分析、高频会员行为画像、会员风险评分 |
| 7 | **控费与管控** | 10 | 控费建议明细、归因分析、赔付率因素分解、控费策略匹配、控费追踪闭环、预授权分析、LLM增强明细、交叉维度分析、预警监控、统一预警面板 |
| 8 | **安全与调查** | 7 | FWA反欺诈、LLM FWA评估、治疗与用药审核、案件关联分析、家庭关联模式、同日聚集分析、案件调查 |
| 9 | **预测与标杆** | 6 | IBNR预测、标杆匹配、既往症分析、健康评分、OCR-理赔关联、24月走势预测 |
| 10 | **🗺️ 风险地图** | 23 | 疾病维度(6)/医院维度(6)/会员维度(6)/投保单位维度(5)四维度交叉视图 |
| 11 | **🤖 ML预测** | 3 | 预测汇总概览(R²/Gini/MAE KPI) + 保单级预测对比(预测vs实际分桶) + 模型评估详情(特征重要性/训练结果/回测) |

> **注**: 系统管理（版本信息、README/AGENTS.md查看、执行日志查询、批量分析历史等）已提升到**主菜单一级入口**，无需进入查询菜单即可访问。

### 快捷操作

- **重新加载数据**: 从查询菜单选择"重新加载数据"可切换分析结果文件
- **批量分析**: 支持从费用明细CSV加载42保单，逐单或批量分析
- **文档阅读**: 系统管理组内置README和AGENTS.md的分页Markdown渲染（60行/页，支持翻页跳转）

```
src/
├── agents/                    # Phase 2: LangGraph Agent (36个Agent文件)
│   ├── base.py               # BaseAgent抽象基类（LLM/规则双路由）
│   ├── state.py              # AgentState状态定义 (122字段)
│   ├── workflow.py           # LangGraph StateGraph（37节点, fan-out/fan-in并行）
│   ├── orchestrator_agent.py # 高级编排API + load_ocr_data()
│   └── *_agent.py            # 36个Agent文件(含medical_rationality/hospital_fee_anomaly/trend_forecast)
│
├── analysis/                  # 核心分析模块（36个模块）
│   ├── precondition_checker.py    # Step 2: 前提条件检查
│   ├── hierarchical_anomaly_detector.py  # Step 3: 分层异常检测
│   ├── ibnr_predictor.py     # Step 4: IBNR预测
│   ├── claim_correlation_analyzer.py  # Step 6: 案件关联分析
│   ├── health_score.py       # Step 11: 健康评分
│   ├── member_profiler.py    # Step 12: 会员画像
│   ├── medical_rationality.py    # 医疗合理性引擎(ICD-10×4类规则)
│   ├── hospital_fee_anomaly.py   # 医院收费标准异常(同等级对比+增长趋势)
│   ├── trend_forecaster.py       # 24月走势预测(线性回归+季节分解)
│   ├── ramp_analyzer.py      # 趋势分析
│   ├── multidimensional_metrics.py  # 多维指标计算
│   ├── fee_rootcause_analyzer.py    # 费用根因分析
│   ├── policy_unit_analyzer.py      # 投保单位维度分析
│   ├── benefit_transfer_analyzer.py # 利益输送检测
│   ├── drug_dosage_checker.py       # 药品剂量检查
│   └── coverage_deviation_analyzer.py # 保障偏差分析
│
├── knowledge/                 # 知识库（17个YAML + 3个匹配器）
│   ├── hospital_costs.py     # 医院费用标准（52家，覆盖20+城市）
│   ├── benchmark_matcher.py  # Step 5: 标杆匹配（25个标杆保单，12维度）
│   └── disease_treatment_matcher.py  # Step 9: 疾病-治疗匹配（44条规则）
│
├── llm/                       # LLM集成（三提供商路由）
│   ├── client.py             # LLMClient（三Provider路由，模型见config.yaml）
│   ├── prompt_manager.py     # 提示模板管理
│   └── prompts.py            # 15个Prompt模板 (PT-001 ~ PT-015)
│
├── recommendations/           # 控费建议
│   └── cost_control_generator.py  # Step 7: 控费建议生成（13项措施）
│
├── report/                    # 报告生成（7类报告 + 三格式导出）
│   ├── report_generator.py   # Markdown报告（含OCR-理赔关联章节）
│   ├── ocr_investigation_report.py # OCR案件调查独立报告
│   ├── cost_control_report.py # 控费独立报告（13章）
│   ├── comprehensive_report_generator.py # 综合可视化HTML报告（32章节, ~6220行）
│   ├── excel_chart_html_generator.py # Chart.js交互式HTML仪表盘
│   ├── report_comparison.py  # 双报告结构/覆盖度/指标对比
│   ├── comparison_report.py  # Phase 1 vs Phase 2对比
│   ├── hybrid_comparison.py  # LLM vs 规则引擎6维对比
│   ├── llm_analysis_report.py # LLM增强分析报告
│   ├── execution_summary.py  # Agent执行摘要
│   └── sections/             # 36个SectionReporter
│
├── models/                    # 统一数据模型
│   └── policy_data.py        # PolicyAnalysisData (6个dataclass)
│
├── data_sources/              # 数据源解析 + ETL管线
│   ├── etl_pipeline.py        # ETL管线（2715行，经过保费动态化+医院名称回填）
│   ├── underwriting_excel_parser.py # Excel理赔数据解析
│   ├── ocr_loader.py         # OCR文档加载（documents.db + extractions缓存）
│   ├── ocr_extractor.py      # 规则引擎字段提取（6种文档类型）
│   ├── ocr_llm_extractor.py  # DeepSeek LLM字段提取（PT-012）
│   ├── ocr_mapper.py         # OCR→ClaimData映射
│   └── ocr_claim_correlator.py # OCR-理赔关联匹配引擎
│
├── tui/                       # TUI实时仪表盘（6文件）
│   ├── app.py                # 主控: Phase 1/2/Compare执行+10组86项查询+风险地图23项
│   ├── buffer.py             # Agent状态/耗时/token/报告路径缓冲(29 Agent, 5阶段)
│   ├── callback.py           # LLM token统计回调
│   ├── layout.py             # Rich四区域Layout
│   └── params.py             # questionary参数收集
│
├── orchestrator.py           # Phase 1: 14步顺序编排器
├── cli.py                    # CLI入口 (8子命令: analyze/validate/test/compare/etl/tui/logs/download)
└── config.py                 # YAML配置加载器（单例）

ml/                           # 独立ML预测子系统
├── data/loader.py            # Polars惰性加载 (5.84GB CSV)
├── data/feature_store.py     # 两阶段特征工程 (41特征11分类)
├── data/split.py             # 时间序列安全划分
├── models/amount_predictor.py # LightGBM Tweedie + 分位数模型
├── evaluate/metrics.py       # 保险行业评估指标 (Gini/MAPE/总量误差)
├── evaluate/backtest.py      # Walk-Forward时序回测
├── pipeline/train.py         # CLI训练入口
├── pipeline/predict.py       # CLI预测入口
├── pipeline/evaluate.py      # CLI评估入口
├── pipeline/run_backtest.py  # 独立回测脚本
├── report/generator.py       # 中文执行报告生成器
└── config_ml.yaml            # ML独立配置

knowledge/                     # 知识库YAML文件（20个）
├── hospitals/cost_standards.yaml   # 医院费用标准(52家, 5级分类, v1.1.0)
├── benchmarks/policy_benchmarks.yaml # 标杆保单 (v1.1.0, 7个数据源)
├── disease_treatment/disease_treatment_rules.yaml # 疾病规则 (v1.1.0)
├── fwa_rules.yaml                  # FWA反欺诈规则(61条: 22欺诈+18浪费+21滥用)
├── benefit_transfer_rules.yaml     # 利益输送检测规则
├── drug_dosage_rules.yaml          # 药品剂量规则(50+药品)
├── clinical_guidelines/            # Cigna临床指南(10组疾病)
├── medical_rationality/             # 医疗合理性规则
├── icd10_disease_names.yaml         # ICD-10中文疾病名称(1,689条+别名)
├── drg_groups.yaml / drg_benchmarks.yaml  # DRG/DIP分组
├── deg_groups.yaml                  # DEG诊断分组
├── ndc_rules.yaml                   # NDC病程校验规则
├── edmp_rules.yaml                  # EDMP早期管理规则
├── ibnr_segments.yaml               # IBNR精算分段
├── cost_control_measures.yaml       # 控费措施
├── cost_control_tracking_rules.yaml # 控费追踪规则
├── fee_structure_rules.yaml         # 费用结构规则
├── responsibility_codes.yaml        # 责任编码
└── occupation_disease_rules.yaml    # 职业疾病匹配规则
```

## Phase 1: 14步顺序编排

| 步骤 | 模块 | 功能 |
|------|------|------|
| 1 | ReconciliationAgent | 数据对账 |
| 2 | StabilityAgent | 前提条件检查（会员/保费稳定性） |
| 3 | AnomalyDetectionAgent | 4级分层异常检测（L1-L3） |
| 4 | IBNRPredictionAgent | IBNR预测 |
| 5 | BenchmarkMatchingAgent | 标杆匹配 |
| 6 | ClaimCorrelationAgent | 案件关联分析（家庭/雇主） |
| 7 | CostControlAgent | 控费建议生成 |
| 8 | PriorConditionAgent | 既往症分析 |
| 9 | DiseaseTreatmentAgent | 疾病-治疗匹配 |
| 10 | HospitalCostAgent | 医院费用标准检查 |
| 11 | HealthScoreCalculator | 健康评分 |
| 12 | MemberProfilerAgent | 会员画像 |
| 13 | SupplementAnalysisAgent | 补充分析 |
| 14 | LLM增强（DeepSeek） | 可选LLM增强分析 |

## Phase 2: LangGraph Agent工作流

30个Agent文件通过LangGraph StateGraph(31节点)编排，29个接入workflow，fan-out/fan-in并行执行。

### Agent分类

| 类型 | Agent | 说明 |
|------|-------|------|
| 纯规则 | ReconciliationAgent | 数据对账 |
| 纯规则 | StabilityAgent | 稳定性检查 |
| 纯规则 | BenchmarkMatchingAgent | 标杆匹配 |
| 纯规则 | RampAnalysisAgent | 趋势分析 |
| 纯规则 | MemberProfilerAgent | 会员画像（含Gini系数/Lorenz曲线） |
| 纯规则 | ExtendedMetricsAgent | 扩展指标（出险率/占比/三色预警/爬坡/聚集/大额/Top30排名/同比环比） |
| 纯规则 | DRGAgent | DRG/DIP分组分析（CHS-DRG 2.0，50个DRG组） |
| 纯规则 | DEGAgent | DEG诊断分组（ICD-10三维度Episode分组统计） |
| 纯规则 | EDMPAgent | EDMP早期疾病管理（慢病识别+风险评估+干预建议） |
| 纯规则 | FeeStructureAgent | 费用明细分项分析（7项分类占比+趋势+异常） |
| 纯规则 | CostControlTrackingAgent | 控费追踪闭环（生命周期+效果监控+告警） |
| 纯规则 | PathADrilldownAgent | 路径A下钻分析（4层嵌套: 责任→疾病→医院→二级责任） |
| 纯规则 | MedicalRationalityAgent | 医疗合理性审核（ICD-10×4类规则） |
| 纯规则 | HospitalFeeAnomalyAgent | 医院收费标准异常检测（同等级对比+增长趋势） |
| 纯规则 | TrendForecastAgent | 24月走势预测（线性回归+季节分解, 3源fallback） |
| 纯规则 | NDCAgent | NDC病程校验（8种慢病治疗约束） |
| 纯规则 | MultidimensionalMetricsAgent | 多维指标计算 |
| 纯规则 | FeeRootcauseAgent | 费用根因分析 |
| 纯规则 | PolicyUnitAgent | 投保单位维度分析(含同比+同级对比+真实性聚合) |
| 纯规则 | BenefitTransferAgent | 利益输送检测(医院Jaccard/会员迁移/单位集中漏斗) |
| 纯规则 | DrugDosageAgent | 药品剂量合理性检查 |
| 纯规则 | CoverageDeviationAgent | 保障偏差分析 |
| LLM增强 | AnomalyDetectionAgent | 异常检测（HIGH/CRITICAL时LLM增强） |
| LLM增强 | IBNRPredictionAgent | IBNR预测 |
| LLM增强 | ClaimCorrelationAgent | 案件关联分析 |
| LLM增强 | PriorConditionAgent | 既往症分析 |
| LLM增强 | DiseaseTreatmentAgent | 疾病-治疗匹配 |
| LLM增强 | HospitalCostAgent | 医院费用检查（含HHI/CR4/CR10集中度） |
| LLM增强 | HealthScoreAgent | 健康评分 |
| LLM增强 | CaseInvestigationAgent | 案件调查 |
| LLM增强 | CostControlAgent | 控费建议 |
| LLM增强 | CrossDimensionAgent | 交叉维度分析（PT-013） |
| LLM增强 | PreAuthorizationAgent | 预授权分析（PT-014） |
| LLM增强 | FWAAgent | FWA分析（61条规则 + PT-015 LLM增强） |
| LLM生成 | ReportGenerationAgent | 报告生成（PT-006） |

### 并行执行

- **并行组A** (5个Agent): benchmark_matching, claim_correlation, prior_condition, disease_treatment, hospital_cost
- **并行组B** (24个Agent): case_investigation, health_score, ramp_analysis, member_profiler, cross_dimension, extended_metrics, pre_authorization, fwa_analysis, drg_analysis, deg_analysis, edmp_analysis, ndc_analysis, fee_structure_analysis, cost_control_tracking, path_a_drilldown, medical_rationality, hospital_fee_anomaly, trend_forecast, multidimensional_metrics, fee_rootcause, drug_dosage, coverage_deviation, benefit_transfer
- **条件路由**: L1异常检测后决定是否进入深度分析

### LLM提供商路由

| Provider | 用途 | 说明 |
|----------|------|------|
| Provider A | PT-006 报告生成 | 具体模型见 config.yaml provider_a.model |
| Provider B | PT-001,003-005,007-015 分析任务 + OCR提取 | 具体模型见 config.yaml provider_b.model |
| Provider C | 备用 | 具体模型见 config.yaml provider_c.model |

## ETL数据管线

从原始Excel理赔文件到标准化PolicyAnalysisData的智能ETL管线。

```bash
# 运行ETL管线
ghb-cost-control etl --verbose

# 指定参数
ghb-cost-control etl --source-dir data/data --output-dir data/etl --policy-id DELOITTE-2025
```

### ETL特性

| 特性 | 说明 |
|------|------|
| **经过保费动态匹配** | 从LR报表动态读取经过月数，上年自动匹配同期，默认12个月 |
| **医院名称跨年回填** | 当年文件缺医院名称时，从上年理赔数据回填（81.5%覆盖率） |
| **会员提取修正** | 从客户信息清单sheet提取（1102人），非理赔行提取 |
| **7项映射修复** | 状态/LOB枚举、经过月数、member_count、ClaimData+13字段 |
| **字段映射报告** | 每次ETL输出52个规范字段的映射状态 |

### ETL输出

| 文件 | 说明 |
|------|------|
| `policy_data.json` | 完整标准化数据（~60MB） |
| `policy_data.xlsx` | 7个Sheet（理赔/会员/医院/疾病/月度） |
| `claims_current.xlsx` | 当年理赔（26,363条 x 31字段） |
| `claims_previous.xlsx` | 上年理赔（22,275条 x 31字段） |
| `member_demographics.xlsx` | 会员人口统计（1,102条） |
| `etl_report.json` | ETL执行摘要 |

## 双模式对比

```bash
# 一键对比规则引擎 vs LLM分析
ghb-cost-control compare --policy deloitte --verbose
```

对比维度（6维）:
1. 异常检测结果对比
2. IBNR预测对比
3. 案件关联对比
4. 控费建议对比
5. 既往症分析对比
6. 疾病-治疗匹配对比

## OCR理赔文档分析

系统支持从OCR识别的理赔文档中提取结构化字段，并与Excel理赔数据进行关联匹配。

### 数据流

```
data/documents.db (1,074份OCR文档, 125个理赔案)
    → OcrDataLoader (加载+extractions缓存)
    → OcrFieldExtractor (规则引擎提取, 6种文档类型)
    → OcrLlmExtractor (DeepSeek PT-012 LLM增强提取)
    → OcrClaimCorrelator (OCR-理赔关联匹配)
    → 主报告 + OCR独立调查报告
```

### 关联匹配策略

| 策略 | 条件 | 置信度 |
|------|------|--------|
| 精确匹配 | OCR案号=理赔ID | 1.00 |
| 模糊匹配(三元组) | 医院+金额(±20%)+日期(±3天) | 0.80 |
| 模糊匹配(二元组) | 医院+金额(±20%) | 0.50 |
| 未匹配 | 以上均未命中 | 0.00 |

### 报告输出

- **主报告**: `analysis_report_{policy_id}_{timestamp}.md` — 含14个分析章节（含"OCR理赔文档与案件关联分析"）
- **OCR独立报告**: `ocr_investigation_{policy_id}_{timestamp}.md` — 提取明细+交叉验证+欺诈信号+关联信息
- **控费报告**: `cost_control_report_{policy_id}_{timestamp}.md` — 12章通用控费报告
- **双报告对比**: `report_comparison_{policy_id}_{timestamp}.md` — 分析报告与控费报告的结构/覆盖度/指标对比

## ML理赔预测子系统 (v1.9.71+)

独立LightGBM预测引擎，与现有规则引擎通过**文件系统松耦合集成**：ML子系统输出 JSON/Parquet，主系统 TUI/综合报表按需读取。

### 双能力定位

| 维度 | 现有系统（因子分析） | ML预测子系统（趋势预测） |
|------|---------------------|------------------------|
| 分析方向 | **向后看** — 已发生的理赔 | **向前看** — 预测未来金额 |
| 核心能力 | 14步德勤下钻发现异常因子 | LightGBM Tweedie回归预测案件金额 |
| 典型问题 | "为什么这笔赔高了？" | "下个月预计赔付多少？" |
| 数据依赖 | 当期理赔明细 | 历史605万条理赔记录 |
| 输出形式 | 异常标记/控费建议/报告 | 预测金额/分位数/分桶统计 |
| 交互入口 | TUI G02-G08 查询组 | TUI G11 "🤖 ML预测" + 综合报表Ch33 |

### 架构

```
┌─ 数据接口层 ─────────────────────────────────────────────┐
│  ml/data/      Polars惰性加载 → 全局统计 → 两阶段特征工程  │
├─ 模型层 ─────────────────────────────────────────────────┤
│  ml/models/    L1-A金额预测(LightGBM Tweedie + 分位数)    │
├─ 管线层 ─────────────────────────────────────────────────┤
│  ml/pipeline/  train.py / predict.py / evaluate.py       │
├─ 输出层 ─────────────────────────────────────────────────┤
│  models/       模型文件(.txt) + 特征Schema + 类别映射      │
│  predictions/  案件预测Parquet + 保单汇总JSON              │
│  reports/ml/   中文执行报告(MD + JSON) + 回测结果          │
├─ 集成层 ─────────────────────────────────────────────────┤
│  src/tui/      G11 "🤖 ML预测" (3项查询)                  │
│  src/report/   Ch33 "ML理赔预测" (综合HTML报表)           │
└──────────────────────────────────────────────────────────┘
```

### 快速开始

```bash
# 1. 训练模型 (5.84GB Doris CSV, ~15秒)
python -m ml.pipeline.train --config ml/config_ml.yaml --mode full

# 2. 批量预测 (全量605万行 → 67MB Parquet)
python -m ml.pipeline.predict \
    --csv data/doris/c001_ghb_poicy_clm_duty_d.csv \
    --policy ALL

# 3. 单张保单预测
python -m ml.pipeline.predict \
    --csv data/doris/c001_ghb_poicy_clm_duty_d.csv \
    --policy GP2202301028502

# 4. 生成中文评估报告
python -m ml.pipeline.evaluate --output reports/ml/

# 5. Walk-Forward回测验证
python ml/pipeline/run_backtest.py
```

### 当前模型指标 (L1-A, v1.9.71 P1: tweedie=1.2 + 季度回测)

| 指标 | 训练验证集 | 全量预测(579万行) | 说明 |
|------|:---:|:---:|------|
| R² | 0.5654 | **0.6467** | 全量预测稳定 |
| Gini | 0.7428 | **0.6882** | 排序能力强 |
| MAE | 3,825元 | **2,994元** | 较基线改善43% |
| 总量误差 | 70.2% | 49.6% | 系统性低估改善中 |
| large_recall | 0.240 | **0.424** | 大额识别有效 |
| 回测 avg Gini | — | **0.723 (季度)** | 4轮稳定 |
| 类别映射 | 11列 / 1042类 | ✅ | 持久化 |
| 训练耗时 | 65.3秒 | best_iter=128 | 232万训练行 |

> P1 季度回测: Gini 0.75/0.69/0.74/0.72 (稳定在0.72±3%)。tweedie_power 1.5→1.2 后 val_l1 改善4%。

### 后续路线图

| 优先级 | 任务 | 状态 |
|:---:|------|:---:|
| P0 | ~~修复类别编码映射~~ | ✅ 已完成 |
| P1 | ~~季度回测 + Tweedie调优~~ | ✅ 已完成 |
| P2 | Optuna超参调优 + 特征交互工程 | ⬜ 计划中 |
| P3 | L1-B大额二分类 + L2保单级预测 | ⬜ 计划中 |

### 使用指南

**日常流程** (每月数据更新后):
```bash
python -m ml.pipeline.train --config ml/config_ml.yaml --mode full     # 重训练 (~2分钟)
python -m ml.pipeline.predict --csv data/doris/c001...csv --policy ALL # 全量预测 (~3分钟)
```

**解读预测**:
| 需求 | 指标 | 位置 |
|------|------|------|
| 案件风险排序 | `predicted_amount` 降序 (Gini=0.68) | `.parquet` |
| 保单下月预估 | `total_pred × 1.80` (校正系数) | `_summary.json` |
| 大额案件筛查 | `predicted_amount > p95分位数` (召回66%) | `.parquet` |
| 模型健康检查 | R²>0.5 & 回测Gini>0.7 | `execution_report.md` |

**聚合校正**: 总量预测偏低约45%，业务使用时建议 `校正总额 = 预测总额 × 1.80`

📖 完整分析报告: `docs/ml_analysis_report_v1.md` — 含专业术语详解、全阶段指标进化、参数解读、业务应用指南

**能力边界**:
- ✅ 排序能力强 (Gini 0.69)，适合风险排序和保单对比
- ✅ 回测稳定 (季度 Gini 0.72±3%)
- ⚠️ 个案金额预测有偏差 (MAPE 82%)，不宜作为精确理赔金额
- ⚠️ 对新上线保单预测能力有限 (依赖历史TE)

详细评估见 `docs/claim_prediction_plan_v3.md` §15。

## 闭环分析模型 — 方法论概述

系统采用**"因子分析→趋势预测→综合决策"**三阶段闭环模型，将向后看的异常发现与向前看的趋势预判整合为完整的控费决策链。

### 模型总览

```
┌─────────────────────────────────────────────────────────────────────┐
│                      闭环分析模型 (Closed-Loop)                      │
│                                                                     │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────┐  │
│  │ 阶段一       │    │ 阶段二       │    │ 阶段三               │  │
│  │ 因子分析     │───▶│ 趋势预测     │───▶│ 综合决策             │  │
│  │ (向后看)    │    │ (向前看)    │    │ (交叉验证+策略)      │  │
│  │             │    │             │    │                      │  │
│  │ 14步德勤下钻 │    │ LightGBM    │    │ 异常×预测交叉矩阵    │  │
│  │ FWA 61条规则 │    │ Tweedie回归 │    │ 风险评分融合         │  │
│  │ 4级分层异常  │    │ 分位数估计  │    │ 控费策略优先级排序   │  │
│  │ 36 Agent并行 │    │ 季度回测    │    │ 闭环追踪反馈         │  │
│  └──────┬───────┘    └──────┬───────┘    └──────────┬───────────┘  │
│         │                   │                       │              │
│         ▼                   ▼                       ▼              │
│  "为什么赔高了？"     "未来会赔多少？"      "如何优化？"          │
│  异常因子清单         预测金额分布           优先级策略矩阵        │
└─────────────────────────────────────────────────────────────────────┘
```

### 阶段一：因子分析（向后看 — 发现异常模式）

基于德勤14步下钻分析法，对已发生的理赔数据进行多维度异常检测：

| 步骤 | 分析模块 | 核心问题 | 产出 |
|:---:|---------|---------|------|
| 1-2 | 数据对账 + 前提检查 | 数据是否可靠？保单是否稳定？ | 对账报告、稳定性评分 |
| 3 | **4级分层异常检测** | 哪些责任/疾病/医院/会员异常？ | L1-L3 异常等级标记 |
| 4 | IBNR 精算预测 | 已发生未报告的赔款有多少？ | IBNR 估值(95% CI) |
| 5 | 标杆匹配 | 同类保单中排名如何？ | 12维度相似度匹配 |
| 6 | 案件关联分析 | 是否存在团伙/家庭关联？ | 关联图谱+可疑标记 |
| 7 | **控费建议生成** | 基于异常模式，如何控费？ | 13项量化措施 |
| 8-10 | 既往症/疾病/医院 | 既往症影响？治疗合理吗？费用标准？ | 3维度深度分析 |
| 11-14 | 健康评分 + 会员画像 + 补充分析 + LLM增强 | 综合风险画像 | 多维度风险评分 |

**因子分析产出 10 大类 86 项指标**（TUI G01-G10 查询组），覆盖核心摘要、保单数据、异常检测、疾病费用、医院分析、会员画像、控费管控、安全调查、预测标杆、风险地图。

### 阶段二：趋势预测（向前看 — 预判未来规模）

独立 LightGBM 子系统，基于 605 万条历史理赔记录训练 Tweedie 回归模型，预测未来案件级赔付金额：

| 能力 | 方法 | 指标 |
|------|------|:---:|
| 案件金额预测 | LightGBM Tweedie (tweedie_power=1.2) | R²=0.65, Gini=0.69 |
| 不确定性量化 | P05/P50/P95 分位数模型 | 95% 预测区间 |
| 大额识别 | 金额 + 分位数双重筛选 | 召回 42%, AUC=0.94 |
| 时序稳定性 | Walk-Forward 季度回测 | Gini=0.72±3% |
| 编码一致性 | 类别映射持久化 (11列1042类) | 全量R² 0.02→0.66 |

**趋势预测产出**（TUI G11 + 综合报表 Ch33）：预测金额分布、分桶统计、模型KPI、回测轨迹。

### 阶段三：综合决策（交叉验证 — 闭环优化）

将因子分析的异常发现与趋势预测的金额预判进行交叉验证，形成优先级矩阵：

```
              预测金额低          预测金额高
           ┌─────────────────┬─────────────────┐
  异常     │                 │  🔴 最高优先级   │
  等级高   │  🟡 需监控       │  立即干预        │
           │  (潜在风险)     │  (已异常+将增长) │
           ├─────────────────┼─────────────────┤
  异常     │                 │                  │
  等级低   │  🟢 低风险       │  🟡 趋势预警     │
           │  (正常)         │  (未异常但预测增) │
           └─────────────────┴─────────────────┘
```

**四象限策略**：
| 象限 | 异常等级 | 预测趋势 | 策略 |
|------|:---:|:---:|------|
| 🔴 右上 | 高 | 高 | **立即干预** — 已出险且预测继续增长，启动调查+控费 |
| 🟡 左上 | 高 | 低 | **监控观察** — 历史异常但趋势收敛，验证控费措施效果 |
| 🟡 右下 | 低 | 高 | **趋势预警** — 当前正常但模型预测金额上升，提前布防 |
| 🟢 左下 | 低 | 低 | **常规管理** — 风险可控，维持现有策略 |

### TUI 闭环操作路径

```
主菜单 → 🚀 开始新分析 → 数据加载 → Phase 2 Agent 分析 (阶段一)
                                         │
                                         ▼
                    分析后菜单 → 🔍 查询分析指标 (G01-G10 查看因子分析)
                                         │
                    ┌────────────────────┘
                    ▼
          运行 ML 预测 (阶段二, 外部命令)
          python -m ml.pipeline.predict --csv ... --policy GP123
                    │
                    ▼
          查询菜单 → 🤖 ML预测 (G11 查看趋势预测)
                    │
                    ▼
          风险地图 → 🗺️ 交叉查看 (G10 四维度+预测叠加)
                    │
                    ▼
          综合报表 → Ch33 ML理赔预测 + 风险地图章节 (阶段三综合)
```

### 方法论理论基础

| 理论方法 | 应用 | 说明 |
|---------|------|------|
| **德勤14步下钻** | 因子分析 | 保险行业标准分析框架，从整体→责任→疾病→医院/会员逐层下钻 |
| **分层异常检测** | 异常发现 | 3级阈值(SIGNIFICANT>20%/MODERATE>10%/SLIGHT>5%)，同比+环比 |
| **Tweedie 回归** | 金额预测 | 处理零膨胀右偏分布，1<p<2区间同时建模零值和连续正值 |
| **Walk-Forward 回测** | 稳定性验证 | 时间序列安全回测，避免前视偏差，季度窗口确保稳定 |
| **分位数回归** | 不确定性量化 | P05/P50/P95 三模型覆盖悲观/中性/乐观三种情景 |
| **交叉验证矩阵** | 综合决策 | 异常等级 × 预测趋势 = 4象限优先级策略 |
| **文件系统松耦合** | 系统集成 | ML子系统输出 JSON/Parquet，主系统按需读取，零代码级依赖 |

### 与 IBNR 的区别

| 维度 | IBNR (精算) | ML 预测 (本系统) |
|------|-----------|-----------------|
| 预测对象 | 已发生未报告赔款 | 未来新发生案件金额 |
| 方法论 | 链梯法/分段精算 | LightGBM Tweedie 回归 |
| 粒度 | 保单/责任级 | **案件级** (单个理赔) |
| 特征 | 赔付三角 | 41个特征 (疾病/医院/会员/时间) |
| 更新频率 | 季度 | 支持日级 |

> **互补关系**：IBNR 估计已发生未报告的存量，ML 预测未来新发生的增量，两者共同构成完整的赔付预测拼图。

### 实施状态 (v1.9.71)

闭环集成已通过 **路径B** 方案落地到 LangGraph 工作流中：

```
LangGraph StateGraph (37节点 → 38节点):

cost_control (控费建议)
    │
    ├── case_investigation, health_score, fwa_analysis ...
    ├── benefit_transfer
    └── ml_prediction  ← 新增: 读取Parquet预测 + 交叉验证anomaly/fwa
            │
            ▼
    report_generation (综合报告)
```

| 集成点 | 位置 | 功能 |
|--------|------|------|
| **LangGraph节点** | `src/agents/ml_prediction_agent.py` | Group B并行加载预测, <2s, 不阻塞 |
| **State字段** | `agent_state.ml_prediction_result` | 交叉验证结果 (Top10责任+医院等级+异常重叠) |
| **TUI风险地图** | G10 "🔮 ML预测维度" (4项) | 风险总览/重叠分析/责任排名/医院分布 |
| **综合报表** | Ch33a "预测×异常交叉矩阵" | 四象限决策 + 责任排名 + 医院分布 |
| **综合报表** | Ch33 "ML理赔预测" | KPI卡片 + 特征重要性 + 回测 |

> 要使用闭环功能，需先在主系统环境安装 ML 预测依赖: `pip install -r requirements-ml.txt`。
> 未安装时 ml_prediction Agent 优雅降级 (available=false)，不影响其他分析功能。

## 知识库管理

独立的知识库管理工具，支持从互联网搜索真实数据源：

```bash
python -m src.tools.knowledge_base_updater status        # 查看当前状态
python -m src.tools.knowledge_base_updater search        # 搜索互联网数据源
python -m src.tools.knowledge_base_updater update --data-file data.json  # 更新YAML
python -m src.tools.knowledge_base_updater report        # 生成版本追踪报告
python -m src.tools.knowledge_base_updater full          # 完整流程
```

数据来源包括：国家医保局、国家卫健委、金融监管总局、省市医保局、中国保险行业协会等。

详见 `docs/knowledge_base_index.md`。

## 性能指标

- **分析时间**: < 0.1秒（德勤案例310条理赔，规则引擎）
- **LLM增强**: 每次分析约4-5分钟（13次LLM调用，79K tokens）
- **测试覆盖**: 4475个测试用例，0失败
- **类型安全**: Pyright检查，55+文件零错误（1个已知可选依赖警告）
- **控费建议**: 自动生成11-13条量化措施
- **可疑模式**: 自动检测20-40个可疑模式
- **Agent并行**: 5+24个Agent同时执行（并行组A和B）
- **报告输出**: 7类报告+三格式导出自动生成（分析/OCR/控费/对比/Excel 42 Sheet/Chart.js HTML/综合可视化HTML）
- **医院名称回填**: 81.5%覆盖率（跨年交叉匹配）

## 依赖项

所有依赖通过 `pip install` 或 `.whl` 自动安装：

- Python >= 3.11
- langgraph >= 1.1.0
- langchain-core >= 1.2.0
- pyyaml >= 6.0
- openpyxl >= 3.1.0
- openai >= 1.0.0
- zai-sdk >= 0.2.0
- pydantic >= 2.10.0,<2.11
- python-dotenv >= 1.0.0

开发依赖（源码模式下 `pip install -r requirements.txt`）：

- pytest >= 7.0.0
- pytest-cov >= 4.0.0
- pytest-html >= 4.0.0
- rich >= 14.0.0
- questionary >= 2.1.0
- httpx >= 0.25.0

## 前置条件

| 条件 | 必需 | 说明 |
|------|------|------|
| Python >= 3.11 | 是 | 运行环境 |
| `config.yaml` | 是 | 配置文件（包内自带） |
| `knowledge/` | 是 | 知识库YAML文件（包内自带） |
| `data/source/` | 分析时需要 | 原始Excel理赔数据 |
| `.env` 文件 | 否 | API密钥（不配则LLM功能自动降级为纯规则引擎） |

## 构建

```bash
# 构建 whl 分发包
pip install build
python -m build
# 产出: dist/ghb_cost_control-1.5.1-py3-none-any.whl

# 验证安装
pip install dist/ghb_cost_control-1.5.1-py3-none-any.whl
ghb-cost-control --version
```

## 开发

```bash
# 运行所有测试
python -m pytest tests/ -v --cov=src

# 运行Agent测试
python -m pytest tests/agents/ -v

# 运行端到端测试
python -m pytest tests/e2e/ -v

# 运行单模块测试
python -m pytest tests/unit/test_precondition_checker.py -v
```

## 配置

通过 `config.yaml` 管理，`Config.load()` 单例读取：

```yaml
llm:
  enabled: true                    # LLM功能开关
  mode: llm_primary                # llm_primary | rule_only | hybrid
  provider_a:                      # 报告生成
    model: "glm-5-turbo"
    temperature: 0.0
  provider_b:                      # 分析任务 + OCR提取
    model: "deepseek-v4-pro"
    temperature: 0.0
    seed: 42
    thinking_enabled: true
  provider_c:                      # 备用
    model: "deepseek-v3.2"
```

## 版本历史

- **v1.9.71** - ML预测子系统: 独立LightGBM Tweedie L1-A案件级金额预测(验证集R²=0.91/Gini=0.75) + 41特征11分类 + Polars惰性加载5.84GB CSV + 两阶段特征工程 + TUI G11 "🤖 ML预测" 查询组3项 + 综合HTML报表Ch33 "ML理赔预测" (KPI卡片+特征重要性+训练概况+回测+分桶统计) + 中文执行报告(MD+JSON) + Walk-Forward回测 + CLI命令对齐(python -m ml.pipeline.train/predict/evaluate)
- **v1.9.70** - Doris批量分析管线: batch_doris_pipeline.py 脚本(传保单号→Phase2分析→agent_state→Q&A提示词) + LLM模式选择表(llm_primary/rule_only/hybrid) + writing子代理切换DeepSeek V4 Pro(解决GLM-4.7超时) + 10保单实测验证 + CLI修复+技能优化: _save_agent_state() + risk-map-analyzer v2.3.0 + 两技能同步至~/.agents/skills/
- **v1.9.69** - P4显示层覆盖修复: workflow._AGENT_OUTPUTS +3字段(修复并行模式数据丢失) + TUI投保单位分析+3面板(同比/同级对比/医院真实性) + TUI风险地图+A021/F023+3详情面板 + 综合HTML风险地图+A021/F023+3详情子章节
- **v1.9.68** - P4分析层增强: 投保单位同比分析(5指标+30%阈值) + 跨单位同级对比(Top5医院+4级偏差) + 医院事故真实性聚合(身份+事故按医院聚合) + FWA非本人就诊4信号(A021) + FWA伪造事故3信号(F023) + FWA规则59→61条
- **v1.9.67** - P3方案实施: 以诊代检4信号增强(关键词34+ICD-10索引25组) + 利益输送检测(3模式: Jaccard/迁移/集中漏斗) + 投保单位代理指标(成员流失率+高风险会员推导) + workflow修复(policy_unit顺序链)
- **v1.9.66** - P0-P2方案实施: multidimensional_metrics多维指标 + fee_rootcause费用根因 + FWA 57→59规则 + DRG住院天数标杆 + 投保单位分析 + DrugDosageChecker药品剂量 + CoverageDeviation保障偏差 + Cigna临床指南 + 医院欺诈聚合 + 36Agent/37节点/122字段
- **v1.9.65** - 移除旧双表(edmp_baz_prd)废弃功能 + Doris连接改为交互式输入用户名密码(TUI+CLI) + download_doris.py脚本重写为宽表模式 + CLI删除--table/--database/--wide旧参数 + config.yaml清理旧配置
- **v1.9.64** - TUI键盘操作一致性+菜单面包屑导航: 安全封装提取至safe_input.py共享模块(safe_select/safe_text/safe_confirm) + params.py 16处raw questionary调用统一为安全封装(Esc返回+1-9快捷键) + app.py内联安全封装重构为共享模块导入 + 6处菜单breadcrumb导航(主菜单/查询/批量/Doris/日志/验证/ETL)
- **v1.9.63** - CLI+TUI功能参数对齐: CLI 7数据源(--doris/--csv/--html) + CLI help/docstring/epilog全面更新 + TUI主菜单工具分组(✅验证+⚙️ETL) + 3处.unsafe_ask()双重调用崩溃bug修复 + 返回文本统一'← 返回'
- **v1.9.62** - 24月走势预测增强: 3源fallback(ramp_trajectory/ramp_analysis.monthly_metrics/claims) + loss_ratio自动计算 + 综合报表数据不足提示章节 + 风险地图4维度(疾病/医院/会员/投保单位) + 统一预警面板
- **v1.9.61** - 风险地图TUI菜单(G10, 23项) + 综合报表32章节(风险地图倒数第二) + FWA语义分类(9类映射) + 不实告知显示连接
- **v1.9.60** - 医院5级分类(公立特需9/公立国际部2/公立普通41) + 投保单位维度分析 + 理赔欺诈深度调查建议(4风险点×4路径)
- **v1.9.59** - P1 FWA 7条新规则(A016-A019拆单/挂床/以诊代检 + F020-F022冒用) + 职业-疾病不匹配检测(12职业组+25规则)
- **v1.9.58** - P0-1医疗合理性引擎(ICD-10×4类规则, 66测试) + P0-2医院收费标准异常(同等级对比+增长趋势, 50测试) + TrendForecastAgent接入workflow(30→31节点)
- **v1.9.57** - FWA规则扩展至54条(原50条) + 医院收费标准异常分析器 + 医疗合理性分析器 + 职业疾病不实告知匹配
- **v1.9.56** - 参考代码v2对齐: ClaimData+10字段(会员人口统计gender/birthday/nationality/relationship + 保单级属性cont_valid_date/pass_months/annual_prem/pass_prem/rnew_flag + duty级金额) + PolicyAnalysisData+2字段(pass_months/est_loss_ratio) + _CSV_TO_CLAIM_MAP+10映射 + DorisCsvLoader投保人名称修正(_fix_applicant_name) + 保单级属性提取 + 路径A/B第4层fee_item_type维度(保留lob_level2_name fallback) + Top30会员门诊分析/诊断Top3/费用细项次数Top5/赔付占比 + 责任排序映射表 + 异常聚集3维度(医院+ICD+日期) + LevelResult新增diff_ratio_to_total/HIGH_NEW
- **v1.9.55** - 新宽表E2E: c001_ghb_poicy_clm_duty_d宽表适配(单表70+字段) + DorisConnector宽表导出 + CSV_MAP大写列名兼容 + ClaimData+9字段 + load_policy_from_wide + extract_policies_from_wide + TUI自动检测宽表 + 进度条动态估算
- **v1.9.54** - Doris数仓批量ETL+分析: TUI一站式(下载/扫描/提取/分析/报告) + ThreadPoolExecutor 4线程并行 + 64MB大缓冲 + 24h扫描缓存 + 分页表格(首页/末页/跳转) + 跨保单汇总报告 + 进度条全覆盖 + 文件占用容错
- **v1.9.43** - TUI菜单emoji图标+Esc返回+操作提示: 全菜单17处统一`_safe_select()`(emoji图标+底部操作提示+Esc键返回哨兵值) + 查询类别emoji(⭐📋🔍🏥🏛️👥💰🛡️📈) + 主菜单emoji(🚀📂📦ℹ️📖📄📋🚪) + 返回选项统一"← 返回"前缀
- **v1.9.42** - TUI主菜单系统管理一级入口: 查询菜单10组→9组56项(移除系统管理组) + 主菜单Separator分组美化(分析/批量/系统) + 系统管理一级入口(版本信息/README/AGENTS.md/执行日志查询子菜单) + "Agent执行详情"移入核心摘要(5→6项)
- **v1.9.40** - TUI菜单分组优化: 13组→10组职责单一化(核心摘要5项黄金首屏/控费与管控7项/安全与调查7项/预测与标杆5项/系统管理14项)
- **v1.9.39** - TUI系统信息菜单: 新增第13组"系统信息"(3项: 版本信息/查看README/查看AGENTS.md) + 分页Markdown渲染(60行/页+翻页) + Rich Table版本详情 + 菜单总计13组69项
- **v1.9.38** - 综合可视化HTML报告优化: TOC双列目录(20项锚点) + 品牌横幅(GHB渐变) + 每章"返回目录"链接 + 浮动"返回顶部"按钮(scroll显隐) + @media print打印样式 + section-nav补全4个缺失锚点
- **v1.9.37** - StandaloneHtmlReport全面优化: TOC导航 + 全面中文化(11节标题/表头/KPI标签/货币单位"万元") + 品牌横幅 + 4新section(执行摘要/保单健康度/路径A下钻/路径B下钻) + 同比列(责任/疾病/医院) + @media print打印样式
- **v1.9.36** - Excel命名对齐: 路径A第4层`6_A_医院_费用`→`6_A_医院_二级责任`(对齐参考Excel) + StandaloneHtmlReport新增策略匹配section(P0-3)
- **v1.9.35** - TUI美化: Rich品牌横幅(GHB+版本号+青色边框) + 查询菜单标题(保单名+分隔线) + questionary提示中文化
- **v1.9.34** - TUI菜单分组优化(12组66项重构): 拆案件与关联/合并疾病+费用/移动5个错配项/中文直写(unicode转义→中文字符)
- **v1.9.33** - 费用细项异常检测概览面板(summary 6指标) + feetype_category.flagged异常标记表
- **v1.9.32** - 医院多维分析+Top医院费用修复: hospital_anomalies按医院聚合(actual_cost非total_paid)
- **v1.9.31** - 医院类型分布3层数据源: _policy_summary预计算(公立/私立) / cost_standards.yaml精确匹配(综合/专科) / 名称启发式推断
- **v1.9.30** - TUI查询菜单7个display函数19处键名不匹配修复: large_claims/fee_structure/cost_control_tracking/pre_authorization/prior_condition/benchmark/ndc_detail
- **v1.9.29** - 医院类型分布预计算: _save_per_policy_state聚合is_public(公立/私立)到_policy_summary
- **v1.5.1** - 总结与控费建议P0: ComparisonAnalysisGenerator纯规则对比分析引擎(591行) + SummaryReporter执行摘要(470行) + PT-016执行摘要增强Prompt + 历史同比贯穿三入口(CLI报告顶部/TUI查询首选项/Excel 22 Sheet含总结) + AI分析文字CSS(.ai-analysis-box/.summary-box/.insight-card) + 责任/疾病Sheet新增6+2列YoY历史对比 + report_generation_agent双模式(rule_executive_summary+llm_executive_summary) + 测试模板数15->16适配
- **v1.5.0** - (内部版本: P0方案设计+Phase A2/A3/C1/C3前置工作)
- **v1.4.3** - whl部署修复: 集中化项目根目录定位(get_project_root: 开发模式+CWD回退) + 10个模块路径引用更新 + pyproject.toml package-data配置
- **v1.4.2** - LLM激活修复: 放宽API Key校验(仅强制Provider B, 可选Provider缺失时降级) + LLM配置诊断消息 + _restore_llm_config()返回诊断结果
- **v1.4.1** - DRG反向查找修复: 标点归一化(全角逗号/斜杠→顿号) + SHAREHOME别名补充(98.7%覆盖率)
- **v1.4.0** - ICD-10中文疾病名称反向查找: 1,689条3位码+44条分组映射+56条别名 + DRG grouper反向查找 + TUI反查统计
- **v1.3.1** - DRG分组明细: 模糊匹配标记+over_budget_drgs+hospital_performance字段映射
- **v1.3.0** - DRG范围码展开: _expand_range_code("J00-J99") + 模糊匹配架构
- **v1.2.8** - TUI优化: 进度条/键盘/菜单/8个渲染函数/6轮字段映射修复/版本号显示/FWA标签修正
- **v1.0.0** - Phase E发布就绪: 3005测试(Phase 2 E2E + 3新analysis模块 + 6薄文件增厚) + 代码质量修复(except...pass/docstring/箭头符号) + 3篇参考文档(配置参考/部署指南/API参考)
- **v0.11.0** - Phase D集成深化: 报告集成(6 SectionReporter接入主报告+控费报告) + Excel+Chart.js扩展(6 Sheet+6图表) + LLM增强(PT-013/014/015 + 3 Agent改造: cross_dimension/pre_authorization/fwa) + llm_primary模式(4 Agent + Provider C故障转移) + TUI 22 Agent显示修复
- **v0.10.0** - Phase C功能完善: 费用明细分项分析(7方法) + 标杆匹配画像(12维度25标杆) + OCR闭环质量(置信度+复核标记) + 医院库扩展(52家20+城市) + 控费追踪闭环(生命周期+效果监控+t-test+告警服务)
- **v0.9.0** - Phase B规则模型增强: FWA反欺诈(50规则) + DRG/DIP分组(CHS-DRG 2.0) + DEG诊断分组(ICD-10) + NDC病程校验(8慢病) + IBNR精算分段 + EDMP早期疾病管理 + Chart.js交互式HTML仪表盘
- **v0.8.0** - Phase A量化控费指标增强: HHI/CR4/CR10医院集中度 + Gini系数/Lorenz曲线 + Top30会员费用排名 + 同比/环比趋势对比 + 描述字段per-claim结构化解析(ClaimData.parsed_description + BaseAgent helper)
- **v0.7.0** - Excel导出恢复 + Chart.js HTML仪表盘: 21 Sheet Excel报告 + Chart.js交互式HTML + TUI双格式自动输出
- **v0.6.0** - 死代码清理(15688行删除) + Pyright 177->1错误 + fitz依赖修复
- **v0.5.0** - ETL智能管线 + TUI实时仪表盘 + pip可安装: 经过保费动态匹配(默认12月) + 医院名称跨年回填(81.5%) + 7项映射修复 + TUI交互模式(Phase 1/2/Compare) + 4类报告输出 + OCR数据加载重构 + DescriptionFieldParser集成 + Pyright类型安全(55文件) + `ghb-cost-control` CLI入口
- **v0.4.0** - OCR理赔文档分析: 双引擎提取(规则+LLM) + OCR-理赔关联匹配 + 独立调查报告 + PT-012提示模板
- **v0.3.0** - 知识库管理工具 + 互联网数据源 + 版本追踪
- **v0.2.1** - LLM-Primary架构 + A/B测试 + 监控 + 灰度发布 + 性能优化
- **v0.2.0** - Phase 2: LangGraph Agent (15个) + CLI + Docker + 双模式对比
- **v0.1.0** - Phase 1: 轻量编排器 (14步) + 22个分析模块

## 许可证

内部使用 - 招商信诺人寿保险有限公司
