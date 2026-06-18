# AGENTS.md - GHB控费AI大模型分析系统

团体保险赔付分析系统。德勤14步下钻分析法。Python 3.11+ + LangGraph 1.1。双模式架构：规则引擎 + LLM增强（Provider A: GLM-5-Turbo / Provider B: DeepSeek-V4-Pro / Provider C: DeepSeek-Flash）。独立ML预测子系统(LightGBM Tweedie + 督导后向分析)。**闭环分析模型**（因子分析→趋势预测→综合决策，三阶段交叉验证）。OCR双引擎提取 + ETL智能管线。TUI实时仪表盘(两级分组查询菜单11组89项 + 风险地图4维度23项 + 主菜单工具+系统管理 + emoji图标 + Esc返回上级 + 面包屑导航 + j/k翻页 + 鼠标滚动 + Doris数仓批量ETL+分析500保单 + 📖查看README查看完整方法论)。CLI 7数据源(--policy/--file/--etl/--doris/--csv) + 综合HTML报告(--html) + ML预测管线(python -m ml.pipeline.train/predict/evaluate)。医疗合理性引擎(ICD-10×4类规则) + 医院收费标准异常(同等级对比+增长趋势) + 24月走势预测(3源fallback)。FWA 61条规则(22欺诈+18浪费+21滥用, 含拆单/挂床/以诊代检/冒用)。不实告知职业疾病匹配。2技能插件(risk-map-analyzer v2.4.0 + ghb-cost-control)。4549测试。v1.9.71。

## 命令

```bash
python -m pytest tests/ -v                                    # 全量(4549)
python -m pytest tests/unit/test_precondition_checker.py -v  # 单文件
python -m pytest tests/ -v --cov=src --cov-report=term-missing  # 覆盖率
python -m pytest tests/agents/test_agents.py::TestCostControlAgent -v  # 单Agent类
python -m pytest tests/unit/test_drg_grouper.py -v           # DRG测试(含反向查找)
python -m src.cli analyze --policy deloitte --agent           # Phase 2
python -m src.cli analyze --policy deloitte --excel           # Phase 2 + Excel导出
python -m src.cli analyze --policy deloitte --html            # Phase 2 + 综合HTML报告
python -m src.cli analyze --policy deloitte --excel --html    # Excel + 综合HTML双格式
python -m src.cli analyze --doris data/doris/policies --doris-policy GP123 --llm-mode llm_primary  # Doris + LLM模式
python skills/ghb-cost-control/scripts/batch_doris_pipeline.py PID1 PID2 --qa --llm-mode llm_primary  # 批量管线
python skills/ghb-cost-control/scripts/batch_doris_pipeline.py PID1 --qa-only  # 仅生成Q&A
python -m src.cli analyze --csv data/csv --csv-policy GP123   # 费用明细CSV数据源
python -m src.cli analyze --policy deloitte --llm-mode hybrid # 指定LLM模式
python -m src.cli analyze --policy deloitte --report general  # 指定报告类型(deloitte/general/both)
python -m src.orchestrator --policy deloitte                  # Phase 1
python -m src.cli compare --policy deloitte --verbose         # 双模式对比
python -m src.cli compare --doris data/doris/policies         # Doris数据双模式对比
python -m src.cli etl --verbose                               # ETL管线
python -m src.cli tui                                         # TUI交互模式
python -m src.cli tui --skip-input                            # TUI(跳过参数输入)
python -m src.cli download --password XXX                     # 从Doris数仓下载CSV(交互输入用户名+密码)
python -m src.cli download --test --password XXX              # 测试Doris连接
python -m src.cli download --test --password XXX              # 测试Doris连接
ghb-cost-control tui                                          # pip安装后(连字符非下划线)
docker compose up                                             # Docker

# ML预测管线 (独立子系统)
python -m ml.pipeline.train --config ml/config_ml.yaml        # 训练L1-A模型
python -m ml.pipeline.train --config ml/config_ml.yaml --mode full --no-backtest  # 快速训练(跳过回测)
python -m ml.pipeline.predict --csv data/doris/c001_ghb_poicy_clm_duty_d.csv  # 全量预测
python -m ml.pipeline.predict --csv ... --policy GP123        # 单保单预测
python -m ml.pipeline.evaluate --output reports/ml/           # 生成中文评估报告
python ml/pipeline/run_backtest.py                            # Walk-Forward回测
```

## 目录结构

```
src/
├── orchestrator.py           # Phase 1 Thin Shell(198行, main()已移除→cli.py)
├── cli.py                    # CLI入口(argparse, 8子命令: analyze/validate/test/compare/etl/tui/logs/download, analyze支持7数据源+--html综合报告)
├── config.py                 # YAML配置单例 + get_project_root()路径定位
├── __main__.py               # python -m src 支持(委托cli:main)
├── __init__.py               # __version__(注意: 需与pyproject.toml同步)
├── models/policy_data.py     # PolicyAnalysisData(6个dataclass)
├── agents/                   # 36 Agent文件 + state.py + workflow.py + base.py
│   ├── orchestrator_agent.py # 高级编排API + load_ocr_data()
│   ├── state.py              # AgentState(TypedDict, 122字段, totalFalse)
│   ├── workflow.py           # LangGraph StateGraph(37节点, fan-out/fan-in)
│   └── case_investigation/   # 子包(6文件)
├── analysis/                 # 36个纯算法模块(无state/LLM/DB依赖, 含medical_rationality/hospital_fee_anomaly/trend_forecaster/multidimensional_metrics/fee_rootcause_analyzer/policy_unit_analyzer/benefit_transfer_analyzer/drug_dosage_checker/coverage_deviation_analyzer)
├── knowledge/                # 3匹配器 + knowledge/目录20个YAML知识文件
│   ├── clinical_guidelines/  # Cigna临床指南(10组疾病)
│   └── drug_dosage_rules.yaml # 药品剂量规则(50+药品)
├── report/                   # 报告模块 + sections/36个SectionReporter(含SummaryReporter)
├── llm/                      # 三提供商路由 PT-001~PT-016(16个Prompt模板)
├── data_sources/             # ETL管线 + OCR链 + 福利计划 + 描述字段解析 + CSV加载器 + Doris连接器
├── tui/                      # TUI实时仪表盘(7文件, app.py为主控, 两级分组10组86项+emoji+Esc返回+面包屑)
│   ├── safe_input.py         # 安全输入封装(safe_select/safe_text/safe_confirm + BACK_SENTINEL + MENU_INSTRUCTION)├── batch/                    # 批量分析(BatchAnalysisOrchestrator + CrossPolicyReport + DorisBatchExtractor + DorisBatchAnalysisOrchestrator)
├── services/                 # OCRService + AlertService + ExecutionLogger
├── tools/                    # 知识库管理 + 文档提取
├── recommendations/          # 控费建议生成器
├── reconciliation/           # 数据对账
├── policy/                   # 保单条款 + 既往症
└── database/schema.py        # SQLite

ml/                           # 独立ML预测子系统
├── data/loader.py            # Polars惰性加载 (5.84GB, 71列)
├── data/feature_store.py     # 两阶段特征工程 (41特征11分类)
├── data/split.py             # 时间序列安全划分
├── models/amount_predictor.py # LightGBM Tweedie + 分位数模型
├── evaluate/metrics.py       # 保险评估指标 (Gini/MAPE/总量误差)
├── evaluate/backtest.py      # Walk-Forward时序回测
├── pipeline/train.py         # CLI训练入口
├── pipeline/predict.py       # CLI预测入口
├── pipeline/evaluate.py      # CLI评估入口
├── pipeline/run_backtest.py  # 独立回测脚本
├── report/generator.py       # 中文执行报告生成器
└── config_ml.yaml            # ML独立配置

tests/                        # 4549 passed
├── unit/(67文件) agents/(25文件) e2e/(2文件)
├── fixtures/deloitte_case.py  # 工厂(310理赔,25会员,10医院)
└── conftest.py               # 三重LLM禁用(autouse session fixture)

knowledge/                    # YAML知识文件(20个, 医院/标杆/疾病/DRG/DEG/FWA/控费/IBNR/ICD-10名称/医疗合理性/职业疾病/药品剂量/Cigna临床指南/利益输送规则)
scripts/                      # 22个辅助脚本
skills/ghb-cost-control/      # Skill插件(8文件: SKILL.md + 5参考文档 + 2脚本)
```

## 入口点

| 入口 | 用途 |
|------|------|
| `ghb-cost-control tui` | TUI交互模式(pip install -e .后可用,连字符) |
| `python -m src.cli tui` | TUI交互模式(等效,无需安装) |
| `python -m src.cli analyze --policy deloitte --agent` | Phase 2 LangGraph Agent |
| `python -m src.cli analyze --policy deloitte` | Phase 1 编排器 |
| `python -m src.cli analyze --policy deloitte --html` | Phase 2 + 综合HTML报告 |
| `python -m src.cli analyze --policy deloitte --excel --html` | Excel + 综合HTML双格式 |
| `python -m src.cli analyze --doris data/doris/policies --doris-policy GP123` | Doris CSV数据源 |
| `python -m src.cli analyze --csv data/csv --csv-policy GP123` | 费用明细CSV数据源 |
| `python -m src.cli compare --policy deloitte --verbose` | 双模式对比 |
| `python -m src.cli compare --doris data/doris/policies` | Doris数据双模式对比 |
| `python -m src.cli etl --verbose` | ETL管线 |
| `python -m src.cli download --password XXX` | 从Doris数仓下载CSV(交互输入用户名+密码) |
| `python -m src.cli download --test --password XXX` | 测试Doris连接 |
| `python -m src.orchestrator --policy deloitte` | Phase 1(直接, main()已移至cli.py) |
| `python -m src.tools.document_extractor scan --dirs "data/images" --workers 2` | OCR扫描 |
| `python -m ml.pipeline.train --config ml/config_ml.yaml` | ML模型训练 |
| `python -m ml.pipeline.predict --csv data/doris/c001_ghb_poicy_clm_duty_d.csv` | ML批量预测 |
| `python -m ml.pipeline.evaluate --output reports/ml/` | ML评估报告 |
| `python ml/pipeline/run_backtest.py` | ML Walk-Forward回测 |

pyproject.toml定义了`[project.scripts] ghb-cost-control = "src.cli:main"`。命令名是**连字符**`ghb-cost-control`，非下划线。

## CLI参数(argparse)

`analyze`和`compare`共用关键参数:
- `--llm-mode`: 覆盖config.yaml的LLM模式(`llm_primary`/`hybrid`/`rule_only`)
- `--api-key-a/b/c`: 运行时注入API密钥(GLM/DeepSeek-V4/DeepSeek)
- `--doris`: Doris数仓CSV保单目录路径(配合`--doris-policy`指定保单号)
- `--doris-policy`: Doris数仓团体保单号(不指定则自动检测)
- `--csv`: 费用明细CSV目录路径(配合`--csv-policy`指定保单号)
- `--csv-policy`: 费用明细CSV团体保单号(不指定则加载首个)

`analyze`独有参数:
- `--report`: 报告类型(`deloitte`/`general`/`both`)
- `--excel`: 触发Excel+Chart.js HTML双格式导出
- `--html`: 生成综合可视化HTML报告(32章节, Chart.js图表+SVG内联)
- `--agent`: 启用Phase 2 LangGraph Agent模式

数据源优先级: `--etl` → `--file` → `--doris` → `--csv` → `--policy`(默认)

## TUI交互模式

入口: `python -m src.cli tui` 或 `ghb-cost-control tui`。主文件: `src/tui/app.py`(~14,965行)。

### 菜单架构(四级)

```
主菜单(启动) → 分析执行 → 分析后菜单 → 查询菜单(10组86项)
                  ↑_______________|  (new_task循环)
```

### 主菜单(启动, line 307)

Separator分组: 分析/批量/工具/系统。

| 选项 | 功能 |
|------|------|
| 🚀 开始新分析 | → 参数收集 → 数据加载 → Phase 1/2/Compare → 报告导出 |
| 📂 加载已有分析数据 | 选agent_state.json → 直接进入查询菜单 |
| 📦 批量分析管理 | CSV/Doris批量分析子菜单(历史/重新生成/加载保单) |
| ✅ 数据验证 | 交互式选择数据源 → Phase 1验证 → 错误/警告面板 |
| ⚙️ ETL管线 | 交互式指定源/输出目录 → 运行ETL → 结果面板 |
| ℹ️ 版本信息 | 系统版本/Python/LangGraph/LLM配置 |
| 📖 查看README | 分页Markdown(60行/页, j/k翻页, 鼠标滚动) |
| 📄 查看AGENTS.md | 分页Markdown(60行/页, j/k翻页, 鼠标滚动) |
| 📋 执行日志查询 | → 系统日志子菜单(7项) |
| 🚪 退出 | return 0 |

### 分析后菜单(line 3658)

| 选项 | 功能 |
|------|------|
| 🔍 查询分析指标 | → 查询菜单(10组86项) |
| 📂 加载其他数据重新查询 | 选agent_state.json → 进入查询菜单 |
| 🚀 执行新任务 | → 回到参数收集 |
| 🚪 退出 | return "exit" |

### 查询菜单(10组86项, line 3741)

**选择类别 → 选择指标 → 查看结果(按回车继续)**

类别选择额外选项: `📄 重新生成综合HTML报告` / `🔄 重新加载数据` / `← 返回主菜单`

| # | 类别 | 项数 | 查询项(→ display函数) |
|---|------|------|----------------------|
| G01 | ⭐ 核心摘要 | 6 | 总结与控费建议(`_display_summary`) / 执行摘要(`_display_executive_summary`) / 关键指标总览(`_display_final_summary`) / 评分结果(`_display_scoring_results`) / 报告文件列表(`_display_generated_reports`) / Agent执行详情(`_display_agent_summary`) |
| G02 | 📋 保单与数据 | 8 | 数据对账(`_display_reconciliation`) / 前提条件检查(`_display_stability`) / 保单属性对比(`_display_policy_comparison`) / 投保单位分析(`_display_policy_unit`) / 出险城市分析(`_display_city_analysis`) / 月度爬坡轨迹(`_display_ramp_trend`) / 大额理赔分析(`_display_large_claims`) / 核保LR对比(`_display_uw_lr_detail`) |
| G03 | 🔍 异常检测与下钻 | 6 | 4级分层异常检测(`_display_anomaly_detail`) / 异常等级分级概览(`_display_anomaly_tier_summary`) / 责任同比对比(`_display_yoy_comparison_simple`) / 环比异常分析(`_display_mom_anomaly_detail`) / 路径A下钻分析(`_display_path_a_drilldown`) / 路径B下钻分析(`_display_path_b_drilldown_detail`) |
| G04 | 🏥 疾病与费用 | 9 | 疾病-治疗匹配(`_display_disease_treatment`) / 疾病ICD分析明细(`_display_disease_icd_detail`) / DRG/DIP分组(`_display_drg_detail`) / DEG诊断分组(`_display_deg_detail`) / 费用明细分项分析(`_display_fee_structure`) / 费用细项异常检测(`_display_fee_item_anomalies`) / EDMP早期疾病管理(`_display_edmp_detail`) / NDC病程校验(`_display_ndc_detail`) / 医疗合理性审核(`_display_medical_rationality`) |
| G05 | 🏛️ 医院分析 | 6 | 医院集中度分析(`_display_hospital_concentration`) / 医院类型分布(`_display_hospital_type`) / 医院多维分析(`_display_hospital_multidimensional`) / Top医院费用细项(`_display_top_hospital_fee`) / 集中就医分析(`_display_concentrated_visit`) / 收费标准异常检测(`_display_hospital_fee_anomaly`) |
| G06 | 👥 会员与画像 | 5 | 会员画像(`_display_member_profile`) / Pareto分析(`_display_pareto`) / Top理赔分析(`_display_top_claims`) / 高频会员行为画像(`_display_member_behavior`) / 会员风险评分(`_display_member_risk_scores`) |
| G07 | 💰 控费与管控 | 10 | 控费建议明细(`_display_recommendation_details`) / 归因分析结果(`_display_attribution_detail`) / 赔付率因素分解(`_display_lr_decomposition`) / 控费策略匹配(`_display_strategy_matching`) / 控费追踪闭环(`_display_cost_control_tracking`) / 预授权分析(`_display_pre_authorization`) / LLM增强明细(`_display_llm_metrics`) / 交叉维度分析(`_display_cross_dimension`) / 预警监控(`_display_alerts`) / 🔔 统一预警面板(`_display_unified_alerts`) |
| G08 | 🛡️ 安全与调查 | 7 | FWA反欺诈(`_display_fwa_detail`) / LLM FWA评估(`_display_llm_fwa_assessment`) / 治疗与用药审核(`_display_medical_audit`) / 案件关联分析(`_display_claim_correlation`) / 家庭关联模式(`_display_family_patterns`) / 同日聚集分析(`_display_same_day_clustering`) / 案件调查(`_display_case_investigation`) |
| G09 | 📈 预测与标杆 | 6 | IBNR预测(`_display_ibnr_detail`) / 标杆匹配(`_display_benchmark`) / 既往症分析(`_display_prior_condition`) / 健康评分(`_display_health_detail`) / OCR-理赔关联(`_display_ocr_correlation`) / 24月走势预测(`_display_trend_forecast`) |
| G10 | 🗺️ 风险地图 | 23 | 🔴疾病维度总览 / TOP疾病赔付 / 疾病-治疗匹配 / 医疗合理性 / 过度医疗(疾病) / 疾病就诊聚集 / 🔵医院维度总览 / 医院五级分类 / 集中度 / 收费标准异常 / 过度医疗(医院) / 就诊聚集 / 🟢会员维度总览 / 风险评分 / 高频行为 / 既往症与不实告知 / 过度医疗(会员) / 家庭关联 / 🟡投保单位维度总览 / 投保单位分析 / 既往症(单位) / 欺诈风险(单位) / 疾病聚集(单位) |
| G11 | 🤖 ML预测 | 3 | 预测汇总概览(`_display_ml_overview`) / 保单级预测对比(`_display_ml_policy_predictions`) / 模型评估详情(`_display_ml_model_metrics`) |

### 系统日志子菜单(line 2295)

| 选项 | 功能 |
|------|------|
| 🕐 最近执行历史 | `_display_recent_executions()` |
| ⏱️ Agent耗时排行 | `_display_agent_duration_ranking()` |
| ❌ 失败/错误日志 | `_display_error_logs()` |
| 📊 按日执行统计 | `_display_daily_stats()` |
| ⚡ 运行性能查询 | `_display_run_performance_query()` |
| 📁 日志文件管理 | `_display_log_file_manager()` |
| 🔍 日志文件查询 | `_display_jsonl_log_query()` |

### 批量分析子菜单(line 2544)

主菜单"📦 批量分析管理"进入, 5项:

| 选项 | 功能 |
|------|------|
| 📜 查看批量分析历史 | `_view_batch_history()` |
| 🔄 从历史结果重新生成报告 | `_regenerate_batch_report()` |
| 📊 批量生成综合分析报告(Skill模式) | `_generate_batch_reports_from_skill()` |
| 🗄️ Doris数仓批量ETL+分析 | → `_doris_batch_menu()`子菜单(6项: 下载/扫描/提取/分析/列表/返回) |
| 📂 加载指定保单分析结果 | `_load_policy_from_batch()` → 进入查询菜单 |

### TUI交互机制

- **面包屑导航**: `_breadcrumb(*parts)`在各菜单入口显示导航路径(如`主菜单 › 📋 执行日志查询`)
- **Esc/← 返回**: `safe_select()`(safe_input.py)注入prompt_toolkit键绑定, 返回`BACK_SENTINEL`哨兵值
- **底部操作提示**: 全菜单统一显示`MENU_INSTRUCTION = "(↑↓ 选择  Enter 确认  1-9 快捷选择  Esc/← 返回上级)"`
- **安全封装共享模块**: `src/tui/safe_input.py` — safe_select/safe_text/safe_confirm + BACK_SENTINEL + MENU_INSTRUCTION, app.py和params.py共用
- **返回选项**: 统一"← 返回"/"← 返回上级"/"← 返回主菜单"前缀
- **Markdown翻页**: `_display_markdown_paginated()`(line 2478)使用`prompt_toolkit.PromptSession`直接读取按键: `j`/`↓`/`Enter`/`Space`/鼠标滚轮下=下一页, `k`/`↑`/鼠标滚轮上=上一页, `g`=首页, `G`=末页, `:`=跳转, `q`/`Esc`/`←`=退出
- **数据来源**: 所有display函数接收`agent_state: dict`(从agent_state.json反序列化或LangGraph stream产出)
- **数据降级**: 从JSON加载时`_policy_summary`(14字段)作为快速摘要缓存, 缺失时从agent_state推导(5个函数有fallback)
- **重新生成HTML**: 查询菜单中`📄 重新生成综合HTML报告`调用`ComprehensiveReportGenerator.generate()`, 支持policy_data=None降级

### TUI四阶段流程

| 阶段 | 函数 | 说明 |
|------|------|------|
| 阶段零 | `run_tui_analysis()`(line 270) | 启动菜单 + `collect_params()`参数收集 |
| 阶段一 | `_run_phase1_tui()` / `_run_phase2_tui()` / `_run_compare_tui()` | Rich Live面板实时显示Agent进度 |
| 阶段二 | `_post_analysis_menu()`(line 3658) | 分析后主菜单(查询/加载/新任务/退出) |
| 阶段三 | `_query_metrics_menu()`(line 3741) | 查询菜单(10组86项) + 自动报告导出 |
| 阶段四 | 自动(阶段二后) | Excel + Chart.js HTML + 综合可视化HTML三格式导出 |

### 添加新查询项

1. `CATEGORY_GROUPS`(line 3767): 在对应组添加菜单项名称
2. `HANDLERS`(line 3854): 添加 `菜单项名称: lambda: _display_xxx(agent_state)`
3. 实现 `_display_xxx(agent_state: dict | None)` 函数
4. 综合HTML报告: `ComprehensiveReportGenerator`添加对应`_chapter_*`方法 + `_build_all_chapters`调用 + `_build_toc`/`_build_section_nav`更新

## 代码风格

### 导入(三组空行分隔)
1. 标准库 2. 第三方 3. 本地(绝对导入 `from src.xxx`)

### 命名
PascalCase类, snake_case函数/变量, UPPER_SNAKE_CASE常量, *Agent后缀, _前缀私有

### 类型注解
所有公共API必须有。AgentState用TypedDict(total=False)。`total=False`的key必须用`.get()`访问。

### 日志
```python
logger = logging.getLogger(__name__)                    # 模块
self.logger = logging.getLogger(f"agent.{self.name}")   # Agent
```

### 错误处理
Agent层: _safe_execute() → state["errors"]。模块层: raise ValueError()。

## 数据流

```
PolicyAnalysisData(统一容器)
  ├── to_stability_check_format() → PreconditionChecker
  ├── to_hierarchical_detector_format() → HierarchicalAnomalyDetector
  ├── to_ibnr_format() → IBNRPredictor
  ├── to_benchmark_format() → BenchmarkMatcher
  ├── to_cost_control_format(anomalies) → CostControlGenerator
  └── to_prior_condition_format() → PriorConditionAnalyzer
```

三入口共用同一套LangGraph StateGraph(37节点)和Agent实现:
- **CLI**: `.invoke()` + OrchestratorAgent.run()
- **TUI**: `.stream()` + 手动编排(绕过OrchestratorAgent, 直接workflow.stream)
- **Skill**: 进程内`import src.cli:main()`调用(A1方案)

## LLM架构

三提供商路由，LLM增强可选，不影响核心流程。

- Provider A: PT-006报告生成(GLM-5-Turbo, zhipuai, api_key_env=GLM_API_KEY)
- Provider B: PT-001,003-005,007-015分析 + OCR提取 + PT-016执行摘要增强(DeepSeek-V4-Pro, deepseek, api_key_env=DEEPSEEK_V4_API_KEY)
- Provider C: 备用(DeepSeek-Flash, tencent cloud, api_key_env=DEEPSEEK_API_KEY) + 故障转移
- 21个纯规则Agent: reconciliation/stability/benchmark/ramp/member_profiler/extended_metrics/drg/deg/edmp/fee_structure/cost_control_tracking/path_a_drilldown/medical_rationality/hospital_fee_anomaly/trend_forecast/multidimensional_metrics/fee_rootcause/policy_unit/benefit_transfer/drug_dosage/coverage_deviation
- FWA 61条规则(22欺诈+18浪费+21滥用): 拆单/挂床/以诊代检/冒用/身份怀疑/事故伪造
- 13个LLM增强Agent: anomaly/ibnr/claim_correlation/prior_condition/disease_treatment/hospital_cost/health_score/case_investigation/cost_control/cross_dimension(PT-013)/pre_authorization(PT-014)/fwa(PT-015)/ramp
- llm_primary模式: 4个Agent(anomaly/ibnr/health_score/cost_control)支持LLM先行+规则验证
- 优雅降级: LLM失败→规则引擎结果
- Token追踪: _safe_execute()快照LLMClient全局统计

## 工作流(当前)

```
reconciliation → stability → anomaly_detection → ibnr
                                                       |
                                                [route_after_macro]
                                               /                    \
                                     start_parallel            direct_report
                                /    |    |    |    \              |
                         benchmark claim prior disease hospital   |
                                \    |    |    |    /             |
                                cost_control                    |
                                     |                         /
                                     /|_______________________
                                    / |   |   |   |   |   |   \
                     case health ramp member cross extend pre_auth
                     fwa   drg    deg   edmp  fee   tracking
                     path_a_drilldown
                     medical_rationality hospital_fee_anomaly
                     trend_forecast benefit_transfer
                                 \   |   |   |   |   |   |   /
                               policy_unit → report_generation → END
```

并行组A(5): benchmark, claim_correlation, prior_condition, disease_treatment, hospital_cost
并行组B(24): case_investigation, health_score, ramp_analysis, member_profiler, cross_dimension, extended_metrics, pre_authorization, fwa_analysis, drg_analysis, deg_analysis, edmp_analysis, fee_structure_analysis, cost_control_tracking, path_a_drilldown, medical_rationality, hospital_fee_anomaly, trend_forecast, multidimensional_metrics, fee_rootcause, drug_dosage, coverage_deviation, benefit_transfer
Sequential chain: fwa_analysis → policy_unit → report_generation

## 报告生成

report_generation_agent生成4类报告(+ 自动HTML):
1. `analysis_report_{policy_id}_{timestamp}.md` — 主报告(40+节/36个SectionReporter, 含总结与控费建议顶部章节)
2. `ocr_investigation_{policy_id}_{timestamp}.md` — OCR调查(条件: ocr_documents/ocr_extracted非空)
3. `cost_control_report_{policy_id}_{timestamp}.md` — 控费报告(13章, 含总结与控费建议首章)
4. `report_comparison_{policy_id}_{timestamp}.md` — 双报告对比

TUI自动触发三格式导出(Excel + Chart.js HTML + 综合可视化HTML):
5. `cost_control_data_{policy_id}_{timestamp}.xlsx` — 42 Sheet Excel数据报告(含"总结与控费建议"Sheet + 责任/疾病历史同比列 + 路径A第4层"二级责任名称"对齐参考Excel)
6. `cost_control_data_{policy_id}_{timestamp}_dashboard.html` — Chart.js交互式HTML仪表盘(10图表类型, click-to-filter, CSV导出)
7. `comprehensive_report_{policy_id}_{timestamp}.html` — 综合可视化HTML报告(32章节, Chart.js图表+SVG内联, TOC导航+品牌横幅+返回目录/顶部, @media print打印样式)

**报告生成器调用关系**:
- `ComprehensiveReportGenerator`(`src/report/comprehensive_report_generator.py`, ~6220行): TUI阶段四调用, 32章节全覆盖, Chart.js图表, **唯一被TUI使用的HTML报告生成器**
- `ExcelChartHtmlGenerator`(`src/report/excel_chart_html_generator.py`): 从Excel数据生成dashboard.html, TUI阶段四调用
- `StandaloneHtmlReport`(`src/report/standalone_html.py`, ~1410行): 单体HTML(无CDN, 内联SVG), 11节+TOC+中文化+品牌横幅, **当前为死代码(未被调用), 预留CLI `--html`触发** — 注意: CLI `--html`已改为使用`ComprehensiveReportGenerator`生成32章节综合HTML报告

所有报告路径通过`_AGENT_OUTPUTS`中的键传递: `report_path`, `ocr_investigation_report_path`, `cost_control_report_path`, `report_comparison_path`。

### 总结与控费建议(v1.5.1+)
双模式AI分析文字架构 — 三入口(CLI报告/TUI查询/Excel仪表盘)全覆盖:
- **ComparisonAnalysisGenerator**(`src/analysis/comparison_analysis_generator.py`, 591行): 纯规则对比分析引擎, 5个public方法(generate_comparison_analysis/generate_responsibility_analysis/generate_disease_analysis/generate_executive_summary/generate_policy_level_analysis), 3级阈值(SIGNIFICANT>20%/MODERATE>10%/SLIGHT>5%), 8组因果推断模板
- **SummaryReporter**(`src/report/sections/summary_reporter.py`, ~470行): 6个方法覆盖CLI(format_executive_summary)/控费报告(_format_for_cost_control_report)/综合HTML(format_for_comprehensive_html)三种报告格式
- **PT-016**: 执行摘要增强Prompt(Provider B, temperature=0.3, max_tokens=4096), 输入`rule_based_analysis`, 输出JSON含executive_summary/responsibility_insights/cost_control_recommendations/risk_alerts
- **State字段**: `rule_executive_summary`(规则引擎结果) + `llm_executive_summary`(PT-016 LLM增强结果)
- **优雅降级**: LLM失败→规则引擎结果, 纯规则始终可用

## 反模式(禁止)

- 硬编码阈值(用config.yaml) — 已知100+处待清理(analysis/report/batch模块)
- TODO/FIXME/HACK标记(0处，保持)
- GBK特殊符号: `Y`->`元`, `*`->`-`, checkmark->`[OK]`
- `# type: ignore` — 已知27处(11文件, 多为openpyxl/TypedDict兼容, 不新增)
- `except: pass` — 仅1处(app.py:746 fire-and-forget日志)

## 配置

Config.load()单例。阈值: level_1/2/3(0.10/0.15/0.20), member_stability(0.10), premium_stability(0.15)。

LLM: enabled(总开关), mode(llm_primary/rule_only/hybrid)。.env含API密钥不提交Git。环境变量: GLM_API_KEY, DEEPSEEK_V4_API_KEY, DEEPSEEK_API_KEY。

## Windows兼容

pathlib.Path不拼字符串。_safe_print()(orchestrator_agent), _vprint()(etl_pipeline)。入口文件: sys.stdout.reconfigure(encoding="utf-8")。git checkout恢复文件时需注意编码--PowerShell Out-File会损坏中文，必须用`git checkout <hash> -- <path>`而非管道重定向。

## 关键陷阱

### 批量分析管线(v1.9.71+)

`skills/ghb-cost-control/scripts/batch_doris_pipeline.py` 提供端到端批量分析:

```bash
# 完整管线(分析+Q&A提示词)
python skills/ghb-cost-control/scripts/batch_doris_pipeline.py PID1 PID2 --qa --llm-mode llm_primary

# 仅对已有agent_state生成Q&A
python skills/ghb-cost-control/scripts/batch_doris_pipeline.py PID1 --qa-only
```

**Q&A 子代理配置必须切换模型**: `oh-my-opencode.json` → `categories.writing.model` = `"deepseek/deepseek-v4-pro"`。GLM-4.7 在 32 问答全量输出时会超时(输出 token 限制~8K)。DeepSeek V4 Pro 已验证 2,335行/53.5KB 无超时。

**子代理数据获取必须直读 JSON**: `json.load(UTF-8)` 直接读取 agent_state.json，**严禁**通过 Shell `python -c` 管道提取(GKB 编码丢失中文医院名和嵌套 FWA 数据)。

### Config污染
Phase 1的`DeloitteAnalyzer(llm_enabled=False)`调用`Config.set_llm_enabled(False)`，会**持久修改全局Config单例**。Phase 2必须在Phase 1之后重新加载config.yaml。TUI通过`_restore_llm_config()`解决。同理，TUI的`_run_phase2_tui()`必须调用`load_ocr_data()`加载OCR数据到initial_state，否则OCR报告和对比报告不会生成。

### conftest三重LLM禁用
autouse session fixture:
1. `config._data["llm"]["enabled"] = False`
2. `config._data["llm"]["mode"] = "rule_primary"`
3. 删除GLM_API_KEY/DEEPSEEK_V4_API_KEY/DEEPSEEK_API_KEY环境变量

### 命令名
`ghb-cost-control`(连字符)是正确命令名。`ghb_cost_control`(下划线)不存在。

### prompt_toolkit 3.0.52+ API变更
`prompt_toolkit.keys.Keys`在3.0.52中移除了`Keys.Space`。空格键必须用字符串字面量`" "`绑定: `@kb.add(" ", eager=True)`。`Keys.Any`/`Keys.Enter`/`Keys.Escape`不受影响。TUI中`_wait_for_keypress()`和`_safe_select()`使用`prompt_toolkit`键绑定，**不能**用Rich markup(如`[dim]`)作为`PromptSession.message`参数——`PromptSession`不支持Rich渲染，会原样显示标签文本。app.py/params.py/layout.py中通过`console.print`/`Panel`/`Table`渲染的`[dim]`标签全部有效(Rich原生)。

### Pyright已知问题
1个已知Pyright错误: bs4可选依赖的reportMissingImports。27个`# type: ignore`(11文件, 多为openpyxl/TypedDict兼容, 不新增)。新增代码不允许引入新的`# type: ignore`或reportMissingImports。

### Polars Categorical 类别编码不匹配 (v1.9.71+ 🔴阻断级 → ✅已修复)

Polars `.cast(pl.Categorical)` 对每个 DataFrame 独立编码整数码。训练和预测时相同字符串值可能映射到不同的整数码，导致模型在生产预测中完全失效（验证集 R²=0.91 → 全量预测 R²=0.02）。

**修复方案**：训练时从 pandas DataFrame（LightGBM 原生分类）提取 `{特征: [类别列表]}` 映射并保存到 `models/category_mappings.json`，预测时用 `pd.Categorical(series, categories=saved_cats)` 强制统一编码。11列1042类完整持久化。

**修复后效果**: 全量R² 0.02→0.66(↑32倍), Gini 0.50→0.69, 回测Gini 0.29→0.72(季)。P1季度回测+tweedie_power 1.5→1.2，val_l1改善4%。

### Excel导出
v0.7.0恢复了Excel导出(excel_writer.py + excel_report_exporter.py)并新增Chart.js HTML生成器(excel_chart_html_generator.py)。TUI自动触发双格式导出，CLI通过`--excel`参数触发。

### OCR扫描
OCR document_extractor使用SQLite DB(`data/documents.db`)。扫描命令: `python -m src.tools.document_extractor scan --dirs "data/images" --workers 2`。影像在`data/images/`非`data/target_cases/`。DB可能包含重复document记录。

### DRG分组反向查找(v1.4.0+)
部分保单Excel(如SHAREHOME HK)没有`ICD分组编码`列，只有`ICD分组名称`(disease_name)。DRGGrouper通过`lookup_by_disease_name()`执行中文疾病名称→ICD-10编码反向查找，3层匹配(精确→别名→子串) + 标点归一化(全角逗号/斜杠→顿号)。数据源: `knowledge/icd10_disease_names.yaml`(1,689条3位码+44条分组映射+56条别名)。反向查找标记: `match_type="name_reverse_lookup"`, `is_fuzzy=True`。

**DRG范围码展开**: `_expand_range_code("J00-J99")` → 100个3位码 → 逐一前缀匹配 → 按ADRG去重。适用于有范围码格式ICD分组编码的保单(如贝壳-AI保单1)。

### DRG知识库v2.0(v1.9.71+)
从OpenDRG CHS-DRG 1.1提取完整知识库: drg_adrg.yaml(376 ADRG编码+名称+MDC归类) + drg_procedure_map.yaml(28,540诊断→ADRG映射+9,617手术→ADRG映射) + drg_cc_mcc.yaml(2,329 MCC+5,831 CC编码, 干净导入). DRGGrouper新增ADRG fallback: 诊断编码→ADRG→DRG和手术编码→ADRG→DRG两条间接匹配路径. _load_procedure_map()修复key不匹配. surgery_code→procedure_code修复. 131 DRG测试(57原始+32 ADRG+42高级).

### Skill插件(v1.4.5+)
`skills/ghb-cost-control/`包含8文件: SKILL.md + 5个references/ + 2个scripts/。A1方案(进程内调用): `run_analysis.py`通过`import src.cli:main()`直接调用，非subprocess。需完整项目源码(`src/`+`config.yaml`+`knowledge/`)，包名`src`非`ghb_cost_control`。`extract_agent_capabilities.py`(681行)通过AST+regex动态提取Agent能力。

### CSV批量分析(v1.9.5+)
`FeeDetailCsvLoader`(620行, stdlib csv, 无pandas)从费用明细CSV加载42保单。`BatchAnalysisOrchestrator`(423行)调用DeloitteAnalyzer逐保单Phase1分析。`CrossPolicyReport`(539行)生成跨保单7章汇总报告。

**关键列映射**: CSV的`amount`列是数量/次数(非金额), `clm_acount_amt`→`ClaimData.amount`(核心金融字段)。
**TUI入口**: "CSV批量分析(费用明细按保单拆分)" 替换了原"CSV目录(多CSV合并)"。支持单保单选择和`__ALL__`批量模式。
**数据位置**: `docs/问题反馈/99/20260520/团险_20260520_2025年的42个千万保单赔付情况_费用明细提取/`(7个CSV, ~854MB)。

### Doris数仓批量ETL+分析(v1.9.53+)

完整数据流: **Doris数仓 → CSV下载 → 大表扫描 → 保单提取 → 批量分析 → 跨保单报告**。

**架构**:
- `DorisConnector`(244行): MySQL协议连接Doris FE(9030端口), `export_both_tables()`导出两张表CSV
- `DorisCsvLoader`(463行): analysis+duty双CSV→PolicyAnalysisData, 复用`_CSV_TO_CLAIM_MAP`, `_fix_applicant_name()`投保人修正, `load_policy_from_wide()`宽表+保单级属性
- `DorisBatchExtractor`(362行): 单次扫描大CSV(2.4M+23.2M行), 批量提取per-policy CSV对, 64MB缓冲+scan_cache
- `DorisBatchAnalysisOrchestrator`(411行): 批量Phase1分析, ThreadPoolExecutor并行(默认4线程), agent_state保存

**TUI菜单**: `_doris_batch_menu()` 6项:
| 📥 从Doris数仓下载CSV | 密码输入→连接测试→WHERE过滤→大表导出(10GB, 10-30分钟) |
| 🔍 扫描Doris大表可用保单 | Rich Progress + 24h缓存(.scan_cache.json) + 分页表格(每页15行, 首页/末页/跳转) |
| 📦 批量提取保单CSV | 选择数量(全部/10/20/50/100/自定义) + 双阶段进度条 |
| 🚀 批量分析已提取保单 | 轻量扫描 + 4线程并行 + _batch_post_menu钻取 + CrossPolicyReport |
| 📋 查看已提取保单列表 | 轻量扫描(仅读文件大小, 不加载duty CSV) |

**关键陷阱**:
- **`gc`变量名冲突**: `for gc in ...`会覆盖`gc`模块导致`gc.collect()`崩溃。用`grp_no`。
- **文件占用**: 写入前`unlink(missing_ok=True)`清理旧文件, OSError时跳过继续不中断批量。
- **CSV大缓冲**: `open(..., buffering=64*1024*1024)`对8.78GB duty CSV提速~30%。
- **扫描缓存**: `.scan_cache.json`24h有效期, 用户可"清除缓存并重新扫描"。limit=0(回车)为不限全量。
- **分页表格**: `_render_paginated_table()`超过page_size时自动分页, 导航: ⏫首页/⬆上/⬇下/⏬末页/跳转。
- **并行安全**: 进度回调用`threading.Lock()`保护, 结果按原始保单顺序返回。

### TUI JSON加载查询(v1.9.5+fix)
保存`agent_state.json`时序列化`_policy_summary`(14字段: policy_id/name/total_paid/premium/loss_ratio/members/responsibilities YoY/diseases YoY等)。5个display函数增加`_policy_summary` fallback: _display_final_summary, _display_policy_comparison, _display_yoy_comparison, _display_uw_lr_detail, _display_city_analysis(降级提示)。旧JSON文件无此字段需重新运行分析。

### TUI Agent耗时实时刷新(v1.9.7)
`buffer.py`新增`agent_start_times` dict, `update_agent_status(status="running")`时记录`time.time()`。`layout.py`的`_update_progress()`对running Agent动态计算elapsed(4Hz Live刷新), completed保留最终耗时。超过60s显示`XmY.Ys`格式。

### 执行日志监控(v1.9.8)
`src/services/execution_logger.py`: ExecutionLogger单例, 按日JSONL归档(`logs/ghb_execution_YYYY-MM-DD.jsonl`), threading.Lock线程安全, 90天自动清理。`src/logging_config.py`新增`setup_file_logging(TimedRotatingFileHandler, when='midnight', backupCount=90)`。TUI第11组菜单"执行监控"(5函数: 最近执行/按日查询/按保单查询/Agent统计/日志清理)。CLI `logs`子命令(`--date/--policy/--status/--days/--limit`)。

### 路径A下钻(v1.9.8)
`src/analysis/path_a_drilldown.py`: PathADrilldownAnalyzer, PathALevel dataclass, 4层嵌套(责任→疾病TOP N→医院TOP5→二级责任TOP5), 每层8指标+同比+异常等级。`src/agents/path_a_drilldown_agent.py`: Agent封装。AgentState新增`path_a_drilldown`字段。workflow.py接入路径A节点。Excel `6_A_医院_二级责任` Sheet对齐参考Excel命名(v1.9.36)。

### 报告增强(v1.9.8)
6个新SectionReporter: `responsibility_detail_reporter`(P0-2责任8指标), `strategy_matching_reporter`(P0-3策略匹配), `path_a_drilldown_reporter`(P0-1路径A下钻), `high_claim_reporter`(P1-2高赔付TOP10)。`src/analysis/strategy_matcher.py`: StrategyMatcher+6类策略模板。`src/analysis/risk_dimension_assessor.py`: RiskDimensionAssessor 4维度(就诊合理性/过度治疗/过度用药/费用结构)。FWA报告新增医院欺诈聚合结果(hospital_fraud_aggregation字段, 按医院维度汇总欺诈行为)。

### SVG图表+单体HTML(v1.9.8)
`src/report/svg_charts.py`: 内联SVG饼图/柱图(svg_pie_chart, svg_bar_chart, render_member_profile_svg)。`src/report/standalone_html.py`: StandaloneHtmlReport单体HTML报告(无CDN依赖, 内联SVG+CSS, 适合邮件发送)。v1.9.37全面优化: 11节+TOC双列目录+中文化+品牌横幅+返回目录/顶部+@media print。**注意: 当前为死代码(未被调用)**。

### 综合可视化HTML报告(v1.9.38+)
`ComprehensiveReportGenerator`(`src/report/comprehensive_report_generator.py`, ~6220行): TUI阶段四自动生成, 32章节(核心摘要/评分/保单数据/异常检测/环比/路径A+B/IBNR/健康评分/控费/归因/集中就医/会员风险/责任详细/策略匹配/控费追踪/FWA/安全调查/DRG/疾病费用/医院/交叉维度+标杆+既往症/会员画像+明细/风险地图/24月走势预测/统一预警/报告索引)。v1.9.44优化: 月度环比TOP20维度截断+路径B(10分组×15子项)截断+路径A(5×5×5×3×5)截断, 报告体积586KB→254KB(-57%)。v1.9.62新增: 风险地图四维度(疾病/医院/会员/投保单位) + 24月走势预测(数据不足时显示提示) + 统一预警面板。

### TUI查询菜单修复(v1.9.29~v1.9.35)
7个display函数19处键名不匹配修复。医院类型分布3层数据源(_policy_summary预计算/cost_standards.yaml/名称启发式)。TUI菜单12组66项分组优化(拆案件与关联/合并疾病+费用/移动错配项/中文直写)。Rich品牌横幅(GHB+版本号+青色边框)。

### TUI系统信息菜单(v1.9.39)
新增第13组"系统信息"(3项: 版本信息/查看README/查看AGENTS.md)。`_display_version_info()`显示系统版本/Python/LangGraph/LLM配置。`_display_readme()`和`_display_agents_md()`使用`_display_markdown_paginated()`分页渲染Markdown(60行/页, j/k翻页+鼠标滚动)。

### TUI菜单emoji图标+Esc返回+操作提示(v1.9.43)
全菜单(17处questionary.select)统一使用`_safe_select()`辅助函数。emoji图标覆盖主菜单(🚀📂📦ℹ️📖📄📋🚪)+查询类别(⭐📋🔍🏥🏛️👥💰🛡️📈)+系统日志+批量管理+翻页。`instruction`参数显示底部操作提示"(↑↓ 选择 Enter 确认 Esc 返回上级)"。Esc键注入`prompt_toolkit`键绑定返回`_BACK_SENTINEL`哨兵值。返回选项统一为"← 返回"/"← 返回上级"/"← 返回主菜单"。

### TUI主菜单系统管理一级入口(v1.9.42)
查询菜单10组→9组56项。系统管理(14项)从查询子菜单提升到主菜单一级入口(Separator分组: 分析/批量/系统)。查询菜单"Agent执行详情"移入"核心摘要"(5→6项)。主菜单新增: 版本信息/查看README/查看AGENTS.md/执行日志查询(调用`_system_log_menu()`子菜单7项)。

### TUI菜单分组优化(v1.9.40)
13组→10组，职责单一化。核心变化: "综合概览"→"核心摘要"(5项黄金首屏) + "控费管理"+"交叉维度"+"预警监控"+"预授权"→"控费与管控"(7项) + "案件关联"+"安全审核"→"安全与调查"(7项) + "预测评估"+OCR→"预测与标杆"(5项) + "系统管理"+"系统信息"→"系统管理"(14项)。菜单总计10组69项。

### Excel命名对齐(v1.9.36)
路径A第4层: `6_A_医院_费用` → `6_A_医院_二级责任`, 列名"费用类型"→"二级责任名称", 对齐参考Excel。PathA docstring "Fee Type"→"Level-2 Responsibility"。StandaloneHtmlReport新增策略匹配section(P0-3)。

### 参考代码v2对齐(v1.9.56)

对齐`docs/问题反馈/14/数据分析python代码-doris数据源v2.txt`全部分析逻辑。三阶段: Phase A(数据层) → Phase B(分析层) → Phase C(优化层)。

**Phase A 数据层**:
- `ClaimData` +10字段: gender/birthday/nationality/relationship(会员人口统计) + cont_valid_date/pass_months/annual_prem/pass_prem/rnew_flag(保单级属性) + duty_amount(duty级金额)
- `ClaimData` +work_addr字段: 常驻城市(ETL映射`常驻城市`列, CSV映射含`常驻城市分布`别名)
- `PolicyAnalysisData` +2字段: pass_months(int, 从claims首行), est_loss_ratio(float, 从EST_LOSS_RATIO列)
- `_CSV_TO_CLAIM_MAP` +10映射(含大写列名兼容INSURED_GENDER/CUST_BIRTHDAY等)
- `FeeDetailCsvLoader._row_to_claim()` +10字段赋值
- `DorisCsvLoader` 新增`_fix_applicant_name()`投保人名称修正(按appnt_no取最新cont_valid_date)
- `DorisCsvLoader.load_policy_from_wide()` 提取保单级属性(effective_date/renewal_type/earned_premium/pass_months/est_loss_ratio)
- ETL会员城市fallback增强: `_fill_member_city()`新增work_addr(常驻城市)→member.city回填

**Phase B 分析层**:
- 路径A第4层: `fee_item_type`优先, fallback `lob_level2_name`
- 路径B第4层: `fee_item_type`优先, fallback `lob_level2_name`
- `LevelResult` +2字段: diff_ratio_to_total/best_rate_metric
- `SeverityLevel` 新增 `HIGH_NEW`("high_new", 历史无数据=高风险)
- 会员画像城市分布增强(city fallback: location→acci_city)
- Top30会员行为画像增强: top_fee_by_count(Top5按次数)/top_diagnoses(Top3诊断)/outpatient_visits/outpatient_monthly_avg/outpatient_paid/paid_share

**Phase C 优化层**:
- 路径A结果排序: RESPONSIBILITY_SORT_ORDER映射表(门诊1/住院2/生育3/牙科4/体检5/眼科6)
- 异常聚集3维度调查: `claim_correlation_analyzer.py`增加(医院+ICD分组+日期)聚集, unique_members≥3触发same_day_aggregation模式

### 以诊代检多信号增强(P3-1, v1.9.67)
`src/analysis/fwa_engine.py`: FWA-A019规则重构为4信号评分(keyword + fee ratio + ICD-10 contra-indicated + repeat pattern), 阈值≥3触发。关键词库16→34项, ICD-10检查正当性索引从`medical_rationality_rules.yaml`加载(25+疾病组)。

### 利益输送检测(P3-2, v1.9.67)
`src/analysis/benefit_transfer_analyzer.py` (~272行): 3种检测模式(医院间Jaccard重叠/会员迁移链/投保单位集中漏斗)。医院名称标准化。`knowledge/benefit_transfer_rules.yaml`配置阈值(Jaccard 0.15/集中度0.6/迁移90天)。`src/agents/benefit_transfer_agent.py` (~82行)封装Agent, AgentState新增`benefit_transfer_result`字段。

### 投保单位代理指标+workflow修复(P3-3, v1.9.67)
`src/analysis/policy_unit_analyzer.py`: 修复`occupation_mismatch_flags`类型不匹配(List[Dict] vs .items()), `high_risk_score_members`从FWA HIGH/CRITICAL匹配推导(原硬编码0), 新增`_compute_turnover_rate()`成员流失率代理指标。`src/agents/workflow.py`: `policy_unit`从并行组B移至fwa_analysis后的顺序链(修复race condition, 需FWA结果)。

### 投保单位同比+同级对比+医院真实性聚合(P4, v1.9.68)
`src/analysis/policy_unit_analyzer.py`: 3个新方法 — `_compute_unit_yoy()`(5指标同比+30%异常阈值) + `_compute_unit_peer_comparison()`(Top5医院同等级peer对比,4级偏差分类) + `_aggregate_hospital_authenticity()`(身份F020-F022+事故F004/F007/F008按医院聚合,风险比率分类)。AgentState新增3字段(policy_unit_yoy/policy_unit_peer_comparison/hospital_authenticity)。`policy_unit_agent.py`新增previous_claims和hospital_fee_anomaly输入传递+3个子结果独立state字段。

### 非本人就诊深度增强(P4-3, v1.9.68)
`src/analysis/fwa_engine.py`: `_agg_treatment_pattern_mismatch()` 4信号评分(age_treatment_mismatch/gender_treatment_mismatch/chronic_disease_onset/impossible_distance), ≥2触发。A021规则(identity_suspicion)。关键词: 儿科治疗(Z23/Z00-Z02)用于>18岁/妇科治疗用于男性/慢病突发/同城双院。

### 伪造事故深度增强(P4-4, v1.9.68)
`src/analysis/fwa_engine.py`: `_agg_accident_diagnosis_mismatch()` 3信号评分(injury_without_treatment/acute_delayed_treatment/weekend_accident_pattern), ≥2触发。F023规则(accident_fabrication)。创伤ICD-10(S00-T88/V00-V99/W00-X59)无对应治疗/急症延迟>3天/事故≥80%周末。

### FWA规则扩展至61条(v1.9.68)
`knowledge/fwa_rules.yaml`: 59→61条(+F023事故-诊断不一致+A021治疗模式不匹配)。总计: 22欺诈+18浪费+21滥用。`src/analysis/fwa_engine.py`: 2个新聚合器+2个dispatch分支。

### 显示层覆盖修复(v1.9.69)
P4分析结果在CLI/TUI/综合报表中的显示覆盖修复:
- `src/agents/workflow.py`: `_AGENT_OUTPUTS` policy_unit +3字段(policy_unit_yoy/policy_unit_peer_comparison/hospital_authenticity) — 修复并行模式数据丢失
- `src/tui/app.py` `_display_policy_unit_analysis()`: +3面板(投保单位同比/同级对比/医院真实性)
- `src/tui/app.py` `_display_risk_map_policy_unit()`: +FWA-A021/F023规则计数 +3风险点行 +3详情面板(同比/同级对比/真实性)
- `src/report/comprehensive_report_generator.py` `_chapter_risk_map()`: `_get_fwa_identity_counts()` +A021/F023 +3 KPI +4表格行 +3详情子章节

### CLI修复+技能优化+批量管线(v1.9.71)

**CLI 3处bug修复**:
- `src/cli.py`: `_save_agent_state()`(line 457-586) Phase 2/compare 保存agent_state.json + 输出目录命名`{policy_id}_{timestamp}/`(line 796-797) + LLM模式验证

**技能优化**:
- `skills/risk-map-analyzer/` v2.3.0: 问询计数修正(33→32) + Q2.5自适应键提取(CR4/cr4双键名+运行模式检测) + Q5.9最低内容深度(≥900行/≥200字) + Q8 agent_state结构差异(CLI/TUI/compare) + Q1强制直读JSON(禁止Shell Python管道) + FWA 57→61 + 会员维度映射修复
- `skills/ghb-cost-control/` v1.9.71: CLI输出目录命名 + `_save_agent_state()`文档 + LLM SDK缺失自动降级 + CLI vs TUI差异表 + 知识库YAML 13→20 + Doris批量管线文档 + LLM模式选择表 + Q&A子代理模型配置
- 两技能同步至 `~/.agents/skills/`

**批量分析管线**:
- `skills/ghb-cost-control/scripts/batch_doris_pipeline.py`: 传保单号即可自动完成完整管线(Phase2 Agent分析 → agent_state → 综合HTML → Q&A提示词)
- CLI命令: `python skills/ghb-cost-control/scripts/batch_doris_pipeline.py PID1 PID2 --qa --llm-mode llm_primary`
- Q&A委托: `task(category="writing", load_skills=["risk-map-analyzer"])` → writing子代理直读agent_state.json生成32问答
- 10保单实测: ~3.5h, agent_state 80MB, Q&A 763KB, 0错误

**模型配置**:
- `oh-my-opencode.json`: `categories.writing.model` 从 `zhipuai-coding-plan/glm-4.7` 改为 `deepseek/deepseek-v4-pro` (解决子代理32问答输出超限)
- LLM模式: `--llm-mode llm_primary`(121K tokens,PT-001~016全部) / `rule_only`(0 tokens,纯规则) / `hybrid`(50K tokens)

**综合报表**:
- `src/report/comprehensive_report_generator.py`: +70行 ICD-10疾病名反向查找(`_load_disease_name_lookup` + `_resolve_disease_name`), Ch22疾病费用章节6处显示修复

**风险地图Q&A**: GP1260200000000002 ×3次运行 + Doris 10保单批量 ×2轮(rule_only + llm_primary)

### 版本同步
`src/__init__.py`的`__version__`需与`pyproject.toml`的`version`保持同步。每次push更新版本号时两个文件都要改。

### 每次push更新版本号
pyproject.toml的version字段在每次push时更新。v1.9.71。
