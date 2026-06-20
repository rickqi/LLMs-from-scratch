# LLMs-from-scratch 项目群 — 待执行计划

> 2026-06-18 | 4 projects | Coverage: 93%

## 优先级排序

| # | 项目 | 行动 | 工作量 | 预期效果 |
|---|------|------|--------|---------|
| **P0** 🔥 | **医疗文本** | **DPO 偏好对齐** | 3h | Ch7→100%, 教程覆盖→97% | ✅ **完成** |
| P1 | Kronos | GRPO Sharpe优化 | 4h | 教程覆盖→100% | 🔄 执行中 |
| P2 | 法规 | DPO (需先构造数据) | 5h | 对齐法规文本质量 |
| P3 | 控费 | 断点续传 + COS备份 | 2h | 补齐缺失能力 |
| P4 | 控费 | Optuna→Kronos超参优化 | 3h | 自动化调参 |

## 关键里程碑

```
当前:  教程覆盖 93%, 生产就绪 Kronos
  ↓ P0
DPO完成: 教程覆盖 97%, 医疗项目对齐
  ↓ P1
GRPO完成: 教程覆盖 100%, Kronos直接优化Sharpe
  ↓ P2-P4
补齐: 法规DPO + 控费续传 + 超参自动化
```

## P0 执行命令

```bash
cd projects/chinese-medical-text-generation
# Step 1: 构造偏好数据
python scripts/prepare_dpo_data.py
# Step 2: DPO 训练
python train_dpo.py --base_model ./output_inst_v3/best_model --preference_data ./data/dpo_pairs.json
# Step 3: 评估
python scripts/eval_compare.py --model_before ./output_inst_v3 --model_after ./output_dpo
```
