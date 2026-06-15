"""
数据准备脚本: 中文医学诊疗指南文本生成
========================================
支持两种模式:
  1. 真实数据模式: python data_prep.py --data_dir ./raw_data --output_dir ./data
  2. 样本数据模式: python data_prep.py --sample --output_dir ./data  (自动生成100篇样本)

数据格式: 接受 .md 文件目录, 每个文件是一份医学指南
"""

import re
import os
import sys
import argparse
import logging
import random
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# ============================================================
#  数据清洗
# ============================================================

def clean_medical_text(text: str) -> str:
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'\!\[.*?\]\(.*?\)', '', text)
    text = re.sub(r'ISBN[^\n]*', '', text)
    text = re.sub(r'page=\d+[^)]*\)', '', text)
    text = re.sub(r'^#{1,6}\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'\n{4,}', '\n\n', text)
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    return '\n'.join(lines)


def load_raw_data(data_dir: str) -> list[str]:
    data_dir = Path(data_dir)
    files = sorted(data_dir.rglob("*.md"))
    logger.info(f"发现 {len(files)} 个 .md 文件")

    texts = []
    for fpath in files:
        try:
            raw = fpath.read_text(encoding="utf-8", errors="ignore")
            cleaned = clean_medical_text(raw)
            if len(cleaned) > 50:
                texts.append(cleaned)
        except Exception as e:
            logger.warning(f"读取 {fpath} 失败: {e}")

    logger.info(f"清洗后有效文本: {len(texts)} 篇, 总字符数: {sum(len(t) for t in texts):,}")
    return texts


# ============================================================
#  样本数据生成 (无真实数据时的降级方案)
# ============================================================

TEMPLATES = {
    "临床表现": [
        "临床表现：患者{{gender}}，{{age}}岁，因\"{{symptom1}}伴{{symptom2}}{{duration}}\"入院。"
        "患者{{duration}}前无明显诱因出现{{symptom1}}，呈{{quality}}，伴{{symptom2}}，"
        "无{{neg_symptom}}等不适。查体：体温{{temp}}℃，脉搏{{pulse}}次/分，"
        "呼吸{{breath}}次/分，血压{{bp}}mmHg。{{body_part}}{{finding}}。",
    ],
    "诊断依据": [
        "诊断依据：{{num}}. 临床表现：患者有{{symptom1}}和{{symptom2}}的典型表现。"
        "{{num}}. 实验室检查：血常规示白细胞计数{{wbc}}×10⁹/L，"
        "中性粒细胞百分比{{neut}}%，C反应蛋白{{crp}}mg/L。"
        "{{num}}. 影像学检查：{{imaging_type}}示{{body_part}}{{imaging_finding}}。"
        "{{num}}. 病理学检查：{{pathology_type}}提示{{pathology_finding}}，"
        "免疫组化：{{ihc}}。综合以上，{{diagnosis}}诊断明确。",
    ],
    "治疗方案": [
        "治疗方案：{{num}}. {{treatment_type}}治疗：{{drug}}，{{dosage}}，"
        "{{route}}，{{frequency}}。{{num}}. 对症支持治疗："
        "根据患者{{symptom1}}情况，给予{{support_treatment}}。"
        "{{num}}. 特殊治疗：{{special_treatment}}。治疗期间需密切监测"
        "{{monitor_params}}。",
    ],
    "预后判断": [
        "预后判断：该患者为{{stage}}期{{diagnosis}}，病理分级{{grade}}。"
        "根据{{classification}}分期标准，{{tnm_stage}}。"
        "综合评估：{{risk}}风险组。五年生存率约为{{survival_rate}}%，"
        "中位生存时间约{{median_survival}}个月。建议{{follow_up}}。",
    ],
    "鉴别诊断": [
        "鉴别诊断：{{num}}. {{disease1}}：{{diff1}}。"
        "{{num}}. {{disease2}}：{{diff2}}。"
        "{{num}}. {{disease3}}：{{diff3}}。"
        "通过{{method}}可资鉴别。",
    ],
    "用药原则": [
        "用药原则：{{num}}. 严格掌握{{drug_type}}的适应证和禁忌证。"
        "{{num}}. 注意药物相互作用：{{drug}}与{{drug2}}联合使用时"
        "{{interaction}}。{{num}}. 个体化给药：根据患者{{factor}}调整剂量。"
        "{{num}}. 监测不良反应：定期复查{{monitor_items}}。",
    ],
}

FILLERS = {
    "gender": ["男", "女"],
    "age": ["45", "52", "63", "38", "71", "56", "47", "68"],
    "symptom1": ["发热", "咳嗽", "腹痛", "头痛", "胸痛", "恶心", "乏力", "体重下降",
                  "呼吸困难", "关节疼痛", "皮肤黄染", "吞咽困难", "便血", "呕血"],
    "symptom2": ["呕吐", "咳痰", "腹泻", "心悸", "盗汗", "食欲减退", "失眠", "水肿",
                  "皮疹", "鼻衄", "血尿", "便秘", "头晕", "抽搐"],
    "duration": ["1周", "3天", "2个月", "半年", "1个月", "10天", "2周", "5天"],
    "quality": ["持续性钝痛", "间歇性绞痛", "进行性加重", "反复发作性", "隐痛",
                "剧烈撕裂样疼痛", "针刺样疼痛", "烧灼感"],
    "neg_symptom": ["发热", "呕吐", "呼吸困难", "休克", "意识障碍"],
    "temp": ["36.5", "37.8", "38.2", "39.1", "36.8", "37.2"],
    "pulse": ["72", "88", "96", "105", "78", "82"],
    "breath": ["16", "18", "20", "22", "17", "19"],
    "bp": ["120/80", "130/85", "140/90", "110/70", "125/82"],
    "body_part": ["腹部", "胸部", "颈部", "肝区", "脾区", "心前区"],
    "finding": ["有压痛，无反跳痛", "可触及肿块", "听诊呼吸音粗", "叩诊浊音",
                "未见明显异常", "可闻及湿啰音"],
    "num": ["1", "2", "3", "4"],
    "wbc": ["8.5", "12.3", "6.7", "15.8", "9.2"],
    "neut": ["72.5", "85.3", "68.9", "78.1", "91.2"],
    "crp": ["25", "48", "12", "67", "33"],
    "imaging_type": ["胸部CT", "腹部超声", "MRI平扫+增强", "PET-CT", "X线胸片"],
    "imaging_finding": ["见占位性病变", "示多发结节影", "可见淋巴结增大",
                        "示弥漫性病变", "见不规则高密度影"],
    "pathology_type": ["穿刺活检", "手术切除标本", "支气管镜活检", "胃镜活检"],
    "pathology_finding": ["腺癌", "鳞状细胞癌", "小细胞癌", "恶性淋巴瘤",
                          "低分化癌", "转移性腺癌"],
    "ihc": ["CK(+), TTF-1(+), NapsinA(+)", "CK5/6(+), P63(+), Ki-67(60%+)",
            "CK20(-), CDX2(+), SATB2(+)", "LCA(+), CD20(+), CD3(-)"],
    "diagnosis": ["原发性支气管肺癌", "胃腺癌", "结直肠癌", "肝细胞癌",
                  "食管鳞状细胞癌", "非霍奇金淋巴瘤"],
    "treatment_type": ["手术", "化学", "放射", "靶向", "免疫"],
    "drug": ["顺铂", "卡铂", "紫杉醇", "吉西他滨", "奥沙利铂", "氟尿嘧啶",
             "贝伐珠单抗", "利妥昔单抗", "厄洛替尼", "克唑替尼"],
    "dosage": ["75mg/m²", "AUC=5", "175mg/m²", "1000mg/m²", "130mg/m²"],
    "route": ["静脉滴注", "口服", "静脉注射", "皮下注射"],
    "frequency": ["每3周一次", "每日一次", "每周一次", "每4周一次"],
    "support_treatment": ["给予止吐、补液等", "营养支持治疗", "镇痛治疗",
                          "预防性升白细胞治疗", "保肝、保肾治疗"],
    "special_treatment": ["立体定向放疗（SBRT）", "射频消融治疗",
                          "放射性粒子植入", "介入化疗栓塞（TACE）"],
    "monitor_params": ["血常规、肝肾功能", "心电图、心肌酶谱", "肿瘤标志物变化",
                       "影像学评估疗效", "不良反应"],
    "stage": ["I", "II", "III", "IV"],
    "grade": ["中分化", "低分化", "高分化", "中-低分化"],
    "classification": ["TNM", "FIGO", "Ann Arbor", "Lugano"],
    "tnm_stage": ["T2N1M0", "T3N2M0", "T1N0M0", "T4N2M1", "T3N1M0"],
    "risk": ["低", "中", "中-高", "高"],
    "survival_rate": ["30", "45", "60", "75", "85", "20"],
    "median_survival": ["12", "18", "24", "36", "48", "8"],
    "follow_up": ["每3个月复查一次，连续2年", "每6个月全面评估一次",
                  "定期复查肿瘤标志物及影像学检查", "终身随访"],
    "disease1": ["肺炎", "肺结核", "间质性肺病", "肺脓肿"],
    "disease2": ["食管癌", "淋巴瘤", "纵隔肿瘤", "胸腺瘤"],
    "disease3": ["肺转移瘤", "肺栓塞", "上腔静脉综合征", "结节病"],
    "diff1": ["临床表现相似，但影像学有特征性改变", "可通过病原学检查予以鉴别",
              "病史及免疫学指标有助于区分", "治疗后病灶变化可资鉴别"],
    "diff2": ["症状位置不同", "淋巴瘤常有全身淋巴结肿大",
              "影像学显示不同来源", "发病年龄和性别分布不同"],
    "diff3": ["已知原发肿瘤病史", "突发胸痛、低氧血症",
              "特征性临床表现及影像学", "呼吸道及全身症状特点不同"],
    "method": ["病理学检查", "免疫组化染色", "基因检测", "影像学动态观察",
               "多学科讨论（MDT）"],
    "drug_type": ["抗肿瘤药物", "化疗药物", "靶向药物", "免疫检查点抑制剂"],
    "drug2": ["卡铂", "紫杉醇", "吉西他滨", "培美曲塞", "伊立替康"],
    "interaction": ["可能增加骨髓抑制风险，需监测血常规",
                    "需注意肝肾毒性叠加，定期监测肝肾功能",
                    "可能增加神经毒性，注意末梢神经症状",
                    "需关注心脏毒性风险，治疗前及治疗中监测心功能"],
    "factor": ["肝肾功能", "体表面积", "基因检测结果", "既往治疗反应",
               "合并症情况", "年龄和一般状况"],
    "monitor_items": ["血常规", "肝肾功能", "心电图", "血清肿瘤标志物",
                      "影像学检查", "超声心动图"],
}


def _fill(text: str) -> str:
    filled = text
    while "{{" in filled:
        start = filled.index("{{")
        end = filled.index("}}") + 2
        key = filled[start+2:end-2]
        if key in FILLERS:
            replacement = random.choice(FILLERS[key])
            filled = filled[:start] + replacement + filled[end:]
        else:
            filled = filled[:start] + f"[{key}]" + filled[end:]
    return filled


def generate_sample_data(num_samples: int = 100) -> list[str]:
    logger.info(f"生成 {num_samples} 篇模拟医学文本 (模板填充)...")
    texts = []
    category_names = list(TEMPLATES.keys())
    for _ in range(num_samples):
        category = random.choice(category_names)
        template = random.choice(TEMPLATES[category])
        text = f"# {category}\n\n{_fill(template)}"
        texts.append(text)
    logger.info(f"生成完成: {len(texts)} 篇, 总字符数: {sum(len(t) for t in texts):,}")
    return texts


# ============================================================
#  数据分割与保存
# ============================================================

def split_train_val(texts: list[str], val_ratio: float = 0.05, seed: int = 42):
    rng = random.Random(seed)
    shuffled = list(texts)
    rng.shuffle(shuffled)
    split = int(len(shuffled) * (1 - val_ratio))
    return shuffled[:split], shuffled[split:]


def save_data(texts: list[str], output_path: str):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    content = "\n\n===SEP===\n\n".join(texts)
    output_path.write_text(content, encoding="utf-8")
    logger.info(f"保存到 {output_path}, 共 {len(texts)} 篇, {output_path.stat().st_size:,} 字节")


# ============================================================
#  统计报告
# ============================================================

def show_stats(texts: list[str], label: str):
    lengths = [len(t) for t in texts]
    logger.info(f"\n{'='*50}")
    logger.info(f"  {label}")
    logger.info(f"  总文本数:   {len(texts)}")
    logger.info(f"  总字符数:   {sum(lengths):,}")
    logger.info(f"  平均字符数: {sum(lengths)//max(len(texts),1):,}")
    logger.info(f"  最短/最长:  {min(lengths)} / {max(lengths)}")
    logger.info(f"{'='*50}")


# ============================================================
#  主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="准备中文医学文本数据")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--data_dir", type=str, default=None, help="原始 .md 文件目录")
    group.add_argument("--sample", action="store_true", help="生成模拟医学数据 (无需原始数据)")
    parser.add_argument("--num_samples", type=int, default=100, help="模拟数据生成篇数")
    parser.add_argument("--output_dir", type=str, default="./data", help="输出目录")
    parser.add_argument("--val_ratio", type=float, default=0.05, help="验证集比例")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()

    random.seed(args.seed)

    # 获取数据
    if args.sample:
        texts = generate_sample_data(args.num_samples)
    elif args.data_dir:
        texts = load_raw_data(args.data_dir)
    else:
        logger.warning("未指定数据来源, 自动使用样本数据模式 (100篇)")
        texts = generate_sample_data(100)

    if not texts:
        logger.error("未找到有效文本, 请检查数据来源")
        sys.exit(1)

    show_stats(texts, "原始数据")

    # 分割
    train_texts, val_texts = split_train_val(texts, args.val_ratio, args.seed)
    show_stats(train_texts, "训练集")
    show_stats(val_texts, "验证集")

    # 保存
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_data(train_texts, output_dir / "train.txt")
    save_data(val_texts, output_dir / "val.txt")

    logger.info(f"\n数据准备完成! 下一步: python train_qwen_lora.py --data_dir {output_dir}")


if __name__ == "__main__":
    main()
