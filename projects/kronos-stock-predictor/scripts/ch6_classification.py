"""
Ch6 分类微调 — 教程覆盖补充

映射教程概念:
  GPT预训练模型 → LSTM回归模型 (已训练)
  替换输出头     → reg_head → cls_head (2分类)
  冻结骨干       → freeze LSTM + MLP layers
  下游任务       → 68半导体未来10日涨跌
  评估           → Accuracy / F1 / Confusion Matrix
"""

import pickle, torch, numpy as np, logging, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model.lstm_model import LSTMModel, FEATURES
from torch.utils.data import DataLoader, Dataset

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger()
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


class ClassifyDataset(Dataset):
    """Ch6 下游分类数据集: 未来10日涨(1)/跌(0)"""
    def __init__(self, data, lookback=180, pred_len=10, max_samples=15000):
        self.samples = []
        for sym, df in data.items():
            vals = df[FEATURES].values.astype(np.float32)
            n = len(vals)
            for i in range(n - lookback - pred_len):
                x = vals[i : i + lookback]
                mu, std = x.mean(0), x.std(0) + 1e-5
                x_norm = np.clip((x - mu) / std, -5, 5)
                future = vals[i + lookback + pred_len - 1, 3]
                current = vals[i + lookback - 1, 3]
                y = 1 if future > current else 0
                self.samples.append((x_norm.astype(np.float32), y))
        self.n = min(len(self.samples), max_samples)

    def __len__(self): return self.n
    def __getitem__(self, i):
        x, y = self.samples[i % len(self.samples)]
        return torch.FloatTensor(x), torch.LongTensor([y])


def freeze_backbone(model: LSTMModel):
    """Ch6: 冻结预训练骨干，仅训练分类头"""
    for name, param in model.named_parameters():
        if "head" not in name:
            param.requires_grad = False
    frozen = sum(1 for p in model.parameters() if not p.requires_grad)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"  Frozen params: {frozen}, Trainable: {trainable}")


def train_and_eval(data, freeze=True):
    ds_train = ClassifyDataset(data["train"])
    ds_val = ClassifyDataset(data["val"], max_samples=3000)
    tl = DataLoader(ds_train, 128, True, drop_last=True)
    vl = DataLoader(ds_val, 128)

    # Step 1: 加载预训练模型
    model = LSTMModel(hidden=128, num_layers=2).to(DEVICE)
    if Path("outputs/production_model.pt").exists():
        model.load_state_dict(torch.load("outputs/production_model.pt", map_location=DEVICE))

    # Step 2: 替换输出头 (regression → classification)
    in_features = model.head[0].in_features
    model.head = torch.nn.Sequential(
        torch.nn.Linear(in_features, in_features // 2),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.2),
        torch.nn.Linear(in_features // 2, 2),  # 2分类
    ).to(DEVICE)

    # Step 3: 冻结骨干
    if freeze:
        freeze_backbone(model)

    # Step 4: 微调训练
    opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()
    best_acc, best_state = 0, None

    for e in range(8):
        model.train()
        for x, y in tl:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            loss_fn(model(x), y.squeeze()).backward()
            opt.step()
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in vl:
                x, y = x.to(DEVICE), y.to(DEVICE)
                preds = model(x).argmax(-1)
                correct += (preds == y.squeeze()).sum().item()
                total += y.size(0)
        acc = correct / total
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if e % 2 == 0:
            logger.info(f"  E{e+1}/8: val_acc={acc:.3f}")

    model.load_state_dict(best_state)
    model.eval()

    # Step 5: 测试集评估
    test_subset = {k: data["test"][k] for k in list(data["test"].keys())[:30]}
    ds_test = ClassifyDataset(test_subset, max_samples=5000)
    test_loader = DataLoader(ds_test, 128)
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(DEVICE)
            preds = model(x).argmax(-1).cpu()
            all_preds.extend(preds.tolist())
            all_labels.extend(y.squeeze().tolist())

    preds = np.array(all_preds)
    labels = np.array(all_labels)
    acc = np.mean(preds == labels)
    tp = np.sum((preds == 1) & (labels == 1))
    fp = np.sum((preds == 1) & (labels == 0))
    fn = np.sum((preds == 0) & (labels == 1))
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return {
        "accuracy": round(acc, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "confusion": {"TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(np.sum((preds == 0) & (labels == 0)))},
        "n_test": len(labels),
    }


if __name__ == "__main__":
    data = {m: pickle.load(open(f"data/processed_real/{m}_data.pkl", "rb")) for m in ["train", "val", "test"]}

    logger.info("=== Ch6: 分类微调 (冻结骨干) ===")
    r1 = train_and_eval(data, freeze=True)
    logger.info(f"  Accuracy:  {r1['accuracy']:.2%}")
    logger.info(f"  Precision: {r1['precision']:.2%}")
    logger.info(f"  Recall:    {r1['recall']:.2%}")
    logger.info(f"  F1:        {r1['f1']:.2%}")
    logger.info(f"  Confusion: {r1['confusion']}")

    logger.info("\n=== Ch6: 全参数微调 (对比) ===")
    r2 = train_and_eval(data, freeze=False)
    logger.info(f"  Accuracy:  {r2['accuracy']:.2%}")
    logger.info(f"  F1:        {r2['f1']:.2%}")

    logger.info(f"\n=== Ch6 教程覆盖完成 ===")
    logger.info(f"冻结微调 (仿教程): Acc={r1['accuracy']:.2%}, F1={r1['f1']:.2%}")
    logger.info(f"全参数微调 (对比):  Acc={r2['accuracy']:.2%}, F1={r2['f1']:.2%}")
