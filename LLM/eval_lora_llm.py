import argparse
import csv
import json
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def parse_args():
    parser = argparse.ArgumentParser(description="评估 LoRA 微调后的 ECG LLM（二分类指令回答）")
    parser.add_argument(
        "--data_folder",
        required=True,
        help="包含 test_data.jsonl 的目录，比如 llm_data_no_encoder / llm_data_with_cnn 等",
    )
    parser.add_argument(
        "--adapter_dir",
        required=True,
        help="LoRA 微调结果目录，例如 ecg_llm_lora_finetuned_Qwen__Qwen2.5-7B",
    )
    parser.add_argument(
        "--base_model",
        default="Qwen/Qwen2.5-7B",
        help="基础模型名称或本地路径（需与微调时一致）",
    )
    parser.add_argument(
        "--file",
        default="test_data.jsonl",
        help="要评估的 JSONL 文件名（默认使用 data_folder/test_data.jsonl）",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=32,
        help="生成的最大 token 数",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="推理时一次生成的样本数（显存足够可适当调大）",
    )
    parser.add_argument(
        "--progress_every",
        type=int,
        default=10,
        help="每多少条样本打印一次进度",
    )
    parser.add_argument(
        "--metrics_path",
        default="",
        help="(可选) 将进度和最终指标追加写入的 JSONL 文件路径，默认空表示不写 JSONL",
    )
    parser.add_argument(
        "--metrics_csv_path",
        default=None,
        help="将最终汇总指标追加写入的 CSV 文件路径；默认写到 data_folder/metrics_log.csv；传空字符串则不落盘",
    )
    return parser.parse_args()


def load_jsonl(path: Path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def build_prompt(instruction: str):
    # 保持与训练时一致：指令：xxx\n回答：
    return f"指令：{instruction}\n回答："


def parse_pred(text: str):
    """
    粗略解析模型输出，判断有/无房颤。
    返回：1=有房颤；0=无房颤；None=无法解析
    """
    # 两份文本：
    # - t_cn: 去掉空白/标点后用于中文关键词匹配
    # - raw_l: 保留分隔符(只做小写/空白归一)用于英文 AF 的正则匹配，避免把 “不是AF” 误判为正类
    raw_l = text.lower()
    raw_l = re.sub(r"[\t\r\n]+", " ", raw_l)
    raw_l = re.sub(r"[，。,.;:!?（）()\[\]{}<>/\\\-_=+\"'`~]", " ", raw_l)
    raw_l = re.sub(r"\s+", " ", raw_l).strip()

    t_cn = (
        text.replace(" ", "")
        .replace("\t", "")
        .replace("\n", "")
        .replace("\r", "")
        .replace("。", "")
        .replace("，", "")
        .replace(",", "")
        .replace(".", "")
        .strip()
    )

    # 重要：先匹配否定
    # 注意“没有房颤”包含子串“有房颤”，若先匹配肯定会误判
    negative_patterns = [
        "无房颤",
        "无心房颤动",
        "没有房颤",
        "没有心房颤动",
        "未见房颤",
        "未见心房颤动",
        "未发现房颤",
        "未发现心房颤动",
        "未检测到房颤",
        "未检测到心房颤动",
        "房颤阴性",
        "非房颤",
        "排除房颤",
        "排除心房颤动",
        "notaf",
        "noaf",
        "noatrialfibrillation",
    ]
    if any(p in t_cn for p in negative_patterns):
        return 0

    # 英文/缩写否定：比如 “不是AF/非AF/无AF/not af/no af”
    af_neg_re = re.compile(r"\b(?:not|no|without)\s*af\b")
    if af_neg_re.search(raw_l):
        return 0
    # 处理中文语境里的 AF 否定（原始输出里可能没有空格）
    if any(p in t_cn.lower() for p in ["不是af", "非af", "无af", "没有af", "排除af"]):
        return 0

    positive_patterns = [
        "有房颤",
        "存在房颤",
        "检测到房颤",
        "检测到心房颤动",
        "提示房颤",
        "提示心房颤动",
        "考虑房颤",
        "房颤阳性",
        "心房颤动",
        "atrialfibrillation",
    ]
    if any(p in t_cn for p in positive_patterns):
        return 1

    # 英文/缩写肯定：匹配独立单词 af，避免误伤 after/safe 等
    af_pos_re = re.compile(r"\baf\b")
    if af_pos_re.search(raw_l):
        return 1
    return None


def compute_binary_metrics(preds, labels):
    n = len(preds)
    if n == 0:
        return 0.0, 0.0, 0.0, 0.0
    tp = sum(1 for p, y in zip(preds, labels) if p == 1 and y == 1)
    tn = sum(1 for p, y in zip(preds, labels) if p == 0 and y == 0)
    fp = sum(1 for p, y in zip(preds, labels) if p == 1 and y == 0)
    fn = sum(1 for p, y in zip(preds, labels) if p == 0 and y == 1)

    acc = (tp + tn) / n
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return acc, precision, recall, f1


def compute_confusion_counts(preds, labels):
    tp = sum(1 for p, y in zip(preds, labels) if p == 1 and y == 1)
    tn = sum(1 for p, y in zip(preds, labels) if p == 0 and y == 0)
    fp = sum(1 for p, y in zip(preds, labels) if p == 1 and y == 0)
    fn = sum(1 for p, y in zip(preds, labels) if p == 0 and y == 1)
    return tp, fp, tn, fn


def compute_auc(scores, labels):
    """计算二分类 AUC（采用简单的成对比较，支持离散分数）。"""
    if not scores or not labels:
        return 0.0

    pos_scores = [s for s, y in zip(scores, labels) if y == 1]
    neg_scores = [s for s, y in zip(scores, labels) if y == 0]

    n_pos, n_neg = len(pos_scores), len(neg_scores)
    if n_pos == 0 or n_neg == 0:
        return 0.0

    better, tie = 0.0, 0.0
    for ps in pos_scores:
        for ns in neg_scores:
            if ps > ns:
                better += 1
            elif ps == ns:
                tie += 1
    return (better + 0.5 * tie) / (n_pos * n_neg)


def main():
    args = parse_args()

    # 默认把 CSV 写到数据文件夹下；若显式传空字符串则禁用
    if args.metrics_csv_path is None:
        args.metrics_csv_path = str(Path(args.data_folder) / "metrics_log.csv")

    data_path = Path(args.data_folder) / args.file
    if not data_path.exists():
        raise FileNotFoundError(f"未找到数据文件: {data_path}")

    print(f"[Info] 加载数据: {data_path}")
    samples = load_jsonl(data_path)

    print(f"[Info] 加载基础模型: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=False, trust_remote_code=True)
    # decoder-only 模型推理需左侧 padding
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        device_map="auto",
        trust_remote_code=True,
    )

    print(f"[Info] 加载 LoRA 适配器: {args.adapter_dir}")
    model = PeftModel.from_pretrained(base_model, args.adapter_dir)
    model.eval()

    device = next(model.parameters()).device

    # 小批量生成并统计
    total, parsed, correct = 0, 0, 0
    pred_buffer, label_buffer, score_buffer = [], [], []
    progress_records = []
    batch_size = args.batch_size

    for start in range(0, len(samples), batch_size):
        batch = samples[start : start + batch_size]

        prompts = [build_prompt(ex["instruction"]) for ex in batch]
        gt_labels = [1 if "有房颤" in ex["answer"] else 0 for ex in batch]

        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
        ).to(device)

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )

        # 重要：batch 内做了 padding（且是 left padding），generate 输出的前缀长度是“padding 后的统一长度”
        # 不能用 attention_mask.sum()（那是非 pad token 数），否则会把 prompt 的一部分误当作生成文本。
        input_len = int(inputs["input_ids"].shape[1])
        for i in range(len(batch)):
            gen_text = tokenizer.decode(out[i][input_len:], skip_special_tokens=True)

            pred_label = parse_pred(gen_text)

            total += 1
            if pred_label is not None:
                parsed += 1
                if pred_label == gt_labels[i]:
                    correct += 1
                pred_buffer.append(pred_label)
                label_buffer.append(gt_labels[i])
                score_buffer.append(float(pred_label))

            if total % args.progress_every == 0:
                acc, prec, rec, f1 = compute_binary_metrics(pred_buffer, label_buffer)
                auc = compute_auc(score_buffer, label_buffer)
                parse_ratio = parsed / total if total else 0.0
                tp, fp, tn, fn = compute_confusion_counts(pred_buffer, label_buffer)
                label_pos_rate = (sum(label_buffer) / len(label_buffer)) if label_buffer else 0.0
                pred_pos_rate = (sum(pred_buffer) / len(pred_buffer)) if pred_buffer else 0.0
                progress_records.append(
                    {
                        "seen": total,
                        "parsed": parsed,
                        "parse_ratio": parse_ratio,
                        "acc": acc,
                        "precision": prec,
                        "recall": rec,
                        "f1": f1,
                        "auc": auc,
                        "tp": tp,
                        "fp": fp,
                        "tn": tn,
                        "fn": fn,
                        "label_pos_rate": label_pos_rate,
                        "pred_pos_rate": pred_pos_rate,
                    }
                )
                print(
                    f"[Progress] {total}/{len(samples)} | 可解析 {parsed} ({parse_ratio:.2%}) | "
                    f"acc {acc:.4f} | precision {prec:.4f} | recall {rec:.4f} | f1 {f1:.4f} | auc {auc:.4f} | "
                    f"pos(y/p) {label_pos_rate:.3f}/{pred_pos_rate:.3f} | TP/FP/TN/FN {tp}/{fp}/{tn}/{fn}"
                )

    acc, prec, rec, f1 = compute_binary_metrics(pred_buffer, label_buffer)
    auc = compute_auc(score_buffer, label_buffer)
    parse_ratio = parsed / total if total else 0.0
    tp, fp, tn, fn = compute_confusion_counts(pred_buffer, label_buffer)
    label_pos_rate = (sum(label_buffer) / len(label_buffer)) if label_buffer else 0.0
    pred_pos_rate = (sum(pred_buffer) / len(pred_buffer)) if pred_buffer else 0.0
    print("=" * 50)
    print(f"[Result] 总样本: {total}")
    print(f"[Result] 可解析: {parsed} (占 {parse_ratio:.2%})")
    print(f"[Result] 准确率: {acc:.4f}")
    print(f"[Result] 精确率: {prec:.4f}")
    print(f"[Result] 召回率: {rec:.4f}")
    print(f"[Result] F1: {f1:.4f}")
    print(f"[Result] AUC: {auc:.4f}")
    print(f"[Result] 正类比例(y): {label_pos_rate:.4f} | 预测正类比例(p): {pred_pos_rate:.4f}")
    print(f"[Result] TP/FP/TN/FN: {tp}/{fp}/{tn}/{fn}")
    print("=" * 50)

    if args.metrics_path:
        record = {
            "data_file": str(data_path),
            "adapter_dir": str(args.adapter_dir),
            "base_model": args.base_model,
            "total": total,
            "parsed": parsed,
            "parse_ratio": parse_ratio,
            "acc": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "auc": auc,
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "label_pos_rate": label_pos_rate,
            "pred_pos_rate": pred_pos_rate,
            "progress": progress_records,
        }
        with open(args.metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    if args.metrics_csv_path:
        csv_path = Path(args.metrics_csv_path)
        write_header = not csv_path.exists()
        fieldnames = [
            "data_file",
            "adapter_dir",
            "base_model",
            "total",
            "parsed",
            "parse_ratio",
            "acc",
            "precision",
            "recall",
            "f1",
            "auc",
            "tp",
            "fp",
            "tn",
            "fn",
            "label_pos_rate",
            "pred_pos_rate",
        ]
        with open(csv_path, "a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow(
                {
                    "data_file": str(data_path),
                    "adapter_dir": str(args.adapter_dir),
                    "base_model": args.base_model,
                    "total": total,
                    "parsed": parsed,
                    "parse_ratio": parse_ratio,
                    "acc": acc,
                    "precision": prec,
                    "recall": rec,
                    "f1": f1,
                    "auc": auc,
                    "tp": tp,
                    "fp": fp,
                    "tn": tn,
                    "fn": fn,
                    "label_pos_rate": label_pos_rate,
                    "pred_pos_rate": pred_pos_rate,
                }
            )


if __name__ == "__main__":
    main()
