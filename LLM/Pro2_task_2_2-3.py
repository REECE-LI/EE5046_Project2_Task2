import json
import os  # 新增
import argparse
from pathlib import Path

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# ========= 0. 打印 GPU 信息 =========
# 注意：在 DDP 环境下，每个进程只能看到自己的视角，或者所有设备
print(f"[Info] Process Rank {os.environ.get('LOCAL_RANK', '0')} | torch.cuda.device_count: {torch.cuda.device_count()}")

NUM_WORKERS = 8

# ========= 1. 统一参数入口 =========
# 可选模型列表
AVAILABLE_MODELS = {
    "1": "Qwen/Qwen2.5-7B",
    "2": "Qwen/Qwen2.5-Math-7B",
    "3": "Qwen/Qwen3-Embedding-8B",
}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath", type=str, help="Path to data folder")
    parser.add_argument("--model_id", type=str, choices=list(AVAILABLE_MODELS.keys()), help="Model selection ID (1/2/3)")
    parser.add_argument("--train_bs", type=int, default=16, help="per_device_train_batch_size")
    parser.add_argument("--eval_bs", type=int, default=16, help="per_device_eval_batch_size")
    parser.add_argument("--grad_accum", type=int, default=1, help="gradient_accumulation_steps")
    parser.add_argument("--max_length", type=int, default=384, help="max sequence length for tokenization")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="离线模式：只从本地缓存加载模型/分词器（不会联网校验/下载）",
    )
    parser.add_argument(
        "--online",
        action="store_true",
        help="在线模式：允许联网校验/下载（会覆盖默认离线设置）",
    )
    # parse_known 避免与分布式启动器的其他参数冲突
    args, _ = parser.parse_known_args()
    return args

def get_data_folder(args):
    if args.datapath:
        if os.path.isdir(args.datapath):
            return args.datapath
        else:
            print(f"[Error] Provided --datapath does not exist: {args.datapath}")
            exit(1)

    print("\n" + "="*50)
    print("请输入训练数据文件夹路径（包含训练和测试JSONL文件）")
    print("例如: /path/to/llm_data_no_encoder")
    print("="*50)
    
    while True:
        folder_path = input("请输入路径: ").strip()
        if folder_path and os.path.isdir(folder_path):
            return folder_path
        else:
            print(f"[Error] 路径不存在或无效: {folder_path}")
            print("请重新输入有效的文件夹路径")

def select_model(args):
    if args.model_id:
        selected = AVAILABLE_MODELS[args.model_id]
        print(f"[Info] Command line selected model: {selected}")
        return selected

    print("\n" + "="*50)
    print("请选择要使用的模型进行训练：")
    print("="*50)
    for key, model in AVAILABLE_MODELS.items():
        print(f"  [{key}] {model}")
    print("="*50)
    
    while True:
        choice = input("请输入选项 (1/2/3): ").strip()
        if choice in AVAILABLE_MODELS:
            selected = AVAILABLE_MODELS[choice]
            print(f"[Info] 已选择模型: {selected}")
            return selected
        else:
            print("[Error] 无效选项，请重新输入")

# 先解析所有参数，再使用
args = parse_args()

offline_mode = (not args.online) or args.offline
if offline_mode:
    # 彻底离线：禁止 huggingface hub/transformers 发起网络请求
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

print("[Info] Checking data folder...")
data_folder = get_data_folder(args)

if not data_folder:
    print("[Error] No folder selected. Exiting...")
    exit(1)

print(f"[Info] Selected data folder: {data_folder}")

# ========= 2. 读 JSONL =========
train_path = Path(data_folder) / "train_data.jsonl"
test_path  = Path(data_folder) / "test_data.jsonl"

# 检查文件是否存在
if not train_path.exists():
    print(f"[Error] Training file not found: {train_path}")
    exit(1)
if not test_path.exists():
    print(f"[Error] Test file not found: {test_path}")
    exit(1)

def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            data.append(obj)
    return data

train_data = load_jsonl(train_path)
test_data  = load_jsonl(test_path)

train_ds = Dataset.from_list(train_data)
test_ds  = Dataset.from_list(test_data)

print(f"[Info] Loaded {len(train_data)} training samples and {len(test_data)} test samples")

model_name = select_model(args)

# 用模型名生成安全的目录名（避免斜杠）
safe_model_name = model_name.replace("/", "__")

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    use_fast=False,
    trust_remote_code=True,
    local_files_only=offline_mode,
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ========= 关键修复：获取 Local Rank（单卡/多卡兼容） =========
local_rank = int(os.environ.get("LOCAL_RANK") or 0)
if torch.cuda.is_available():
    torch.cuda.set_device(local_rank)
else:
    print("[Warning] CUDA not available; training will fall back to CPU (not recommended for this script)")

# 训练/计算精度：支持 bf16 则用 bf16，否则用 fp16
compute_dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16

# ---- 8bit 量化配置 ----
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0,
)

# ---- 加载模型（8bit 量化）----
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map={"": local_rank},
    quantization_config=bnb_config,
    dtype=compute_dtype,
    trust_remote_code=True,
    local_files_only=offline_mode,
)

# 预处理模型 (LayerNorm 转 fp32 等)
base_model = prepare_model_for_kbit_training(base_model)
base_model.config.use_cache = False  # 与 gradient checkpointing 兼容
try:
    base_model.gradient_checkpointing_enable(use_reentrant=False)
except TypeError:
    base_model.gradient_checkpointing_enable()


# ========= 2.2 配置 LoRA =========
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
)

model = get_peft_model(base_model, lora_config)
# 只在主进程打印参数，避免刷屏
if local_rank == 0:
    model.print_trainable_parameters()

# ========= 3. 文本拼接 =========
def build_example(ex):
    instr = ex["instruction"]
    ans = ex["answer"]
    prompt = f"指令：{instr}\n回答："
    full_text = prompt + ans
    enc = tokenizer(
        full_text,
        truncation=True,
        max_length=args.max_length,
        padding="max_length",
    )
    input_ids = enc["input_ids"]
    labels = input_ids.copy()
    enc["labels"] = labels
    return enc

# DDP 下数据处理最好加上 load_from_cache_file=False 或者处理好缓存路径，
# 但简单的 map 也是可以的
train_tokenized = train_ds.map(build_example, batched=False, num_proc=NUM_WORKERS)
test_tokenized  = test_ds.map(build_example, batched=False, num_proc=NUM_WORKERS)

# ========= 4. 训练参数 =========
training_args = TrainingArguments(
    output_dir=str(Path(data_folder) / f"ecg_llm_lora_ckpt_{safe_model_name}"),

    per_device_train_batch_size=args.train_bs,
    per_device_eval_batch_size=args.eval_bs,
    gradient_accumulation_steps=args.grad_accum,

    learning_rate=5e-5,
    num_train_epochs=80,

    logging_steps=10,
    logging_first_step=True,
    eval_strategy="epoch",
    save_strategy="epoch",

    bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
    fp16=torch.cuda.is_available() and (not torch.cuda.is_bf16_supported()),
    gradient_checkpointing=True,

    save_total_limit=2,
    report_to="none",

    # DDP 关键参数
    ddp_find_unused_parameters=False,
    dataloader_num_workers=8,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized,
    eval_dataset=test_tokenized,
)

if __name__ == "__main__":
    # 注意：在 DDP 中，trainer.train() 会自动处理多卡同步
    trainer.train()

    # 保存模型最好只由主进程执行
    if local_rank == 0:
        save_dir = str(Path(data_folder) / f"ecg_llm_lora_finetuned_{safe_model_name}")
        trainer.save_model(save_dir)
        tokenizer.save_pretrained(save_dir)
        print(f"[Info] LoRA model saved to: {save_dir}")