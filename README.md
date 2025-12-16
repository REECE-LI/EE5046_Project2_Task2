# Task 2：ECG 房颤检测与 LLM 微调说明

- `ecg/`：PhysioNet 2017 单导 ECG 原始数据（`training2017/`）及五折划分 (`cv/cv0-4.csv`)，`train_model.ipynb` 是分类器探索笔记。
- `LLM/`：全部可运行脚本与产物：
  - `Pro2_task_1_2.py`：多尺度 CNN（MS-CNN）房颤检测基线，5 折交叉验证。
  - `Pro2_task_2_1.py`：ECG→指令/答案 JSONL 生成器 + GUI，可选 CNN/Mamba 编码。
  - `Pro2_task_2_2-3.py`：Qwen 系列模型的 LoRA 微调（QLoRA 8bit 风格）。
  - `eval_lora_llm.py`：LoRA 评估脚本，输出 acc/precision/recall/F1/AUC，可写 CSV/JSONL。
  - 生成的数据集与微调模型：`llm_data_no_encoder/`、`llm_data_with_cnn/`、`llm_data_with_mamba/`（内含 `train_data.jsonl`、`test_data.jsonl`、metrics CSV、`ecg_llm_lora_finetuned_Qwen__Qwen2.5-7B/`）。
  - `mscnn_fold4.pth`：CNN 基线权重示例。
- `project2.pdf`：题目说明。

## 已完成内容

1) **ECG 二分类器（Pro2_task_1_2.py）**  
   - 预处理：60 Hz 低通 FIR，300→150→120 Hz 下采样，z-score 归一化，pad/截断到 600。  
   - 模型：单流 CNN 基线 vs 双流 MS-CNN（3×3 与 7/3 混合卷积），共享全连接头。  
   - 数据：`cv0-4.csv` 5 折，训练 4 折、测试 1 折。  
   - 训练：Adam 1e-4，BCE，早停（耐心 20），最多 100 轮；保存每折最佳 ckpt、loss 曲线到 `results/`，并汇总 `all_data.npz`。

2) **ECG→LLM 数据集构建 + GUI（Pro2_task_2_1.py）**  
   - 读取 `cv/` 与 `training2017/*.mat`，归一化，下采样到 100 Hz，裁剪/填充至 3000 点。  
   - 标签映射：A=有房颤，N/O=无房颤，`~` 噪声丢弃；指令预览截断特征避免超长 prompt。  
   - 三种模式：原始特征 / CNN 编码 / 精简 Mamba 编码；GUI 按钮生成对应模式的 `train_data.jsonl`、`test_data.jsonl`（存于 `llm_data_no_encoder/`、`llm_data_with_cnn/`、`llm_data_with_mamba/`）。  
   - 可作为模块导入调用 `process_data(...)` 实现无界面批处理。

3) **LLM LoRA 微调（Pro2_task_2_2-3.py）**  
   - 支持 Qwen/Qwen2.5-7B、Qwen/Qwen2.5-Math-7B、Qwen3-Embedding-8B（`--model_id` 选择）。  
   - 8bit 加载（BitsAndBytes），LoRA r=8/alpha=16/dropout=0.05，目标投影层；开启梯度检查点。  
   - Token 组装 `指令：...\n回答：...`，max length 384，labels=inputs。  
   - 训练参数：批量与累积可配，80 轮，每轮评估+保存；输出 `ecg_llm_lora_ckpt_*`，最终适配器+分词器存 `ecg_llm_lora_finetuned_*`。  
   - 默认离线（`HF_HUB_OFFLINE=1`），需要联网加 `--online`。

4) **LoRA 评估（eval_lora_llm.py）**  
   - 加载 base+adapter，确定性生成，解析中英房颤关键词，统计 acc/precision/recall/F1/AUC 与混淆矩阵。  
   - 支持批量生成、定期进度打印，可写 JSONL/CVS 记录。

## 快速开始

- **环境**：`python>=3.8`，`numpy`，`pandas`，`scipy`，`matplotlib`，`scikit-learn`，`torch`，`transformers`，`datasets`，`peft`，`bitsandbytes`；GUI 需 `tkinter`（通常自带）；多卡可装 `accelerate`。

- **训练 CNN 基线**  
  - 确保 `cv0-4.csv` 在工作目录（可从 `ecg/cv/` 拷贝），脚本中 `ecg_dir` 指向 `training2017/`（如 `../ecg/training2017`）。  
  - 运行：`python LLM/Pro2_task_1_2.py`（结果保存在 `LLM/results/`）。

- **生成 LLM 用 JSONL**  
  - GUI：`python LLM/Pro2_task_2_1.py`，选择含 `cv/` 与 `training2017/` 的目录，点击模式按钮即可。  
  - CLI 示例：
    ```python
    from LLM.Pro2_task_2_1 import process_data
    process_data("/path/to/ecg", use_encoder=False)  # 原始特征模式
    ```

- **LoRA 微调 Qwen**  
  - 单卡示例（原始特征）：  
    ```bash
    cd LLM
    python Pro2_task_2_2-3.py --datapath llm_data_no_encoder --model_id 1 --train_bs 8 --eval_bs 8 --grad_accum 2
    ```
  - 多卡（推荐 torchrun，示例 4 卡，显存不足可调小 batch/accum）：  
    ```bash
    cd LLM
    CUDA_VISIBLE_DEVICES=0,1,2,3 \
    torchrun --nproc_per_node=4 Pro2_task_2_2-3.py \
      --datapath llm_data_no_encoder \
      --model_id 1 \
      --train_bs 8 \
      --eval_bs 8 \
      --grad_accum 2
    ```
  - 多卡（accelerate）：
    ```bash
    cd LLM
    CUDA_VISIBLE_DEVICES=0,1,2,3 \
    accelerate launch --num_processes=4 Pro2_task_2_2-3.py \
      --datapath llm_data_no_encoder \
      --model_id 1 \
      --train_bs 8 \
      --eval_bs 8 \
      --grad_accum 2
    ```

- **评估 LoRA**  
  ```bash
  cd LLM
  python eval_lora_llm.py \
    --data_folder llm_data_no_encoder \
    --adapter_dir llm_data_no_encoder/ecg_llm_lora_finetuned_Qwen__Qwen2.5-7B \
    --base_model Qwen/Qwen2.5-7B
  ```
  - 指标打印到控制台，默认追加到 `metrics_log.csv`。

## 现有产物

- 预生成 JSONL 与微调适配器位于：
  - `LLM/llm_data_no_encoder/`
  - `LLM/llm_data_with_cnn/`
  - `LLM/llm_data_with_mamba/`
- CNN 权重示例：`LLM/mscnn_fold4.pth`。




### 与传统CNN分类器（任务1）相比，这种基于LLM的方法（任务2）的主要优势和劣势是什么？
传统 CNN (任务1) 的优势：
- 计算效率与部署： CNN 结构紧凑，参数量远小于 LLM（例如 ResNet vs. Qwen-7B）。对于嵌入式工程师而言，CNN 更容易部署在边缘设备（如 FPGA 或 MCU）上进行实时心电监测，延迟极低。
- 归纳偏置 (Inductive Bias)： CNN 的卷积操作具有平移不变性，非常适合处理 ECG 这种具有局部特征（如 QRS 波群）的时间序列信号。
- 黑盒可控性： 虽然也是黑盒，但 CNN 的输出直接是概率分布，更容易通过阈值调整来平衡灵敏度和特异性。
  
基于 LLM 方法 (任务2) 的优势：
- 多模态融合潜力： LLM 的核心优势在于它可以处理多模态信息。在临床场景中，诊断不仅仅依赖 ECG 波形，还依赖患者的主诉（文本）、年龄、性别等。LLM 架构允许将这些异构数据统一 embedding 后输入。
- 泛化与零样本能力： 经过微调的 LLM 可能具备更强的语义理解能力。如果任务变为“描述这段 ECG 的异常特征”，CNN 需要重新设计输出层和训练，而 LLM 可能只需改变指令（Prompt）。
- 交互性： LLM 可以通过自然语言解释判断依据（尽管需要专门的对齐训练），提供比单纯 label 更丰富的信息。
LLM 方法的劣势：
- 资源过剩 (Overkill)： 对于单纯的二分类任务，使用 7B 参数的模型是巨大的算力浪费。
- 幻觉风险 (Hallucination)： 生成式模型可能会输出非预期的文本（除非做严格的 constraint decoding），导致后处理解析困难。
- 鬼知道LLM会输出什么？我单独写了一个检查LLM会说出什么花来的程序，每次都是不一样的，并且有时候会答非所问。

### 在这个任务中，CNN编码器的作用是什么？是否可以尝试不同的编码器？
CNN 编码器的作用：
- 特征提取与降维： 原始 ECG 信号（时间序列）维度过高，且包含大量冗余信息。LLM 的 Context Window 是有限的，且 Attention 机制的计算复杂度是 $O(N^2)$。CNN 将长序列信号压缩为包含高层语义的紧凑特征向量（Embedding），使其能够作为“视觉/信号 Token”输入到 LLM 中。
- 模态对齐： 它充当了从“信号空间”到“语言语义空间”的桥梁（通常配合一个 Projection Layer/MLP）。
可尝试的替代编码器：
- Transformer (如 1D-ViT 或 AST): 利用 Self-attention 捕捉 ECG 信号中的长距离依赖关系（例如心律不齐的周期性模式），这方面通常优于 CNN 的局部感受野。
- Mamba / State Space Models (SSM): 结合了 RNN 的推理效率和 Transformer 的训练并行性，非常适合处理长序列生理信号。

实际感受：感觉差不多。。。

### 指令的措辞（例如, "是否有房颤？" vs. "此心电图是否显示心房颤动？"）是否会影响模型性能？为什么？
几乎不会影响模型性能，原因如下：
- 微调适应性： LLM 经过微调后，模型会学习到不同措辞下的语义等价关系。只要训练数据中涵盖了多样化的指令表达，模型就能泛化到类似的变体。
- 语义理解： LLM 的强大之处在于其对自然语言的深层理解能力。不同措辞本质上表达了相同的查询意图，模型能够捕捉到这种语义一致性。
- 数据量与多样性： 如果训练数据量足够大且多样化，模型会更关注问题的核心内容（是否有房颤）而非具体措辞。
不过，如果指令过于复杂或含糊，可能会引入理解歧义，影响性能。因此，保持指令简洁明了是一个好的实践。