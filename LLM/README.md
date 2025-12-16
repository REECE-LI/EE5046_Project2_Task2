# ECG 数据预处理工具


## 📋 项目说明

`Pro2_task_2_1.py` 是一个用于将 ECG 心电信号数据转换为 LLM（大语言模型）可用格式的预处理工具。支持多种编码器选项，用于房颤（Atrial Fibrillation）检测任务。

## ✨ 主要功能

### 1. 数据格式转换
- 将原始 ECG `.mat` 文件转换为 JSONL 格式
- 自动进行数据标准化和裁剪/填充（统一为 3000 个采样点）
- 过滤噪声样本（标签为 `~`）
- 生成适合 LLM 微调的指令-答案对

### 2. 多种编码器支持
提供三种数据处理模式：

#### 🔹 无编码器模式
- 直接使用原始 ECG 特征向量（3000 维）
- 适用于：Transformer-based LLM 直接处理序列数据
- 输出目录：`llm_data_no_encoder/`

#### 🔹 CNN 编码器模式
- 使用 CNN 提取局部特征（需自行实现模型）
- 适用于：捕捉 ECG 信号的局部模式和形态特征
- 输出目录：`llm_data_with_cnn/`

#### 🔹 Mamba 编码器模式
- 使用 Mamba 状态空间模型（需自行实现模型）
- 适用于：高效捕捉长时序依赖关系，处理心律不齐等周期性模式
- 输出目录：`llm_data_with_mamba/`

### 3. 交叉验证数据集划分
- **训练集**：CV 折 0, 1, 2 合并
- **测试集**：CV 折 3, 4 合并
- 自动生成独立的训练和测试 JSONL 文件

### 4. GUI 界面
- 简洁的图形界面，无需命令行操作
- 文件夹选择器，支持可视化选择数据路径
- 实时状态显示，处理进度提示
- 跨平台兼容（支持 macOS/Windows/Linux）

## 🚀 运行方法

### 环境要求
```bash
python >= 3.8
numpy
pandas
scipy
torch
tkinter (通常随 Python 自带)
```

### 安装依赖
```bash
pip install numpy pandas scipy torch
```

### 启动程序
```bash
cd /path/to/homework2upload
python Pro2_task_2_1.py
```

### 使用步骤
1. **启动程序**：运行后会弹出 GUI 窗口
2. **选择数据文件夹**：点击"选择文件夹"按钮，选择包含以下结构的目录：
   ```
   ecg/
   ├── cv/
   │   ├── cv0.csv
   │   ├── cv1.csv
   │   ├── cv2.csv
   │   ├── cv3.csv
   │   └── cv4.csv
   └── training2017/
       ├── A00001.mat
       ├── A00001.hea
       ├── A00002.mat
       └── ...
   ```
3. **选择处理模式**：
   - 点击"生成数据（无编码器）" - 生成原始特征数据
   - 点击"生成数据（使用 CNN）" - 使用 CNN 编码器
   - 点击"生成数据（使用 Mamba）" - 使用 Mamba 编码器
4. **等待处理完成**：程序会自动处理所有数据并生成 JSONL 文件

## 📂 Output Format

### File Structure
```
homework2upload/
├── llm_data_no_encoder/
│   ├── llm_no_encoder_train_cv012.jsonl
│   └── llm_no_encoder_test_cv34.jsonl
├── llm_data_with_cnn/
│   ├── llm_cnn_train_cv012.jsonl
│   └── llm_cnn_test_cv34.jsonl
└── llm_data_with_mamba/
    ├── llm_mamba_train_cv012.jsonl
    └── llm_mamba_test_cv34.jsonl
```

### JSONL Sample Format
```json
{
  "ecg_features": [0.1234, -0.5678, ...],  // 3000 floats
  "instruction": "请判断这个 ECG 信号是否有房颤？\nECG 信号特征向量（仅展示前 200 个点）：...",
  "answer": "有房颤。"  // or "无房颤。"
}
```

## 🔧 Encoder Implementation

Current encoder implementation is a **placeholder**, actual model needs to be added in `feature_extractor()` function:

### CNN Encoder Implementation Example
```python
if model_type == "cnn":
    with torch.no_grad():
        ecg_tensor = torch.FloatTensor(ecg_data).unsqueeze(0).unsqueeze(0)  # (1, 1, 3000)
        features = model(ecg_tensor)
        return features.squeeze().cpu().numpy()
```

### Mamba Encoder Implementation Example
```python
elif model_type == "mamba":
    with torch.no_grad():
        ecg_tensor = torch.FloatTensor(ecg_data).unsqueeze(0)  # (1, 3000)
        features = model(ecg_tensor)
        return features.squeeze().cpu().numpy()
```

### Load Model
Add in `ECGDataProcessorGUI.__init__()`:
```python
# Load CNN model
self.cnn_model = torch.load("path/to/cnn_model.pth")
self.cnn_model.eval()

# Or load Mamba model
self.mamba_model = torch.load("path/to/mamba_model.pth")
self.mamba_model.eval()
```

## 📊 Data Statistics

- **Total Samples**: ~8000+ records
- **Class Distribution**:
  - N (Normal): ~5000+
  - O (Other): ~2000+
  - A (Atrial Fibrillation): ~700+
  - ~ (Noise): Filtered out
- **Train/Test Ratio**: ~6:4 (based on CV folds)

## ⚙️ Core Parameters

- **Sampling Points**: 3000 (unified length after preprocessing)
- **Instruction Display Length**: 200 points (to avoid overly long prompts)
- **Encoding Scheme**:
  - N → [1,0,0,0] → No AF
  - O → [0,1,0,0] → No AF
  - A → [0,0,1,0] → AF
  - ~ → [0,0,0,1] → Skip

## 📝 注意事项

1. **数据路径**：确保选择的文件夹包含 `cv/` 和 `training2017/` 子目录
2. **内存占用**：处理全部数据需要约 2-4GB 内存
3. **处理时间**：
   - 无编码器：约 2-5 分钟
   - 使用编码器：取决于模型复杂度，可能需要 10-30 分钟
4. **重复运行**：每次运行会覆盖之前的输出文件

## 🐛 故障排除

### 问题：找不到 cv 文件夹
**解决**：确保选择的是包含 `cv/` 和 `training2017/` 的父目录

### 问题：处理过程中内存溢出
**解决**：关闭其他程序释放内存，或分批处理数据

### 问题：GUI 窗口无法显示
**解决**：
- macOS: `brew install python-tk`
- Ubuntu: `sudo apt-get install python3-tk`
- Windows: 重新安装 Python 并勾选 tcl/tk 选项

## 🤖 LLM 微调训练（Pro2_task_2_2-3.py）

`Pro2_task_2_2-3.py` 使用 LoRA 对大模型做微调。脚本现在只在开头解析一次命令行参数。

### 输入数据要求
- 将训练/测试 JSONL 放在同一目录，并命名为 `train_data.jsonl`、`test_data.jsonl`。
- 若使用 `Pro2_task_2_1.py` 生成的数据，可拷贝或软链：
  ```bash
  cp llm_data_no_encoder/llm_no_encoder_train_cv012.jsonl train_data.jsonl
  cp llm_data_no_encoder/llm_no_encoder_test_cv34.jsonl  test_data.jsonl
  ```

### 支持模型
- 1: Qwen/Qwen2.5-7B
- 2: Qwen/Qwen2.5-Math-7B
- 3: Qwen/Qwen3-Embedding-8B

### 关键命令行参数
- `--datapath`: 数据目录，需包含 `train_data.jsonl`、`test_data.jsonl`。省略则进入交互式输入。
- `--model_id`: 选择模型 ID（1/2/3）。省略则进入交互式选择。

### 单卡示例
```bash
python Pro2_task_2_2-3.py \
  --datapath /path/to/data \
  --model_id 1
```

### 多卡示例（torchrun 推荐）
```bash 
cd xxxx
CUDA_VISIBLE_DEVICES=1,2,3 \
torchrun --nproc_per_node=3 Pro2_task_2_2-3.py \
  --datapath llm_data_no_encoder \
  --model_id 1
```
> 将 `CUDA_VISIBLE_DEVICES` 和 `--nproc_per_node` 改成你要用的卡数；例如 4 卡用 `0,1,2,3` 且 `--nproc_per_node=4`。

### 加速器可选方案（accelerate）
```bash
pip install accelerate
CUDA_VISIBLE_DEVICES=1,2,3 accelerate launch --num_processes=3 Pro2_task_2_2-3.py \
  --datapath /path/to/data \
  --model_id 1
```

### 训练配置（脚本内置）
- 4-bit NF4 量化 (BitsAndBytes)
- LoRA: r=8, alpha=16, dropout=0.05
- Batch: 16 / GPU；Epochs: 20；LR: 5e-5
- 输出：检查点 `./ecg_llm_7b_lora_ckpt/`，最终模型 `./ecg_llm_7b_lora_finetuned/`

## 📚 后续步骤

生成的 JSONL 文件可用于：
1. **LLM 微调**：使用 LoRA、QLoRA 等方法微调大语言模型
2. **Few-shot Learning**：作为上下文示例输入 LLM
3. **评估测试**：使用测试集评估模型性能

## 📧 联系方式

如有问题或建议，请联系项目负责人。

---

**最后更新**：2025年12月10日
