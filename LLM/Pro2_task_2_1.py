import os
import json
import numpy as np
import pandas as pd
import scipy.io as io
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk

# ========= ECG 数据集定义 =========
class ECG_dataset(Dataset):
    def __init__(self, base_file=None, cv=0, is_train=True, transform=None):
        self.is_train = is_train
        self.file_list = []
        self.base_file = base_file

        for i in range(5):
            data = pd.read_csv(f"{base_file}/cv/cv{i}.csv")
            self.file_list.append(data.to_numpy())

        if is_train:
            del self.file_list[cv]
            self.file = self.file_list[0]
            for i in range(1, 4):
                self.file = np.append(self.file, self.file_list[i], axis=0)
        else:
            self.file = self.file_list[cv]

    def __len__(self):
        return self.file.shape[0]

    def load_data(self, file_name, label):
        mat_file = f"{self.base_file}/training2017/{file_name}.mat"
        data = io.loadmat(mat_file)["val"]
        if label == "N":
            one_hot = torch.tensor([1, 0, 0, 0])
        elif label == "O":
            one_hot = torch.tensor([0, 1, 0, 0])
        elif label == "A":
            one_hot = torch.tensor([0, 0, 1, 0])
        elif label == "~":
            one_hot = torch.tensor([0, 0, 0, 1])
        else:
            raise ValueError(f"Unknown label: {label}")
        return data, one_hot

    def crop_padding(self, data, time):
        if data.shape[0] <= time:
            data = np.pad(data, (time - data.shape[0]), "constant")
        elif data.shape[0] > time:
            end_index = data.shape[0] - time
            start = np.random.randint(0, end_index)
            data = data[start : start + time]
        return data

    def data_process(self, data):
        data = data[::3]
        data = data - data.mean()
        data = data / data.std()
        data = self.crop_padding(data, 3000)
        return data

    def __getitem__(self, idx):
        file_name = self.file[idx][1]
        label = self.file[idx][2]
        data, one_hot = self.load_data(file_name, label)
        data = self.data_process(data[0])
        return data, one_hot, file_name

# ========= 工具：ECG → 文本 =========
def ecg_to_str(ecg_vec, max_len_show=200):
    """
    ecg_vec: 1D numpy array，长度约 3000 左右（data_process 之后）
    max_len_show: 指令里最多保留多少个点，防止 prompt 太长
    """
    ecg_vec = ecg_vec[:max_len_show]
    return " ".join(f"{x:.4f}" for x in ecg_vec)


def build_instruction_text(ecg_vec, max_len_show=200):
    """按照指定模板构造指令文本，向模型展示截断后的信号预览。"""
    preview = ecg_to_str(ecg_vec, max_len_show=max_len_show)
    return (
        "请判断这个 ECG 信号是否有房颤？\n"
        f"ECG 信号特征向量（仅展示前 {max_len_show} 个点）：{preview}"
    )


# ========= one-hot → AF / 非AF 标签 =========
def one_hot_to_af_label(one_hot):
    """
    one_hot 编码（你原来的逻辑）：
        N -> [1,0,0,0]
        O -> [0,1,0,0]
        A -> [0,0,1,0]
        ~ -> [0,0,0,1]

    返回：
        1 = 有房颤 (A)
        0 = 无房颤 (N 或 O)
       -1 = 噪声 (~)，直接丢弃
    """
    idx = int(torch.argmax(one_hot).item())
    if idx == 2:          # A
        return 1
    elif idx in (0, 1):   # N 或 O
        return 0
    else:                 # '~'
        return -1


# ========= CNN 编码器（来自 Pro2_task_1_2）=========
class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, k):
        super().__init__()
        self.conv = nn.Conv1d(in_c, out_c, k, padding=k // 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv(x))


class Stream(nn.Module):
    def __init__(self, kernels):
        super().__init__()
        cin = [1, 64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512]
        cout = [64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512]

        self.blocks = nn.ModuleList(
            [ConvBlock(cin[i], cout[i], kernels[i]) for i in range(13)]
        )

        self.pool_pos = [2, 4, 7, 10, 13]
        self.pools = nn.ModuleList(
            [
                nn.MaxPool1d(3, stride=3),
                nn.MaxPool1d(2, stride=2),
                nn.MaxPool1d(2, stride=2),
                nn.MaxPool1d(2, stride=2),
                nn.MaxPool1d(2, stride=2),
            ]
        )

    def forward(self, x):
        pid = 0
        for i, blk in enumerate(self.blocks, 1):
            x = blk(x)
            if i in self.pool_pos:
                x = self.pools[pid](x)
                pid += 1
        return x


class MSCNN(nn.Module):
    def __init__(self, input_len=3000, use_stream2=True):
        super().__init__()
        self.use_stream2 = use_stream2
        self.stream1 = Stream([3] * 13)  # 单尺度
        if use_stream2:
            self.stream2 = Stream([7] * 4 + [3] * 9)  # 多尺度

        self.flat_dim = self._get_flat_dim(input_len)
        self.fc = nn.Sequential(
            nn.Linear(self.flat_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def _get_flat_dim(self, L):
        x = torch.randn(1, 1, L)
        y1 = self.stream1(x)
        if self.use_stream2:
            y2 = self.stream2(x)
            y = torch.cat([y1, y2], dim=1)
        else:
            y = y1
        return y.flatten(1).shape[1]

    def extract_feature(self, x):
        y1 = self.stream1(x)
        if self.use_stream2:
            y2 = self.stream2(x)
            y = torch.cat([y1, y2], dim=1)
        else:
            y = y1
        return y.flatten(1)

    def forward(self, x):
        feat = self.extract_feature(x)
        return torch.sigmoid(self.fc(feat))


def build_cnn_encoder(input_len=3000, use_stream2=True, ckpt_path=None, device=None):
    """
    初始化 CNN 编码器，可选加载 checkpoint。
    """
    model = MSCNN(input_len=input_len, use_stream2=use_stream2)
    if ckpt_path and os.path.exists(ckpt_path):
        state = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(state, strict=False)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return model


# ========= 简化版 Mamba 编码器 =========
class SimpleMambaBlock(nn.Module):
    """
    一个轻量的 Mamba 风格模块，使用深度可分离卷积与门控。
    旨在无外部依赖情况下提供长序列建模能力。
    """

    def __init__(self, dim, kernel_size=7):
        super().__init__()
        self.depthwise = nn.Conv1d(dim, dim, kernel_size, padding=kernel_size // 2, groups=dim)
        self.gate = nn.Conv1d(dim, dim, 1)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        # x: (B, dim, L)
        residual = x
        conv_out = self.depthwise(x)
        gated = conv_out * torch.sigmoid(self.gate(x))
        gated = gated.transpose(1, 2)  # (B, L, dim)
        return self.norm(gated + residual.transpose(1, 2)).transpose(1, 2)


class SimpleMambaEncoder(nn.Module):
    def __init__(self, dim=128, n_layers=4, kernel_size=7):
        super().__init__()
        self.input_proj = nn.Conv1d(1, dim, kernel_size=3, padding=1)
        self.blocks = nn.ModuleList([SimpleMambaBlock(dim, kernel_size) for _ in range(n_layers)])
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x):
        # x: (B, L)
        x = self.input_proj(x.unsqueeze(1))  # (B, dim, L)
        for blk in self.blocks:
            x = blk(x)
        x = self.pool(x).squeeze(-1)
        return self.out_proj(x)


def build_mamba_encoder(dim=128, n_layers=4, kernel_size=7, ckpt_path=None, device=None):
    """
    初始化简化版 Mamba 编码器，可选加载 checkpoint。
    """
    model = SimpleMambaEncoder(dim=dim, n_layers=n_layers, kernel_size=kernel_size)
    if ckpt_path and os.path.exists(ckpt_path):
        state = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(state, strict=False)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return model


# ========= CNN/Mamba 特征提取接口 =========
def feature_extractor(ecg_data, model=None, model_type="cnn"):
    """
    通用特征提取接口 - 支持 CNN 和 Mamba 编码器
    
    参数:
        ecg_data: numpy array, shape (3000,)
        model: 可选的编码器模型，如果为 None 则返回原始数据
        model_type: 模型类型 ["cnn", "mamba"]
    
    返回:
        features: numpy array, 处理后的特征向量
    
    使用示例:
        # 不使用编码器（直接用于 LLM）
        features = feature_extractor(ecg_data)
        
        # 使用 CNN 编码器
        cnn_model = load_cnn_model()
        features = feature_extractor(ecg_data, model=cnn_model, model_type="cnn")
        
        # 使用 Mamba 编码器
        mamba_model = load_mamba_model()
        features = feature_extractor(ecg_data, model=mamba_model, model_type="mamba")
    """
    if model is None:
        # 无编码器处理，直接返回原始数据
        return ecg_data
    
    ecg_tensor = torch.as_tensor(ecg_data, dtype=torch.float32)
    device = next(model.parameters()).device
    model.eval()

    if model_type == "cnn":
        with torch.no_grad():
            inp = ecg_tensor.unsqueeze(0).unsqueeze(0).to(device)  # (1,1,L)
            if hasattr(model, "extract_feature"):
                feat = model.extract_feature(inp)
            else:
                feat = model(inp)
            return feat.squeeze().detach().cpu().numpy()

    if model_type == "mamba":
        with torch.no_grad():
            inp = ecg_tensor.unsqueeze(0).to(device)  # (1,L)
            feat = model(inp)
            return feat.squeeze().detach().cpu().numpy()

    raise ValueError(f"不支持的模型类型: {model_type}，仅支持 'cnn' 和 'mamba'")


# ========= 针对单个 dataset 写入 JSONL（追加写入） =========
def append_dataset_to_jsonl(dataset, jsonl_path, max_len_show=200, encoder_model=None, use_encoder=False, model_type="cnn"):
    """
    参数:
        dataset: ECG_dataset 实例
        jsonl_path: 输出的 JSONL 文件路径
        max_len_show: 指令中展示的特征点数量
        encoder_model: 编码器模型实例（可选，支持 CNN/Mamba）
        use_encoder: 是否使用编码器提取特征
        model_type: 编码器类型 ["cnn", "mamba"]
    """
    total = len(dataset)
    count = 0

    # 使用 a 模式追加写入；第一次调用前外面记得先清空文件
    with open(jsonl_path, "a", encoding="utf-8") as f:
        for idx in range(total):
            data, one_hot, fname = dataset[idx]  # ECG_dataset 返回的三个

            y = one_hot_to_af_label(one_hot)
            if y == -1:
                # 噪声记录，跳过
                continue

            # 可选：使用编码器提取特征
            if use_encoder:
                processed_data = feature_extractor(data, model=encoder_model, model_type=model_type)
            else:
                processed_data = data

            # data 是 numpy 数组，先截断生成提示，再完整保存特征向量
            instruction = build_instruction_text(processed_data, max_len_show=max_len_show)
            ecg_features = processed_data.tolist()
            answer = "有房颤。" if y == 1 else "无房颤。"

            sample = {
                "ecg_features": ecg_features,  # 模型或后处理可直接取用的数值向量
                "instruction": instruction,
                "answer": answer
            }

            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
            count += 1

            if (idx + 1) % 500 == 0:
                print(f"[{os.path.basename(jsonl_path)}] processed {idx + 1}/{total}")

    print(f"[{os.path.basename(jsonl_path)}] 新增样本数 = {count}")
    return count


def select_folder():
    """弹出文件夹选择对话框，返回选中的文件夹路径"""
    root = tk.Tk()
    root.withdraw()  # 隐藏主窗口
    folder_path = filedialog.askdirectory(title="请选择 ECG 数据文件夹（包含 cv/ 和 training2017/）")
    root.destroy()
    return folder_path


# ========= 数据处理核心函数 =========
def process_data(base_dir, use_encoder=False, encoder_model=None, model_type="cnn"):
    """
    数据处理核心函数
    
    参数:
        base_dir: ECG 数据文件夹路径
        use_encoder: 是否使用编码器处理
        encoder_model: 编码器模型实例（可选，支持 CNN/Mamba）
        model_type: 编码器类型 ["cnn", "mamba"]
    """
    if not base_dir:
        print("未选择文件夹，处理取消。")
        return
    
    print(f"\n{'='*60}")
    print(f"选择的 ECG 数据文件夹: {base_dir}")
    if use_encoder:
        print(f"处理模式: 使用 {model_type.upper()} 编码器提取特征")
    else:
        print(f"处理模式: 不使用编码器（原始特征）")
    print(f"{'='*60}\n")
    
    # 输出目录默认为当前脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    if use_encoder:
        # 按编码方式命名文件夹，文件统一命名 train_data / test_data
        out_dir = os.path.join(script_dir, f"llm_data_with_{model_type}")
    else:
        out_dir = os.path.join(script_dir, "llm_data_no_encoder")

    train_jsonl = os.path.join(out_dir, "train_data.jsonl")
    test_jsonl  = os.path.join(out_dir, "test_data.jsonl")
    
    os.makedirs(out_dir, exist_ok=True)
    print(f"输出目录: {out_dir}\n")

    # 0,1,2 → 训练集
    train_folds = [0, 1, 2]
    # 3,4 → 测试集
    test_folds = [3, 4]

    # 先清空 / 新建文件
    open(train_jsonl, "w", encoding="utf-8").close()
    open(test_jsonl, "w", encoding="utf-8").close()

    # ======== 生成训练集（cv 0,1,2 各自整合进去） ========
    total_train = 0
    for cv in train_folds:
        print(f"====== 处理训练集 cv = {cv} ======")
        ds = ECG_dataset(base_file=base_dir, cv=cv, is_train=False)
        total_train += append_dataset_to_jsonl(
            ds, train_jsonl, max_len_show=200, 
            encoder_model=encoder_model, use_encoder=use_encoder, model_type=model_type
        )
    print(f"\n>>> 训练集总样本数（cv0+1+2） = {total_train}\n")

    # ======== 生成测试集（cv 3,4 各自整合进去） ========
    total_test = 0
    for cv in test_folds:
        print(f"====== 处理测试集 cv = {cv} ======")
        ds = ECG_dataset(base_file=base_dir, cv=cv, is_train=False)
        total_test += append_dataset_to_jsonl(
            ds, test_jsonl, max_len_show=200,
            encoder_model=encoder_model, use_encoder=use_encoder, model_type=model_type
        )
    print(f"\n>>> 测试集总样本数（cv3+4） = {total_test}\n")
    print(f"{'='*60}")
    print("数据处理完成！")
    print(f"{'='*60}\n")


# ========= GUI 应用类 =========
class ECGDataProcessorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("ECG 数据处理工具")
        self.root.geometry("500x450")
        self.root.resizable(False, False)
        
        self.base_dir = None
        self.cnn_model = None
        self.mamba_model = None
        
        # 创建界面
        self.create_widgets()
    
    def create_widgets(self):
        # 标题
        title_label = tk.Label(
            self.root, 
            text="ECG 数据预处理工具", 
            font=("Arial", 16, "bold"),
            pady=20
        )
        title_label.pack()
        
        # 数据文件夹选择区域
        folder_frame = tk.LabelFrame(self.root, text="数据文件夹", padx=10, pady=10)
        folder_frame.pack(padx=20, pady=10, fill="x")
        
        self.folder_label = tk.Label(folder_frame, text="未选择文件夹", fg="gray")
        self.folder_label.pack(side="left", fill="x", expand=True)
        
        select_btn = tk.Button(
            folder_frame, 
            text="选择文件夹", 
            command=self.select_data_folder,
            padx=10
        )
        select_btn.pack(side="right")
        
        # 处理模式选择
        mode_frame = tk.LabelFrame(self.root, text="处理模式", padx=10, pady=10)
        mode_frame.pack(padx=20, pady=10, fill="x")
        
        # 按钮1：不使用编码器
        btn_no_encoder = tk.Button(
            mode_frame,
            text="生成数据（无编码器）",
            command=self.process_without_encoder,
            font=("Arial", 11),
            pady=8,
            width=22
        )
        btn_no_encoder.pack(pady=4)
        
        # 按钮2：使用 CNN
        btn_with_cnn = tk.Button(
            mode_frame,
            text="生成数据（使用 CNN）",
            command=self.process_with_cnn,
            font=("Arial", 11),
            pady=8,
            width=22
        )
        btn_with_cnn.pack(pady=4)
        
        # 按钮3：使用 Mamba
        btn_with_mamba = tk.Button(
            mode_frame,
            text="生成数据（使用 Mamba）",
            command=self.process_with_mamba,
            font=("Arial", 11),
            pady=8,
            width=22
        )
        btn_with_mamba.pack(pady=4)
        
        # 状态栏
        self.status_label = tk.Label(
            self.root,
            text="状态: 等待操作",
            relief="sunken",
            anchor="w"
        )
        self.status_label.pack(side="bottom", fill="x")
    
    def select_data_folder(self):
        """选择数据文件夹"""
        folder = filedialog.askdirectory(title="请选择 ECG 数据文件夹（包含 cv/ 和 training2017/）")
        if folder:
            self.base_dir = folder
            self.folder_label.config(text=folder, fg="black")
            self.status_label.config(text=f"状态: 已选择文件夹 {os.path.basename(folder)}")
    
    def process_without_encoder(self):
        """不使用编码器处理数据"""
        if not self.base_dir:
            messagebox.showwarning("警告", "请先选择 ECG 数据文件夹！")
            return
        
        self.status_label.config(text="状态: 正在处理数据（无编码器）...")
        self.root.update()
        
        try:
            process_data(self.base_dir, use_encoder=False)
            messagebox.showinfo("成功", "数据处理完成（无编码器）！\n输出目录: llm_data_no_encoder/")
            self.status_label.config(text="状态: 数据处理完成（无编码器）")
        except Exception as e:
            messagebox.showerror("错误", f"处理失败: {str(e)}")
            self.status_label.config(text="状态: 处理失败")
    
    def process_with_cnn(self):
        """使用 CNN 编码器处理数据"""
        if not self.base_dir:
            messagebox.showwarning("警告", "请先选择 ECG 数据文件夹！")
            return
        
        self.status_label.config(text="状态: 正在处理数据（使用 CNN）...")
        self.root.update()
        
        try:
            process_data(self.base_dir, use_encoder=True, encoder_model=self.cnn_model, model_type="cnn")
            messagebox.showinfo("成功", "数据处理完成（使用 CNN）！\n输出目录: llm_data_with_cnn/")
            self.status_label.config(text="状态: 数据处理完成（使用 CNN）")
        except Exception as e:
            messagebox.showerror("错误", f"处理失败: {str(e)}")
            self.status_label.config(text="状态: 处理失败")
    
    def process_with_mamba(self):
        """使用 Mamba 编码器处理数据"""
        if not self.base_dir:
            messagebox.showwarning("警告", "请先选择 ECG 数据文件夹！")
            return
        
        self.status_label.config(text="状态: 正在处理数据（使用 Mamba）...")
        self.root.update()
        
        try:
            process_data(self.base_dir, use_encoder=True, encoder_model=self.mamba_model, model_type="mamba")
            messagebox.showinfo("成功", "数据处理完成（使用 Mamba）！\n输出目录: llm_data_with_mamba/")
            self.status_label.config(text="状态: 数据处理完成（使用 Mamba）")
        except Exception as e:
            messagebox.showerror("错误", f"处理失败: {str(e)}")
            self.status_label.config(text="状态: 处理失败")


def main():
    """启动 GUI 应用"""
    root = tk.Tk()
    app = ECGDataProcessorGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
