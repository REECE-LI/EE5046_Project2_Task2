import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import firwin, lfilter, iirnotch, filtfilt
from sklearn.metrics import roc_auc_score, f1_score
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import csv
import os

# =========================================================
# ============= ECG Preprocessing (论文一致) ================
# =========================================================

def preprocess_ecg(data, orig_sr=300, target_sr=120):
    # ------ 60 Hz low-pass filter (论文标准) ------
    lp_cutoff = 60 / (orig_sr / 2)
    # lp_fir = firwin(numtaps=501, cutoff=lp_cutoff, pass_zero=True)
    lp_fir = firwin(numtaps=101, cutoff=lp_cutoff, pass_zero=True)

    data = filtfilt(lp_fir, 1.0, data)

    # ------ 300 → 150 → 120 Hz 下采样 ------
    data = data[::2]  # 300 →150
    t_old = np.linspace(0, 1, len(data))
    t_new = np.linspace(0, 1, int(len(data) * target_sr / 150))
    data = np.interp(t_new, t_old, data)

    # ------ Normalization ------
    data = (data - data.mean()) / (data.std() + 1e-6)

    return data.astype(np.float32)



# =========================================================
# ====================== DATA LOADER ======================
# =========================================================

class ECGDataset(Dataset):
    def __init__(self, csv_path, ecg_dir, max_len=600):
        self.df = pd.read_csv(csv_path)
        self.ecg_dir = ecg_dir
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        name = self.df.iloc[idx]["file_name"]
        label = 1 if self.df.iloc[idx]["label"] == "A" else 0

        path = f"{self.ecg_dir}/{name}.mat"
        sig = loadmat(path)["val"][0]
        data = preprocess_ecg(sig)

        # Padding / Truncation to fixed length
        if len(data) >= self.max_len:
            data = data[:self.max_len]
        else:
            pad = np.zeros(self.max_len, dtype=np.float32)
            pad[:len(data)] = data
            data = pad

        return torch.tensor(data).unsqueeze(0), torch.tensor(label).float()


def load_fold(fold_id, ecg_dir, batch_size=32):
    cvs = [f"cv{i}.csv" for i in range(5)]

    train_ds = ConcatDataset([
        ECGDataset(cvs[i], ecg_dir) for i in range(5) if i != fold_id
    ])
    test_ds = ECGDataset(cvs[fold_id], ecg_dir)

    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    )


# =========================================================
# ====================== MODEL ============================
# =========================================================

class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, k):
        super().__init__()
        self.conv = nn.Conv1d(in_c, out_c, k, padding=k//2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv(x))


class Stream(nn.Module):
    def __init__(self, kernels):
        super().__init__()
        cin  = [1,64,64,128,128,256,256,256,512,512,512,512,512]
        cout = [64,64,128,128,256,256,256,512,512,512,512,512,512]

        self.blocks = nn.ModuleList([
            ConvBlock(cin[i], cout[i], kernels[i]) for i in range(13)
        ])

        self.pool_pos = [2,4,7,10,13]
        self.pools = nn.ModuleList([
            nn.MaxPool1d(3, stride=3),
            nn.MaxPool1d(2, stride=2),
            nn.MaxPool1d(2, stride=2),
            nn.MaxPool1d(2, stride=2),
            nn.MaxPool1d(2, stride=2)
        ])

    def forward(self, x):
        pid = 0
        for i, blk in enumerate(self.blocks, 1):
            x = blk(x)
            if i in self.pool_pos:
                x = self.pools[pid](x)
                pid += 1
        return x


class MSCNN(nn.Module):
    def __init__(self, input_len=600, use_stream2=True):
        super().__init__()

        self.use_stream2 = use_stream2
        self.stream1 = Stream([3]*13)              # 单尺度
        if use_stream2:
            self.stream2 = Stream([7]*4 + [3]*9)   # 多尺度

        self.flat_dim = self._get_flat_dim(input_len)

        self.fc = nn.Sequential(
            nn.Linear(self.flat_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def _get_flat_dim(self, L):
        x = torch.randn(1,1,L)
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


# =========================================================
# ========================== TRAIN =========================
# =========================================================

def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0

    for x, y in loader:
        x, y = x.cuda(), y.cuda().unsqueeze(1)

        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model, loader):
    model.eval()
    preds, gts = [], []

    with torch.no_grad():
        for x, y in loader:
            x = x.cuda()
            out = model(x).cpu().numpy().flatten()
            preds.extend(out)
            gts.extend(y.numpy())

    auc = roc_auc_score(gts, preds)
    f1  = f1_score(gts, (np.array(preds) > 0.5).astype(int))
    return auc, f1, preds, gts


# =========================================================
# ====================== MAIN TRAINING =====================
# =========================================================

if __name__ == "__main__":

    ecg_dir = "../training2017"
    os.makedirs("results", exist_ok=True)

    EPOCHS = 100              # 原论文 20–30，适度增加
    patience = 20            # 早停机制

    preds_base_all, gts_base_all = [], []
    preds_ms_all, gts_ms_all = [], []
    loss_base_all, loss_ms_all = [], []

    for fold in range(5):

        print(f"\n=========== Fold {fold} ===========")
        train_loader, test_loader = load_fold(fold, ecg_dir, batch_size=32)

        # ----------------- Baseline CNN -----------------
        model_base = MSCNN(use_stream2=False).cuda()
        optimizer = torch.optim.Adam(model_base.parameters(), lr=1e-4)
        criterion = nn.BCELoss()

        loss_list_base = []
        best_auc = 0
        best_epoch = 0

        for ep in range(EPOCHS):
            loss = train_one_epoch(model_base, train_loader, optimizer, criterion)
            loss_list_base.append(loss)

            auc_b, f1_b, preds_b, gts_b = evaluate(model_base, test_loader)

            print(f"[Baseline] Epoch {ep+1}/{EPOCHS} Loss={loss:.4f} AUC={auc_b:.4f}")

            # Early stopping
            if auc_b > best_auc:
                best_auc = auc_b
                best_epoch = ep
                torch.save(model_base.state_dict(), f"results/baseline_fold{fold}.pth")
            elif ep - best_epoch > patience:
                print("Baseline early stopping!")
                break

        preds_base_all.append(preds_b)
        gts_base_all.append(gts_b)
        loss_base_all.append(loss_list_base)

        # ----------------- MS-CNN -----------------
        model_ms = MSCNN(use_stream2=True).cuda()
        optimizer = torch.optim.Adam(model_ms.parameters(), lr=1e-4)

        loss_list_ms = []
        best_auc = 0
        best_epoch = 0

        for ep in range(EPOCHS):
            loss = train_one_epoch(model_ms, train_loader, optimizer, criterion)
            loss_list_ms.append(loss)

            auc_m, f1_m, preds_m, gts_m = evaluate(model_ms, test_loader)
            print(f"[MS-CNN] Epoch {ep+1}/{EPOCHS} Loss={loss:.4f} AUC={auc_m:.4f}")

            if auc_m > best_auc:
                bestauc = auc_m
                best_epoch = ep
                torch.save(model_ms.state_dict(), f"results/mscnn_fold{fold}.pth")
            elif ep - best_epoch > patience:
                print("MS-CNN early stopping!")
                break

        preds_ms_all.append(preds_m)
        gts_ms_all.append(gts_m)
        loss_ms_all.append(loss_list_ms)

        # 保存每折结果
        np.savez(f"results/fold{fold}_msc_pred_label.npz", preds=preds_m, gts=gts_m)
        np.savez(f"results/fold{fold}_base_pred_label.npz", preds=preds_b, gts=gts_b)

        # Loss 曲线保存
        plt.figure(figsize=(6,4))
        plt.plot(loss_list_base, label="Baseline")
        plt.plot(loss_list_ms, label="MS-CNN")
        plt.title(f"Loss Curve Fold {fold}")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.legend()
        plt.savefig(f"results/loss_fold{fold}.png")
        plt.close()

    # 保存所有数据用于 plot script
    np.savez("results/all_data.npz",
             preds_base_all=preds_base_all,
             preds_ms_all=preds_ms_all,
             gts_base_all=gts_base_all,
             gts_ms_all=gts_ms_all,
             loss_base_all=loss_base_all,
             loss_ms_all=loss_ms_all)

    print("\nTraining complete. All results saved in /results/")
