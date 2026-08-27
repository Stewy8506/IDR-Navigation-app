# Kaggle GPU Training Guide: Experiment 6A

This guide explains how to run **Experiment 6A (Closed-Loop State-Conditioned Velocity Observer)** on a Kaggle GPU (e.g. NVIDIA T4 or P100) in under **10 minutes** for all 15 epochs.

---

## 1. Dataset Setup on Kaggle

1. Go to [Kaggle](https://www.kaggle.com/) -> **Datasets** -> **New Dataset**.
2. Upload the `Categorised IOVNB Dataset` folder (located locally at `ml/external/IO-VNBD_repo/Synchronised V abd S datasets/Categorised IOVNB Dataset/`) or upload a zip archive of it.
3. Title the dataset: `iovnb-dataset`.

---

## 2. Running in a Kaggle Notebook

1. In Kaggle, create a **New Notebook**.
2. Under **Notebook settings** (right panel):
   - **Accelerator:** Select **GPU T4 x2** or **GPU P100**.
   - **Persistence:** Files / Output enabled.
3. Attach the `iovnb-dataset` under **Input Data** (it will mount at `/kaggle/input/iovnb-dataset/`).

### Notebook Cell 1: Clone or Upload Code
If cloning this repository:
```bash
!git clone https://github.com/Stewy8506/INSS-Navigation-app.git /kaggle/working/app
%cd /kaggle/working/app
```
*(Or upload the `ml/` folder directly to the notebook).*

### Notebook Cell 2: Run Smoke Test
```bash
!python ml/kaggle/train_exp6a_kaggle.py --smoke-test --output-dir /kaggle/working
```

### Notebook Cell 3: Launch Full 15-Epoch Training
```bash
!python ml/kaggle/train_exp6a_kaggle.py \
    --data-dir "/kaggle/input/iovnb-dataset/Categorised IOVNB Dataset" \
    --output-dir "/kaggle/working" \
    --epochs 15 \
    --batch-size 64 \
    --lr 1e-3 \
    --seq-len 32 \
    --seq-stride 16
```

---

## 3. GPU Expected Performance & Profiles

| Hardware | Batch Size | Time / Epoch | Total 15 Epochs | Samples / Sec | VRAM Usage |
|---|:---:|:---:|:---:|:---:|:---:|
| **Kaggle NVIDIA T4 (16GB)** | 64 | **~35 seconds** | **~8.5 minutes** | ~26,000 smp/s | ~1.4 GB |
| **Kaggle NVIDIA P100 (16GB)**| 64 | **~25 seconds** | **~6.0 minutes** | ~36,000 smp/s | ~1.4 GB |
| **Apple M-Series (MPS)** | 32 | ~420 seconds | ~105 minutes | ~2,200 smp/s | Shared RAM |

---

## 4. Expected Output Files & Checkpoint Download

The training runner writes all artifacts directly to `/kaggle/working`:
* **`exp6a_best_spectral_speed_filter.pt`**: Best model checkpoint according to the frozen **8-bin balanced MAE** criterion on Driver A S3a.
* **`exp6a_final_spectral_speed_filter.pt`**: Final model state at Epoch 15.
* **`exp6a_history.json`**: Complete machine-readable epoch-by-epoch loss, MAE, speed-bin metrics, and training speed logs.

### Downloading from Kaggle to Local:
Once the notebook run finishes:
1. In the right panel under **Output**, click the three dots next to `exp6a_best_spectral_speed_filter.pt` -> **Download**.
2. Place the downloaded checkpoint in your local repository at:
   `ml/weights/exp6a_best_spectral_speed_filter.pt` and `ml/weights/best_spectral_speed_filter.pt`.
3. Run the local evaluation suite:
   ```bash
   python ml/src/evaluate_experiment6a.py
   ```
