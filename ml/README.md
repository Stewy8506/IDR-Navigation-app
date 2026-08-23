# IDR-Nav — ML Speed & Vibration Estimator Pipeline

This directory contains the Python training, evaluation, and export pipeline for the AI speed & vibration filter.

## Directory Structure

```text
ml/
├── data/
│   └── IO-VNBD/              <--- Place your extracted IO-VNBD dataset files here
│       ├── Train/
│       └── Test/
├── external/                 <--- Place external/reference repos (like cloned IO-VNBD) here
├── notebooks/                <--- Jupyter notebooks for data analysis & drift plots
├── src/
│   ├── __init__.py
│   ├── dataset.py            # IO-VNBD data loader & sliding window generator
│   ├── model.py              # 1D-CNN / TCN speed estimator architecture
│   ├── train.py              # Model training script
│   ├── evaluate.py           # Evaluation on test drives & error metrics
│   └── export_tflite.py      # Exports model to .tflite / ONNX for Flutter
└── requirements.txt
```

## Setup Instructions

1. Create a Python virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. Place the IO-VNBD dataset in `ml/data/IO-VNBD/`.

3. Train the model:
   ```bash
   python -m ml.src.train --data_dir ml/data/IO-VNBD
   ```

4. Export to TFLite for Flutter:
   ```bash
   python -m ml.src.export_tflite --weights weights/best_model.pt --output ../app/assets/models/speed_filter.tflite
   ```
