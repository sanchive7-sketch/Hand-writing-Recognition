# Setup and run the handwriting recognition app

This project recognizes **one handwritten digit (0–9) or one English letter (A–Z) per image**. It does not recognize words. The app can open before training, but predictions require saved models for the selected mode.

## 1. Open PowerShell in the project folder

```powershell
cd "C:\Users\Sanchive Kumar\handwriting Recoganition\handwritten-digit-recognition"
```

If you moved the project, replace that path with its new location. Run all later commands from the folder containing `app.py`.

## 2. Check Python and create a virtual environment

Use Python **3.10, 3.11, or 3.12**.

```powershell
python --version
python -m venv .venv
```

These commands use the virtual environment's Python directly, so PowerShell activation is not required. If `python` is not found, install a supported Python version and reopen PowerShell.

## 3. Install dependencies

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The first training command may download and cache MNIST or EMNIST. Keep an internet connection available for that first run. TensorFlow installation and model training can take time.

## 4. Train the models you want to use

### Digits (0–9)

```powershell
.\.venv\Scripts\python.exe -m src.train_mlp
.\.venv\Scripts\python.exe -m src.train_svm
.\.venv\Scripts\python.exe -m src.train_cnn
.\.venv\Scripts\python.exe -m src.evaluate_models
```

### Letters (A–Z)

```powershell
.\.venv\Scripts\python.exe -m src.train_mlp --task letter
.\.venv\Scripts\python.exe -m src.train_svm --task letter
.\.venv\Scripts\python.exe -m src.train_cnn --task letter
.\.venv\Scripts\python.exe -m src.evaluate_models --task letter
```

You may train only one model to start using a mode; the app will identify the missing models. For the full three-model comparison, run all three training commands for that mode and then its evaluation command. The SVM uses 20,000 training images by default; its full-dataset training can be slow.

Training saves models in `models/`. Evaluation saves digit results to `results/metrics.csv`, letter results to `results/metrics_letters.csv`, and graphs and confusion matrices under `results/`.

## 5. Launch the web app

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open the local address printed in PowerShell, usually `http://localhost:8501`. Choose **Digit (0–9)** or **Letter (A–Z)**, upload an image with one clear handwritten character, and select **Predict Digit** or **Predict Letter**. Press `Ctrl+C` in PowerShell to stop the app.

## Common problems

- **`No module named tensorflow` or another missing package:** Run the dependency installation command in step 3 using `.\.venv\Scripts\python.exe`.
- **No trained models found:** Run at least one training command for the mode selected in the app. Restart Streamlit after training so it reloads the saved models.
- **No comparison table or graphs:** Run the evaluation command for the selected mode after training.
- **Blank image warning:** Upload a clear image containing one character with strong contrast against its background.
- **EMNIST download fails:** Check your connection and retry the letter training command; TensorFlow Datasets handles the download and cache.
- **Port 8501 is busy:** Start Streamlit with `--server.port 8502` and open the address it prints.

For model descriptions, preprocessing details, and evaluation metrics, see [README.md](README.md).
