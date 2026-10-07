# Comparative Analysis of Backpropagation Neural Networks and Support Vector Machines for Handwritten Digit and Letter Recognition

## Problem statement and objectives

Recognizing handwriting is difficult because each person writes differently. This mini project classifies single digits **0 through 9** and single English letters **A through Z**. It compares a backpropagation neural network (MLP), a support vector machine (SVM), and a convolutional neural network (CNN). The web page lets you choose digit or letter mode, upload one handwritten character, and see all available models' predictions. It does not read words or distinguish letter case.

The objectives are to train the three models on MNIST for digits and EMNIST Letters for letters, evaluate models on the matching test set, compare accuracy and other metrics, and test them on a custom image. This is an educational comparison, not a production handwriting reader.

## Technologies and dataset

Python 3, TensorFlow/Keras, TensorFlow Datasets, scikit-learn, NumPy, Pandas, OpenCV, Matplotlib, Joblib, Pillow, and Streamlit are used. MNIST is loaded by `tensorflow.keras.datasets.mnist.load_data()` and has **60,000 training images** and **10,000 test images**. EMNIST Letters is loaded by `tensorflow_datasets` and provides **88,800 training images** and **14,800 test images** in its TensorFlow Datasets split. Uppercase and lowercase handwriting is merged into 26 letter classes. Both datasets contain 28 × 28 grayscale character images. The first training run may download and cache the datasets; EMNIST's source download is larger than its prepared Letters split. See the [TensorFlow Datasets catalog](https://www.tensorflow.org/datasets/catalog/emnist) and [NIST's EMNIST description](https://www.nist.gov/itl/products-and-services/emnist-dataset).

## System architecture

```text
MNIST digits / EMNIST Letters → normalization → MLP, SVM, CNN
                             → separate saved models → matching test evaluation
Uploaded image → grayscale/invert/crop/center/normalize → three model inputs
             → predictions + confidences → majority vote
```

For each task, the three models use the same test set. The SVM trains on a repeatable random sample of 20,000 training images by default because full RBF SVM training is slow. When comparing scores, remember that its training set is smaller unless you change `--train-size`. Digit and letter scores belong to different datasets and should be compared separately.

## Model descriptions

**MLP (backpropagation neural network):** A Flatten layer changes 28 × 28 pixels into 784 values. Dense layers with 128 and 64 ReLU neurons learn patterns, dropout of 0.2 reduces overfitting, and a softmax layer outputs 10 digit or 26 letter probabilities. It uses Adam at learning rate 0.001, sparse categorical crossentropy, 18 epochs by default, batch size 128, and a 10% validation split.

**How backpropagation works:** During forward propagation, the input passes through each layer to produce a prediction. A loss function measures the difference between the prediction and the correct digit. During backward propagation, gradients of the loss with respect to the weights are calculated. The optimizer updates weights using the idea `new_weight = old_weight - learning_rate * gradient`. This repeats over batches and epochs until the network improves.

**SVM:** The image is flattened into 784 normalized features. The RBF kernel helps form a nonlinear boundary between digit classes. `C=10` controls the penalty for mistakes; `gamma="scale"` sets the kernel scale from the data. `probability=True` enables probability estimates for the app. It also makes training slower. The SVM is trained with a reproducible random sample (`random_state=42`).

**CNN:** Two 3 × 3 convolution layers (32 and 64 filters), each followed by 2 × 2 max pooling, learn local stroke patterns. A 128-neuron dense layer and 0.3 dropout lead to a 10-class digit or 26-class letter softmax output. It uses Adam, sparse categorical crossentropy, 12 epochs by default, batch size 128, and a 10% validation split.

## Custom image preprocessing

An uploaded image is converted to grayscale. Transparent images are placed on white. The border pixels estimate whether its background is light or dark, and the foreground is inverted when necessary so it resembles the datasets' light character on a dark background. OpenCV blurs and thresholds the image, removes tiny components, finds the character bounding box, crops it, scales its longest side to 20 pixels while preserving its aspect ratio, and centers it on a black 28 × 28 canvas. Pixel values are divided by 255. Blank images show a message instead of a prediction. EMNIST images are transposed into upright orientation before training and evaluation. Upload one clear character at a time; textured backgrounds or multiple characters may not work well.

## Installation

Use **Python 3.10–3.12** for broad TensorFlow compatibility. From this project directory:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On macOS/Linux, activate with `source .venv/bin/activate`. On Windows PowerShell, if activation is blocked by local policy, invoke `.venv\Scripts\python.exe` in place of `python` for the commands below.

## Training and running

Run these commands from the folder containing `app.py`. Commands without `--task` are for digits:

```powershell
python -m src.train_mlp
python -m src.train_svm
python -m src.train_cnn
python -m src.evaluate_models
streamlit run app.py
```

Train and evaluate the three **letter** models separately:

```powershell
python -m src.train_mlp --task letter
python -m src.train_svm --task letter
python -m src.train_cnn --task letter
python -m src.evaluate_models --task letter
streamlit run app.py
```

Training creates separate model files: `mlp_model.keras`, `svm_model.pkl`, and `cnn_model.keras` for digits, and `letter_mlp_model.keras`, `letter_svm_model.pkl`, and `letter_cnn_model.keras` for letters. Evaluation creates `results/metrics.csv` for digits and `results/metrics_letters.csv` for letters, plus task-specific classification reports and confusion matrix images. MLP and CNN training also create accuracy and loss graphs. The app can start before training, but prediction requires at least one saved model for the selected mode. If files change while the app is running, restart Streamlit to reload its cached models.

For a quick development check, use smaller samples and fewer epochs:

```powershell
python -m src.train_mlp --train-size 1000 --epochs 1
python -m src.train_svm --train-size 1000
python -m src.train_cnn --train-size 1000 --epochs 1
python -m src.train_mlp --task letter --train-size 1000 --epochs 1
python -m src.train_svm --task letter --train-size 1000
python -m src.train_cnn --task letter --train-size 1000 --epochs 1
```

These quick models are for checking the workflow, **not** for reporting final comparative results. Retrain with the default settings before presenting metrics. You may increase the SVM sample size if you have enough time and memory.

## Evaluation metrics

`Accuracy` is the fraction of correct predictions. `Precision` measures how often a predicted class is correct. `Recall` measures how many actual examples of a class are found. `F1 Score` balances precision and recall. Each CSV uses **macro averages** for precision, recall, and F1, giving each class equal weight. It also stores training time, total inference time over its task's test set, and the number of training samples. A confusion matrix shows actual versus predicted classes, while each classification report gives per-class details. Training times depend on the computer and are saved when each training script runs.

## Expected project structure

```text
handwritten-digit-recognition/
├── app.py
├── requirements.txt
├── README.md
├── models/                 # generated digit and letter .keras/.pkl files
├── results/
│   ├── metrics.csv         # generated digit evaluation
│   ├── metrics_letters.csv # generated letter evaluation
│   ├── confusion_matrices/
│   └── training_plots/
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── train_mlp.py
│   ├── train_svm.py
│   ├── train_cnn.py
│   ├── evaluate_models.py
│   └── predict.py
└── samples/custom_digits/  # optional images to upload in the app
```

Generated model binaries and results are absent until training and evaluation run. They are ignored by Git because they can be large or machine specific.

## Future enhancements

Possible improvements include a drawing canvas, word recognition with a separate sequence model and word-level dataset, data augmentation for custom handwriting, improved centering based on image moments, confidence calibration, and a larger study that gives all models the same training sample size. The current interface uses image upload because it works across Streamlit versions without an extra canvas component.
