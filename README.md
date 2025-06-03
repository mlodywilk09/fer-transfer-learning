# Facial Emotion Recognition (FER) — Transfer Learning with InceptionResNetV2

## Project Overview
This project aims to classify facial expressions into **positive (happy)** and **negative (angry, fearful, sad)** emotions using transfer learning on the **FER2013** dataset. A binary classification task is performed using a pretrained **InceptionResNetV2** model as a feature extractor, fine-tuned with custom dense layers.

---

## Dataset
- **FER2013**: A popular facial emotion recognition dataset with grayscale 48x48 pixel images.
- Selected emotions:
  - Positive class: Happiness (emotion label 3)
  - Negative class: Anger (0), Fear (2), Sadness (4)
- Balanced dataset with 8500 samples per class for training and validation.

---

## Model Architecture
- Base model: **InceptionResNetV2** pretrained on ImageNet, excluding the top classifier layers (`include_top=False`).
- Input shape adapted to `(139, 139, 3)` RGB images (converted from grayscale).
- Base model weights frozen during training.
- Added classifier head with:
  - GlobalAveragePooling2D
  - Dense layers with L2 regularization and Dropout for regularization
  - Output layer with sigmoid activation for binary classification

---

## Preprocessing
- Raw pixel strings from FER2013 dataset converted to 48x48 images.
- Images resized to 139x139 pixels and converted to 3-channel RGB by repeating the grayscale channel.
- Preprocessing includes normalization via `preprocess_input` from InceptionResNetV2.

---

## Training
- Loss function: Binary Crossentropy
- Optimizer: Adam (learning rate = 0.001)
- Early stopping on validation loss with patience of 10 epochs.
- Batch size: 128, max epochs: 50
- Dataset split:
  - 60% training
  - 20% cross-validation
  - 20% test

---

## Results on Validation Set
- Accuracy: ~79.4%
- Precision: ~80.7%
- Recall: ~77.5%
- F1 Score: ~79.0%

Confusion matrix visualization shows balanced classification performance.

---

## Dependencies
- Python 3.x  
- TensorFlow / Keras  
- Pandas, NumPy, Matplotlib  
- scikit-learn

---

## How to Use
1. Clone the repository.
2. Download the `fer2013.csv` dataset and place it in the project directory.
3. Run the Jupyter notebook or Python scripts to preprocess data, train the model, and evaluate results.
4. Modify parameters or model layers as needed for experimentation.

---

## License
This project is licensed under the [MIT License](LICENSE).
