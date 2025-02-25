# Cat Detection Project

## 📌 Overview
This project is designed to detect cats in images using deep learning models.  
It leverages **pre-trained models (`MobileNetV1`, `EfficientNetB0`, `DenseNet121`)**  
from **Keras 3 with PyTorch backend** for high efficiency and accuracy.

## 📂 Project Structure
```
/CatDetection/
│   
README.md
└───main.py
└───balance_dataset.py
```

## Requirements
- Python **3.12**
- PyTorch + torchvision + Keras 3  
- Other dependencies

## Setup
1. Clone the repository
2. Install main dependencies: PyTorch, Keras, torchvision
3. Install other dependencies:
```bash
pip install pillow, matplotlib, sklearn, numpy
```
4. Download dataset manually (cats and not cats) and structure it:
```bash
dataset/
└───train/
    └───cats/
    └───not_cats/
└───validation/
    └───cats/
    └───not_cats/
└───test/
    └───cats/
    └───not_cats/
```
💡 Tip: If your dataset is imbalanced, run:
```bash
python balance_dataset.py
```
to make cats and not_cats 50/50.

## Usage
1. Train a specific model:
```bash
python main.py train EfficientNetB0
```
2. Train all models:
```bash
python main.py train_all
```
3. Run GUI for testing:
```bash
python main.py gui <model_name>
```

## Contributing
Feel free to submit issues and pull requests.

## License
MIT License
