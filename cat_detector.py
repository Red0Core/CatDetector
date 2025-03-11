import os
import sys
os.environ["KERAS_BACKEND"] = "torch"
import keras  # Используем Keras 3 с PyTorch Backend
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import time
import tkinter as tk
from tkinter import filedialog

# Устройство (CPU/GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
emoji_of_videocard = "🔥🔥🔥" if torch.cuda.is_available() else "🥔🥔🥔"
print(f"{emoji_of_videocard} {device} is using")

# Используем стандартные параметры нормализации для предобученных моделей
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

model_densenet = keras.models.load_model("010_0.993363_DenseNet121_best_version.keras")
model_mobilenet = keras.models.load_model("best_hand_MobileNet_epoch_6.keras")
model_efficient = keras.models.load_model("016_0.733407_EfficientNetB0_best_version.keras")
def select_file():
    file_path = filedialog.askopenfilename(
        title="Выберите изображение",
        filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")]
    )
    if file_path:
        # Загружаем и обрабатываем изображение
        img = Image.open(file_path).convert("RGB")
        img = transform(img).unsqueeze(0).to(device)
        start = time.time()
        with torch.no_grad():
            output_densenet = model_densenet.predict(img).item()
            output_mobilenet = model_mobilenet.predict(img).item()
            output_efficient = model_efficient.predict(img).item()
        elapsed = (time.time() - start) * 1000  # время в мс
        result = [output_efficient > 0.6, output_mobilenet > 0.85, output_densenet > 0.85]
        cat_str = ""
        if all(result):
            cat_str = "Котик 🐱"
        elif result.count(True) == 2:
            cat_str = "Вероятно, котик 2/3 🐱"
        else:
            cat_str = "Не котик"
        result_label.config(text=
                            f"{cat_str}\n"
                            f"Вероятность DenseNet(кот): {output_densenet*100:.2f}%\n"
                            f"Вероятность MobileNet(кот): {output_mobilenet*100:.2f}%\n"
                            f"Вероятность EfficientNetB0(кот): {output_efficient*100:.2f}%\n"
                            f"Время: {elapsed:.2f} мс")
root = tk.Tk()
root.title("Cat Detector GUI")
root.geometry("400x200")
btn = tk.Button(root, text="Выбрать изображение", command=select_file, width=25, height=2)
btn.pack(pady=20)
result_label = tk.Label(root, text="Результат появится здесь", font=("Helvetica", 12))
result_label.pack(pady=10)
root.mainloop()
