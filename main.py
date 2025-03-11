import os
import sys
os.environ["KERAS_BACKEND"] = "torch"
import keras  # Используем Keras 3 с PyTorch Backend
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import time
import tkinter as tk
from tkinter import filedialog
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

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

class CatDataset(Dataset):
    """
    Класс для загрузки датасета с котами и не котами
    """
    def __init__(self, root_dir):
        """
        Ожидается, что в root_dir находятся две папки: 'cats' и 'not_cats'
        """
        self.root_dir = root_dir
        self.image_paths = []
        self.labels = []
        # Категории: label 0 -> not_cats, label 1 -> cats
        for label, category in enumerate(["not_cats", "cats"]):
            category_dir = os.path.join(root_dir, category)
            for filename in os.listdir(category_dir):
                filepath = os.path.join(category_dir, filename)
                self.image_paths.append(filepath)
                self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert("RGB")
        image = transform(image)
        return image, torch.tensor(label, dtype=torch.float32)

def build_model(model_name: str) -> keras.Model:
    """
    Возвращает модель для обучения на основе его названия
    """
    inputs = keras.layers.Input(shape=(3, 224, 224))
    x = keras.layers.Permute((2, 3, 1))(inputs)

    if model_name == "MobileNet":
        base_model = keras.applications.MobileNet(input_shape=(224, 224, 3), include_top=False, input_tensor=x, weights="imagenet")
        dropout_rate = 0.2
        learning_rate = 0.001
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

    elif model_name == "EfficientNetB0":
        base_model = keras.applications.EfficientNetB0(input_shape=(224, 224, 3), include_top=False, input_tensor=x, weights="imagenet")
        dropout_rate = 0.3
        learning_rate = 0.0005
        optimizer = keras.optimizers.RMSprop(learning_rate=learning_rate)

    elif model_name == "DenseNet121":
        base_model = keras.applications.DenseNet121(input_shape=(224, 224, 3), include_top=False, input_tensor=x, weights="imagenet")
        dropout_rate = 0.5
        learning_rate = 0.0003
        optimizer = keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)

    else:
        raise ValueError("Неизвестная модель")

    base_model.trainable = False

    x = keras.layers.GlobalAveragePooling2D()(base_model.output)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(dropout_rate)(x)
    outputs = keras.layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(inputs, outputs, name=model_name)
    model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])

    return model


def test_model(model: keras.Model, test_loader: DataLoader, filename) -> float:
    """
    Тестирует модель на тестовой выборке и выводит метрики
    """
    predictions = []
    true_labels = []
    
    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model.predict(images).flatten()
            preds = outputs  # В model уже есть sigmoid
            predictions.extend(preds)
            true_labels.extend(labels.numpy().flatten())

    predictions_rounded = np.round(predictions)

    # 🔹 Оцениваем метрики
    accuracy = accuracy_score(true_labels, predictions_rounded)
    precision = precision_score(true_labels, predictions_rounded, zero_division=0)
    recall = recall_score(true_labels, predictions_rounded, zero_division=0)
    f1 = f1_score(true_labels, predictions_rounded, zero_division=0)
    roc_auc = roc_auc_score(true_labels, predictions)

    print(f"\n✅ Test Accuracy: {accuracy:.4f}")
    print(f"🎯 Precision: {precision:.4f}")
    print(f"🔍 Recall: {recall:.4f}")
    print(f"📊 F1-score: {f1:.4f}")
    print(f"📈 ROC-AUC: {roc_auc:.4f}")

    cm = confusion_matrix(true_labels, predictions_rounded)

    plt.figure(figsize=(5, 4))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ["Not Cat", "Cat"])
    plt.yticks(tick_marks, ["Not Cat", "Cat"])

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")

    # 🔹 Подписи значений внутри ячеек
    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], horizontalalignment="center", verticalalignment="center", color="black")

    plt.savefig(filename)

    return accuracy


def plot_hist(hist: keras.callbacks.History, filename):
    """
    Строит графики обучения модели (accuracy и loss)
    """
    plt.plot(hist.history["accuracy"])
    plt.plot(hist.history["val_accuracy"])
    plt.title("model accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(["train", "validation"], loc="upper left")
    plt.savefig(filename)

def create_gui(model: keras.Model):
    """
    GUI для загрузки изображения и предсказания на уже натренировнной модели
    """
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
                output = model.predict(img).item()
            elapsed = (time.time() - start) * 1000  # время в мс
            probability = output * 100
            result_label.config(text=f"Вероятность (кот): {probability:.2f}%\nВремя: {elapsed:.2f} мс")
    root = tk.Tk()
    root.title("Cat Detector GUI")
    root.geometry("400x200")
    btn = tk.Button(root, text="Выбрать изображение", command=select_file, width=25, height=2)
    btn.pack(pady=20)
    result_label = tk.Label(root, text="Результат появится здесь", font=("Helvetica", 12))
    result_label.pack(pady=10)
    root.mainloop()

def get_data_loaders(model_name: str) -> tuple[DataLoader]:
    """
    Грузит данные и возвращает DataLoader для обучения, валидации и тестирования
    """
    # Пути к данным
    train_dir = "dataset/train"
    val_dir = "dataset/validation"
    test_dir = "dataset/test"
    
    # Гиперпараметры
    batch_size_dict = {"MobileNet": 32, "EfficientNetB0": 16, "DenseNet121": 16}
    batch_size = batch_size_dict[model_name]

    # Создаем датасеты
    train_dataset = CatDataset(train_dir)
    val_dataset = CatDataset(val_dir)
    test_dataset = CatDataset(test_dir)
    
    # DataLoader для загрузки данных батчами
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, val_loader, test_loader

class SaveEpochCallback(keras.callbacks.Callback):
    """
    Callback для сохранения модели после каждой эпохи с указанием номера эпохи в имени файла
    """
    def __init__(self, model_name):
        super().__init__()
        self.model_name = model_name
        
    def on_epoch_end(self, epoch, logs=None):
        # Сохраняем модель после каждой эпохи
        filename = f"{self.model_name}_epoch_{epoch}.keras"
        self.model.save(filename)
        print(f"\nМодель сохранена как {filename}\n\n")

def train(model: keras.Model, epochs: int, train_loader: DataLoader, val_loader: DataLoader, callbacks):
    """
    Обучает модель и возвращает время обучения и историю обучения
    """
    start_time = time.time()
    hist = model.fit(train_loader, epochs=epochs, validation_data=val_loader, callbacks=callbacks)
    training_time = time.time() - start_time
    
    return training_time, hist

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(f"main.py train <model> - для обучения модели\nmain.py gui <model> - для запуска GUI\nmain.py train_all - для обучения всех моделей")
        sys.exit(0)

    if 'train' in sys.argv[1]:
        if len(sys.argv) < 3 and 'train_all' != sys.argv[1]:
            print("Укажите модель для обучения")
            sys.exit(1)
        elif 'train_all' == sys.argv[1]:
            models = ["MobileNet", "EfficientNetB0", "DenseNet121"]
        else:
            models = [sys.argv[2]]
        
        for model_name in models:
            print(f"Обучаю модель {model_name}")
            
            epochs_dict = {"MobileNet": 15, "EfficientNetB0": 20, "DenseNet121": 25}
            epochs = epochs_dict[model_name]
            train_loader, val_loader, test_loader = get_data_loaders(model_name)
            print("Загружены данные")


            # Добавляем callback для сохранения лучшей модели
            best_model_callback = keras.callbacks.ModelCheckpoint(
                filepath="{epoch:03d}_{val_accuracy:03f}_"+f"{model_name}_best_version.keras",
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            )
            model = build_model(model_name)
            training_time, hist = train(model, epochs, train_loader, val_loader, callbacks=[SaveEpochCallback(model_name), best_model_callback])
            print(f"Время обучения {model_name}: {training_time:.2f} сек")

            plot_hist(hist, f"Accuracy_Loss_{model_name}_{epochs}.png")
        
            # Тестирование модели
            accuracy = test_model(model, test_loader, f"Test_Model_Function_{model_name}_{epochs}.png")
            print(f"Точность на тестовой выборке: {accuracy:.4f}")

            model_filename = f"{model_name}.keras"
            model.save(model_filename)
            print(f"Успешно сохранена модель в {model_filename}")
        
    elif 'gui' == sys.argv[1]:
        if len(sys.argv) < 3:
            print("Укажите модель для GUI")
            sys.exit(1)

        model_name = sys.argv[2]
        is_started = False
        for model_filename in os.listdir("."):
            if 'best' in model_filename and model_name in model_filename:
                print(f'Запускаю GUI с моделью "{model_filename}"')
                is_started = True
                create_gui(keras.models.load_model(model_filename))
        if not is_started:
            print(f"Отсутствует {model_filename}")
        
    elif 'test' == sys.argv[1]:
        def analyze_folder(folder_path, threshold=70):
            results = []
            for filename in os.listdir(folder_path):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    file_path = os.path.join(folder_path, filename)
                    img = Image.open(file_path).convert("RGB")
                    img_tensor = transform(img).unsqueeze(0).to(device)
                    
                    with torch.no_grad():
                        output = model.predict(img_tensor).item()
                    
                    probability = output * 100
                    if probability > threshold:
                        results.append((file_path, probability))

            return results

        def display_results(results):
            num_images = len(results)
            cols = 3
            rows = (num_images + cols - 1) // cols

            fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
            fig.suptitle(f"Изображения с вероятностью > 70%", fontsize=16)

            for i, (file_path, probability) in enumerate(results):
                img = Image.open(file_path)
                ax = axes[i//cols, i%cols] if rows > 1 else axes[i%cols]
                ax.imshow(img)
                ax.set_title(f"{os.path.basename(file_path)}\nВероятность: {probability:.2f}%")
                ax.axis('off')

            # Скрыть пустые подграфики
            for i in range(num_images, rows*cols):
                ax = axes[i//cols, i%cols] if rows > 1 else axes[i%cols]
                ax.axis('off')

            plt.tight_layout()
            plt.show()

        if len(sys.argv) < 3:
            print("Укажите модель для GUI")
            sys.exit(1)

        model_name = sys.argv[2]
        # train_loader, val_loader, test_loader = get_data_loaders(model_name)
        # print("Загружены данные")
        for model_filename in os.listdir("."):
            if 'best' in model_filename and model_name in model_filename:
                print(f'Запускаю GUI с моделью "{model_filename}"')
                break
        model = keras.models.load_model(model_filename)
        # accuracy = test_model(model, test_loader, f"Test_Model_Function_{model_name}.png")
        # print(f"Точность на тестовой выборке: {accuracy:.4f}")

        # Использование функций
        folder_path = os.path.join("hand_validataion/nekogirls")
        results = analyze_folder(folder_path)
        display_results(results)
