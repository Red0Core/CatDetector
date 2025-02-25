import os
import random

def balance_dataset(dataset_path="dataset/train"):
    """
    Удаляет лишние изображения из папки, чтобы сбалансировать количество `cats` и `not_cats`
    """
    cats_dir = os.path.join(dataset_path, "cats")
    not_cats_dir = os.path.join(dataset_path, "not_cats")

    # Получаем списки файлов
    cat_files = [os.path.join(cats_dir, f) for f in os.listdir(cats_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    not_cat_files = [os.path.join(not_cats_dir, f) for f in os.listdir(not_cats_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    # Количество изображений в каждой категории
    cat_count = len(cat_files)
    not_cat_count = len(not_cat_files)

    print(f"🐱 Cats: {cat_count}, 🚫 Not Cats: {not_cat_count}, 📁 Path: {dataset_path}")

    # Определяем, где больше фото
    if cat_count > not_cat_count:
        extra_files = random.sample(cat_files, cat_count - not_cat_count)
        print(f"📉 Удаляем {len(extra_files)} фото из `cats/`")
    elif not_cat_count > cat_count:
        extra_files = random.sample(not_cat_files, not_cat_count - cat_count)
        print(f"📉 Удаляем {len(extra_files)} фото из `not_cats/`")
    else:
        print("✅ Датасет уже сбалансирован!")
        return

    # Удаляем лишние фото
    for file in extra_files:
        os.remove(file)

    print("✅ Датасет успешно сбалансирован!")

# Запуск балансировки для тренировочного датасета
balance_dataset("dataset/train")

# Можно также сбалансировать валидационный и тестовый набор
balance_dataset("dataset/validation")
balance_dataset("dataset/test")
