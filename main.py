# Проект: Распознавание объектов с озвучиванием результатов
# Описание: Система распознает объекты на изображении и озвучивает результаты

# ============================================
# УСТАНОВКА ЗАВИСИМОСТЕЙ
# ============================================

# Устанавливаем необходимые библиотеки
!pip install -q ultralytics
!pip install -q gtts
!pip install -q Pillow

# ============================================
# ИМПОРТ БИБЛИОТЕК
# ============================================

import torch
from PIL import Image
from ultralytics import YOLO
from gtts import gTTS
import os
from IPython.display import Audio, display, HTML
import matplotlib.pyplot as plt
import numpy as np
import cv2
import matplotlib.image as mpimg
from PIL import Image
import requests
from io import BytesIO


# ============================================
# ЗАГРУЗКА МОДЕЛЕЙ
# ============================================

print("=" * 50)
print("ЗАГРУЗКА МОДЕЛЕЙ")
print("=" * 50)

# Модель 1: YOLOv8 для детекции объектов (более новая и стабильная)
print("\n[1/2] Загрузка YOLOv8...")
model_detection = YOLO('yolov8n.pt')  # YOLOv8 nano - быстрая и эффективная
print("✓ YOLOv8 загружена успешно!")

# Модель 2: gTTS для синтеза речи
print("[2/2] Инициализация gTTS...")
print("✓ gTTS готова к работе!")

print("\n" + "=" * 50)
print("ВСЕ МОДЕЛИ ЗАГРУЖЕНЫ")
print("=" * 50 + "\n")

# ============================================
# СЛОВАРЬ ДЛЯ ПЕРЕВОДА НАЗВАНИЙ ОБЪЕКТОВ
# ============================================

# Словарь для перевода названий классов с английского на русский
translation_dict = {
    'person': 'человек',
    'bicycle': 'велосипед',
    'car': 'машина',
    'motorcycle': 'мотоцикл',
    'airplane': 'самолет',
    'bus': 'автобус',
    'train': 'поезд',
    'truck': 'грузовик',
    'boat': 'лодка',
    'traffic light': 'светофор',
    'fire hydrant': 'пожарный гидрант',
    'stop sign': 'знак стоп',
    'parking meter': 'парковочный счетчик',
    'bench': 'скамейка',
    'bird': 'птица',
    'cat': 'кот',
    'dog': 'собака',
    'horse': 'лошадь',
    'sheep': 'овца',
    'cow': 'корова',
    'elephant': 'слон',
    'bear': 'медведь',
    'zebra': 'зебра',
    'giraffe': 'жираф',
    'backpack': 'рюкзак',
    'umbrella': 'зонт',
    'handbag': 'сумка',
    'tie': 'галстук',
    'suitcase': 'чемодан',
    'frisbee': 'фрисби',
    'skis': 'лыжи',
    'snowboard': 'сноуборд',
    'sports ball': 'мяч',
    'kite': 'воздушный змей',
    'baseball bat': 'бейсбольная бита',
    'baseball glove': 'бейсбольная перчатка',
    'skateboard': 'скейтборд',
    'surfboard': 'доска для серфинга',
    'tennis racket': 'теннисная ракетка',
    'bottle': 'бутылка',
    'wine glass': 'бокал',
    'cup': 'чашка',
    'fork': 'вилка',
    'knife': 'нож',
    'spoon': 'ложка',
    'bowl': 'миска',
    'banana': 'банан',
    'apple': 'яблоко',
    'sandwich': 'сэндвич',
    'orange': 'апельсин',
    'broccoli': 'брокколи',
    'carrot': 'морковь',
    'hot dog': 'хот-дог',
    'pizza': 'пицца',
    'donut': 'пончик',
    'cake': 'торт',
    'chair': 'стул',
    'couch': 'диван',
    'potted plant': 'растение в горшке',
    'bed': 'кровать',
    'dining table': 'обеденный стол',
    'toilet': 'унитаз',
    'tv': 'телевизор',
    'laptop': 'ноутбук',
    'mouse': 'мышь',
    'remote': 'пульт',
    'keyboard': 'клавиатура',
    'cell phone': 'телефон',
    'microwave': 'микроволновка',
    'oven': 'духовка',
    'toaster': 'тостер',
    'sink': 'раковина',
    'refrigerator': 'холодильник',
    'book': 'книга',
    'clock': 'часы',
    'vase': 'ваза',
    'scissors': 'ножницы',
    'teddy bear': 'плюшевый мишка',
    'hair drier': 'фен',
    'toothbrush': 'зубная щетка'
}

# ============================================
# ОСНОВНАЯ ФУНКЦИЯ ОБРАБОТКИ
# ============================================

def detect_and_speak(image_path, language='ru', confidence=0.4):
    """
    Основная функция для детекции объектов и их озвучивания
    
    Параметры:
    - image_path: путь к изображению или URL
    - language: язык озвучивания ('ru' - русский, 'en' - английский)
    - confidence: порог уверенности (0-1)
    """
    
    print("\n" + "=" * 50)
    print("НАЧАЛО ОБРАБОТКИ")
    print("=" * 50)
    
    # Шаг 1: Загрузка изображения
    print(f"\n[Шаг 1/4] Загрузка изображения: {image_path}")
    try:
        img = Image.open(image_path)
        print(f"✓ Изображение загружено. Размер: {img.size}")
    except Exception as e:
        print(f"✗ Ошибка загрузки изображения: {e}")
        return
    
    # Шаг 2: Детекция объектов
    print("\n[Шаг 2/4] Распознавание объектов с помощью YOLOv8...")
    results = model_detection.predict(image_path, conf=confidence, verbose=False)
    
    # Получаем результаты
    detections = results[0]
    boxes = detections.boxes
    
    print(f"✓ Найдено объектов: {len(boxes)}")
    
    if len(boxes) == 0:
        print("⚠ Объекты не обнаружены на изображении")
        text = "На изображении не обнаружено объектов"
        
        # Все равно показываем изображение
        plt.figure(figsize=(12, 8))
        plt.imshow(img)
        plt.axis('off')
        plt.title('Изображение (объекты не найдены)', fontsize=16, pad=20)
        plt.tight_layout()
        plt.show()
        
        # Озвучиваем
        tts = gTTS(text=text, lang=language, slow=False)
        audio_file = 'output_speech.mp3'
        tts.save(audio_file)
        display(Audio(audio_file, autoplay=True))
        return
    
    # Выводим таблицу результатов
    print("\nОбнаруженные объекты:")
    print("-" * 50)
    
    object_names = []
    for idx, box in enumerate(boxes):
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        name = detections.names[cls_id]
        object_names.append(name)
        print(f"  {idx+1}. {name} (уверенность: {conf:.2f})")
    print("-" * 50)
    
    # Шаг 3: Формирование текста для озвучивания
    print("\n[Шаг 3/4] Формирование текста...")
    
    # Подсчитываем объекты
    from collections import Counter
    object_counts = Counter(object_names)
    
    # Переводим на русский и формируем список
    objects_ru = []
    for obj, count in object_counts.items():
        obj_ru = translation_dict.get(obj, obj)
        if count > 1:
            objects_ru.append(f"{count} {obj_ru}")
        else:
            objects_ru.append(obj_ru)
    
    # Формируем финальный текст
    if len(objects_ru) == 1:
        text = f"На этом изображении {objects_ru[0]}"
    elif len(objects_ru) == 2:
        text = f"На этом изображении {objects_ru[0]} и {objects_ru[1]}"
    else:
        text = f"На этом изображении " + ", ".join(objects_ru[:-1]) + f" и {objects_ru[-1]}"
    
    print(f"✓ Текст сформирован: '{text}'")
    
    # Шаг 4: Синтез речи
    print("\n[Шаг 4/4] Синтез речи с помощью gTTS...")
    try:
        tts = gTTS(text=text, lang=language, slow=False)
        audio_file = 'output_speech.mp3'
        tts.save(audio_file)
        print(f"✓ Аудио сохранено в файл: {audio_file}")
    except Exception as e:
        print(f"✗ Ошибка синтеза речи: {e}")
        return
    
    # Визуализация результатов
    print("\n" + "=" * 50)
    print("РЕЗУЛЬТАТЫ")
    print("=" * 50)
    
    # Показываем изображение с детекциями
    annotated_img = results[0].plot()
    
    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title('Распознанные объекты', fontsize=16, pad=20)
    plt.tight_layout()
    plt.show()
    
    # Воспроизводим аудио
    print("\n🔊 Воспроизведение аудио:")
    print(f"Текст: '{text}'")
    display(Audio(audio_file, autoplay=True))
    
    print("\n" + "=" * 50)
    print("ОБРАБОТКА ЗАВЕРШЕНА")
    print("=" * 50 + "\n")
    
    return text, audio_file

def load_image(image_path_or_url):
    """Загрузка изображения"""
    try:
        if image_path_or_url.startswith('http'):
            r = requests.get(image_path_or_url, timeout=10)
            img = Image.open(BytesIO(r.content)).convert('RGB')
        else:
            img = Image.open(image_path_or_url).convert('RGB')
        return img
    except Exception as e:
        print(f"Ошибка загрузки изображения: {e}")
        return None

# ============================================
# ПРИМЕРЫ ИСПОЛЬЗОВАНИЯ
# ============================================

print("\n" + "=" * 70)
print("ПРИМЕРЫ ИСПОЛЬЗОВАНИЯ")
print("=" * 70)

print("""
# Пример 1: Загрузка изображения из интернета
detect_and_speak('https://images.unsplash.com/photo-1514888286974-6c03e2ca1dba')

# Пример 2: Загрузка локального файла (сначала загрузите файл в Colab)
from google.colab import files
uploaded = files.upload()
image_name = list(uploaded.keys())[0]
detect_and_speak(image_name)

# Пример 3: Изменение порога уверенности
detect_and_speak('image.jpg', confidence=0.5)  # Более строгий порог
""")

# ============================================
# ЗАПУСК ДЕМОНСТРАЦИИ
# ============================================

print("\n🚀 Запускаем демонстрацию с тестовым изображением...")

# Скачиваем тестовое изображение
import urllib.request
test_url = 'https://ultralytics.com/images/bus.jpg'
test_image = 'demo_image.jpg'

try:
    urllib.request.urlretrieve(test_url, test_image)
    # response = requests.get('https://ultralytics.com/images/bus.jpg')
    # img = Image.open(BytesIO(response.content))
    image = load_image(test_url)
    display(image.resize((400, 400)))
    print(f"✓ Тестовое изображение загружено\n")
    
    # Запускаем обработку
    detect_and_speak(test_image)
    
except Exception as e:
    print(f"✗ Ошибка: {e}")
    print("\nВы можете загрузить свое изображение:")
    print("from google.colab import files")
    print("uploaded = files.upload()")
    print("detect_and_speak(list(uploaded.keys())[0])")

# ============================================
# ДОПОЛНИТЕЛЬНЫЕ ФУНКЦИИ
# ============================================

def batch_process(image_paths, confidence=0.4):
    """
    Обработка нескольких изображений
    """
    results_list = []
    for path in image_paths:
        print(f"\n{'='*70}")
        print(f"Обработка: {path}")
        print('='*70)
        result = detect_and_speak(path, confidence=confidence)
        if result:
            results_list.append((path, result))
    return results_list

def get_statistics(image_path, confidence=0.4):
    """
    Получение статистики по обнаруженным объектам без озвучивания
    """
    results = model_detection.predict(image_path, conf=confidence, verbose=False)
    boxes = results[0].boxes
    
    object_names = []
    confidences = []
    
    for box in boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        name = results[0].names[cls_id]
        object_names.append(name)
        confidences.append(conf)
    
    from collections import Counter
    object_counts = Counter(object_names)
    
    stats = {
        'total_objects': len(boxes),
        'unique_classes': len(object_counts),
        'object_counts': dict(object_counts),
        'avg_confidence': sum(confidences) / len(confidences) if confidences else 0,
        'min_confidence': min(confidences) if confidences else 0,
        'max_confidence': max(confidences) if confidences else 0
    }
    
    return stats

def interactive_demo():
    """
    Интерактивная демонстрация - загрузка своего изображения
    """
    print("📤 Загрузите свое изображение:")
    from google.colab import files
    uploaded = files.upload()
    
    if uploaded:
        image_name = list(uploaded.keys())[0]
        print(f"\n✓ Файл загружен: {image_name}")
        detect_and_speak(image_name)
    else:
        print("❌ Файл не загружен")

print("\n" + "=" * 70)
print("✓ Код готов к использованию!")
print("=" * 70)
print("\n💡 Попробуйте:")
print("   interactive_demo()  # Загрузить свое изображение")
print("   get_statistics('demo_image.jpg')  # Получить статистику")
