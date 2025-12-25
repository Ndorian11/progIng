# app.py
import streamlit as st
from ultralytics import YOLO
from gtts import gTTS
from PIL import Image, ImageDraw, ImageFont
import requests
from io import BytesIO
import tempfile
import os
from collections import Counter

# ----------------------------
# Конфигурация страницы
# ----------------------------
st.set_page_config(
    page_title="👁️ Распознавание объектов с озвучкой",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ----------------------------
# Словарь перевода
# ----------------------------
translation_dict = {
    'person': 'человек',
    'bicycle': 'велосипед',
    'car': 'машина',
    'motorcycle': 'мотоцикл',
    'airplane': 'самолёт',
    'bus': 'автобус',
    'train': 'поезд',
    'truck': 'грузовик',
    'boat': 'лодка',
    'traffic light': 'светофор',
    'fire hydrant': 'пожарный гидрант',
    'stop sign': 'знак стоп',
    'parking meter': 'парковочный счётчик',
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
    'toothbrush': 'зубная щётка'
}

# ----------------------------
# Загрузка модели (кешируется)
# ----------------------------
@st.cache_resource
def load_yolo_model():
    return YOLO('yolov8n.pt')

model = load_yolo_model()

# ----------------------------
# Функции
# ----------------------------
def format_text_from_objects(objects: list, lang: str = 'ru') -> str:
    if not objects:
        return "На изображении не обнаружено объектов."
    
    counts = Counter(objects)
    items = []
    for obj, cnt in counts.items():
        obj_ru = translation_dict.get(obj, obj) if lang == 'ru' else obj
        items.append(f"{cnt} {obj_ru}" if cnt > 1 else obj_ru)
    
    if len(items) == 1:
        return f"На этом изображении {items[0]}."
    elif len(items) == 2:
        return f"На этом изображении {items[0]} и {items[1]}."
    else:
        return f"На этом изображении " + ", ".join(items[:-1]) + f" и {items[-1]}."

def generate_speech(text: str, lang: str = 'ru') -> str:
    tts = gTTS(text=text, lang=lang, slow=False)
    audio_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3").name
    tts.save(audio_file)
    return audio_file

# ----------------------------
# Интерфейс
# ----------------------------
st.title("👁️ Распознавание объектов с озвучкой")
st.markdown("Загрузите изображение — система найдёт объекты и озвучит результат на русском языке.")

# Выбор способа загрузки
option = st.radio("Выберите способ загрузки:", ("По URL", "Загрузить файл"), horizontal=True)

image = None

if option == "По URL":
    url = st.text_input("Введите URL изображения", placeholder="https://example.com/image.jpg")
    if url:
        try:
            response = requests.get(url, timeout=10)
            image = Image.open(BytesIO(response.content)).convert("RGB")
        except Exception as e:
            st.error(f"Ошибка загрузки: {e}")
elif option == "Загрузить файл":
    uploaded_file = st.file_uploader("Выберите изображение", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")

# Порог уверенности
confidence = st.slider("Порог уверенности", 0.1, 1.0, 0.4, 0.05)

# Кнопка анализа
if image and st.button("🔍 Анализировать", use_container_width=True):
    with st.spinner("Распознавание объектов..."):
        # Сохраняем изображение во временный файл для YOLO
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            image.save(tmp.name)
            temp_path = tmp.name
        
        # Детекция
        results = model.predict(temp_path, conf=confidence, verbose=False)
        boxes = results[0].boxes
        
        # Извлекаем названия
        detected_names = []
        for box in boxes:
            cls_id = int(box.cls[0])
            name = results[0].names[cls_id]
            detected_names.append(name)
        
        # Удаляем временный файл
        os.unlink(temp_path)
        
        # Отображаем исходное изображение
        st.image(image, caption="Исходное изображение", use_column_width=True)
        
        # === УДАЛЕНО: results[0].plot() — потенциальный источник ошибки libGL.so.1 ===
        # Вместо этого — просто выводим список объектов
        
        # Формируем текст и озвучиваем
        speech_text = format_text_from_objects(detected_names, lang='ru')
        st.subheader("🎙️ Результат:")
        st.write(speech_text)
        
        # Генерация и воспроизведение аудио
        with st.spinner("Генерация речи..."):
            audio_file = generate_speech(speech_text, lang='ru')
            st.audio(audio_file, format="audio/mp3")
        
        # Статистика
        st.subheader("📊 Статистика:")
        st.write(f"Всего объектов: {len(detected_names)}")
        if detected_names:
            counts = Counter(detected_names)
            st.write("Обнаружены:")
            for obj, cnt in counts.items():
                st.write(f"- {translation_dict.get(obj, obj)}: {cnt}")

elif not image and st.button("🔍 Анализировать", use_container_width=True):
    st.warning("Пожалуйста, загрузите изображение.")

# ----------------------------
# Подвал
# ----------------------------
st.markdown("---")
st.caption("Используется YOLOv8 + gTTS • Все вычисления выполняются на сервере")
