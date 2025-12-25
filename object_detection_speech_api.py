# object_detection_speech_api.py

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import torch
from ultralytics import YOLO
from gtts import gTTS
from PIL import Image
import requests
from io import BytesIO
import os
import tempfile
import cv2
import numpy as np
from collections import Counter
import shutil

# ------------------------
# Инициализация моделей
# ------------------------

print("🚀 Инициализация YOLOv8...")
model_detection = YOLO('yolov8n.pt')
print("✅ YOLOv8 готова")
# gTTS не требует предварительной инициализации

# ------------------------
# Словарь перевода
# ------------------------

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

# ------------------------
# Вспомогательные функции
# ------------------------

def load_image_from_url(url: str) -> Image.Image:
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        return Image.open(BytesIO(resp.content)).convert('RGB')
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Ошибка загрузки изображения: {str(e)}")

def load_image_from_bytes(data: bytes) -> Image.Image:
    try:
        return Image.open(BytesIO(data)).convert('RGB')
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Ошибка загрузки изображения: {str(e)}")

def format_text_from_objects(objects: List[str], lang: str = 'ru') -> str:
    if not objects:
        return "На изображении не обнаружено объектов."

    counts = Counter(objects)
    items = []

    for obj, cnt in counts.items():
        if lang == 'ru':
            obj = translation_dict.get(obj, obj)
        if cnt > 1:
            items.append(f"{cnt} {obj}")
        else:
            items.append(obj)

    if len(items) == 1:
        return f"На этом изображении {items[0]}."
    elif len(items) == 2:
        return f"На этом изображении {items[0]} и {items[1]}."
    else:
        return f"На этом изображении " + ", ".join(items[:-1]) + f" и {items[-1]}."

# ------------------------
# FastAPI приложение
# ------------------------

app = FastAPI(
    title="YOLOv8 + gTTS: Распознавание и озвучивание объектов",
    description="API для детекции объектов на изображении и генерации речи на русском/английском языке",
    version="1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Папка для временных аудиофайлов
AUDIO_DIR = "audio_output"
os.makedirs(AUDIO_DIR, exist_ok=True)

class DetectionResponse(BaseModel):
    detected_objects: List[str]
    object_counts: Dict[str, int]
    speech_text: str
    audio_url: str
    total_objects: int
    success: bool

@app.post("/detect_and_speak", response_model=DetectionResponse)
async def detect_and_speak_endpoint(
    image_url: Optional[str] = None,
    file: Optional[UploadFile] = File(None),
    language: str = Query("ru", regex="^(ru|en)$"),
    confidence: float = Query(0.4, ge=0.0, le=1.0)
):
    # Загрузка изображения
    if image_url:
        image_pil = load_image_from_url(image_url)
        # Сохраняем временно для YOLO
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            image_pil.save(tmp.name)
            image_path = tmp.name
    elif file:
        contents = await file.read()
        image_pil = load_image_from_bytes(contents)
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            image_pil.save(tmp.name)
            image_path = tmp.name
    else:
        raise HTTPException(status_code=400, detail="Требуется image_url или файл")

    try:
        # Детекция
        results = model_detection.predict(image_path, conf=confidence, verbose=False)
        boxes = results[0].boxes

        detected_names = []
        for box in boxes:
            cls_id = int(box.cls[0])
            name = results[0].names[cls_id]
            detected_names.append(name)

        # Формируем текст
        speech_text = format_text_from_objects(detected_names, lang=language)

        # Генерация аудио
        tts = gTTS(text=speech_text, lang=language, slow=False)
        audio_filename = f"{next(tempfile._get_candidate_names())}.mp3"
        audio_path = os.path.join(AUDIO_DIR, audio_filename)
        tts.save(audio_path)

        # Подсчёт объектов
        counts = dict(Counter(detected_names))

        # Удаляем временный файл изображения
        if os.path.exists(image_path):
            os.unlink(image_path)

        return DetectionResponse(
            detected_objects=detected_names,
            object_counts=counts,
            speech_text=speech_text,
            audio_url=f"/audio/{audio_filename}",
            total_objects=len(detected_names),
            success=True
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка обработки: {str(e)}")

@app.get("/audio/{filename}")
async def get_audio(filename: str):
    file_path = os.path.join(AUDIO_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Аудиофайл не найден")
    return FileResponse(file_path, media_type="audio/mpeg", filename=filename)

@app.get("/")
def root():
    return {
        "message": "YOLOv8 + gTTS API запущен!",
        "endpoints": {
            "POST /detect_and_speak": "Детекция + озвучка",
            "GET /audio/{filename}": "Скачать аудио"
        }
    }
