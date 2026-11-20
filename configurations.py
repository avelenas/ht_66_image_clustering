from pathlib import Path

# Основные папки проекта
BASE_DIR = Path(__file__).resolve().parent

IMAGE_DIR = BASE_DIR / "images"
MODEL_DIR = BASE_DIR / "model"
OUTPUT_DIR = BASE_DIR / "output"

# Файлы модели
MODEL_PATH = MODEL_DIR / "best_model.keras"
CLASS_NAMES_PATH = MODEL_DIR / "class_names.json"

# Настройки изображений
IMAGE_SIZE = (250, 250)
BATCH_SIZE = 16

# Настройки DBSCAN
EPS = 0.4
MIN_SAMPLES = 2




# print("Project:", BASE_DIR)
# print("Images:", IMAGE_DIR)
# print("Model exists:", MODEL_PATH.exists())
# print("Class names exists:", CLASS_NAMES_PATH.exists())
