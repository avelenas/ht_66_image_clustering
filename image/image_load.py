from pathlib import Path
from configurations import IMAGE_DIR

class ImageLoad:
    def __init__(self, image_dir):
        self.image_dir = Path(image_dir)
        self.extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    def load_images(self):
        image_paths = [path for path in self.image_dir.rglob("*")
                       if path.is_file() and path.suffix.lower() in self.extensions]
        if not image_paths:
            raise ValueError("No images found")
        return image_paths


# image_loader = ImageLoad(IMAGE_DIR)
# images = image_loader.load_images()
# print("Images found:", len(images))
# for image in images[:5]:
#     print(image)

