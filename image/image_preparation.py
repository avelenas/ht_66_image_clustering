import numpy as np
from PIL import Image
from image.image_load import ImageLoad
from configurations import IMAGE_DIR
from configurations import IMAGE_SIZE

class ImagePreparation:
    def prepare_image(self, image_path):
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            img = img.resize(IMAGE_SIZE)
            image_array = np.asarray(img, dtype="float32")
        return image_array


# image_loader = ImageLoad(IMAGE_DIR)
# image_preparation = ImagePreparation()
# images = image_loader.load_images()
# image = image_preparation.prepare_image(images[0])
# print("Image shape:", image.shape)
# print("Image type:", image.dtype)
# print("Minimum:", image.min())
# print("Maximum:", image.max())



