import numpy as np
from configurations import IMAGE_DIR
from image.image_load import ImageLoad
from image.image_preparation import ImagePreparation
from classification.image_classifier import ImageClassifier

image_loader = ImageLoad(IMAGE_DIR)
image_preparation = ImagePreparation()
classifier = ImageClassifier()
images = image_loader.load_images()
correct = 0
total = 0
batch_size = 32
# Проверка изображений партиями
for start in range(0, len(images), batch_size):
    batch_paths = images[start:start + batch_size]
    batch_images = []
    for image_path in batch_paths:
        image = image_preparation.prepare_image(image_path)
        batch_images.append(image)
    batch_images = np.array(batch_images)
    predictions = classifier.model.predict(
        batch_images,
        verbose=0)

    for image_path, prediction in zip(batch_paths, predictions):
        probability = float(prediction[0])
        class_index = 1 if probability >= 0.5 else 0
        predicted_class = classifier.class_names[class_index]
        true_class = image_path.parent.name
        if predicted_class == true_class:
            correct += 1
        total += 1
    print(f"Processed: {total} / {len(images)}", flush=True)
accuracy = correct / total * 100

print("Total images:", total)
print("Correct predictions:", correct)
print("Accuracy:", round(accuracy, 2), "%")






# from configurations import IMAGE_DIR
# from image.image_load import ImageLoad
# from image.image_preparation import ImagePreparation
# from classification.image_classifier import ImageClassifier
#
# image_loader = ImageLoad(IMAGE_DIR)
# image_preparation = ImagePreparation()
# classifier = ImageClassifier()
# # images = image_loader.load_images()[:20]
# images = [path for path in image_loader.load_images() if path.parent.name == "dogs"][:20]
# correct = 0
# total = 0
# for image_path in images:
#     image = image_preparation.prepare_image(image_path)
#     predicted_class, probability = classifier.predict(image)
#     true_class = image_path.parent.name
#     if predicted_class == true_class:
#         correct += 1
#     total += 1
#     print("Processing:", image_path.name, flush=True)
# accuracy = correct / total * 100
#
# print("Total images:", total)
# print("Correct predictions:", correct)
# print("Accuracy:", round(accuracy, 2), "%")
