# import numpy as np
from clustering.image_cluster import ImageCluster

# features = np.array([
#     [1.0, 0.0, 0.1],
#     [0.9, 0.1, 0.1],
#     [0.0, 1.0, 0.1],
#     [0.1, 0.9, 0.1],
#     [-1.0, 0.0, 0.0]
# ])
# cluster = ImageCluster()
# labels = cluster.cluster_images(features)
# print("Clusters:", labels)
#

from collections import Counter
from configurations import IMAGE_DIR
from image.image_load import ImageLoad
from image.image_preparation import ImagePreparation
from classification.image_classifier import ImageClassifier

image_loader = ImageLoad(IMAGE_DIR)
image_preparation = ImagePreparation()
classifier = ImageClassifier()
image_cluster = ImageCluster()
# images = image_loader.load_images()
images = image_loader.load_images()
cat_images = [path for path in images if path.parent.name == "cats"][:20]
dog_images = [path for path in images if path.parent.name == "dogs"][:20]
images = cat_images + dog_images
class_features = {"cats": [],"dogs": []}
# Получение признаков изображений
for image_path in images:
    image = image_preparation.prepare_image(image_path)
    predicted_class, probability = classifier.predict(image)
    features = classifier.extract_features(image)
    class_features[predicted_class].append(features)
    print("Processed:", image_path.name, flush=True)
# Проверка разных значений EPS
# for eps in [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
for eps in [0.4]:
    print("\nEPS:", eps)
    image_cluster.model.set_params(eps=eps)
    for class_name, features in class_features.items():
        labels = image_cluster.cluster_images(features)
        counts = Counter(labels)
        clusters = len([label for label in counts if label != -1])
        noise = counts.get(-1, 0)
        cluster_sizes = {
            label: count
            for label, count in counts.items()
            if label != -1}
        print(
            class_name,
            "| Clusters:", clusters,
            "| Sizes:", cluster_sizes,
            "| Noise:", noise)


