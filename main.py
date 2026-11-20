from configurations import IMAGE_DIR, OUTPUT_DIR
from image.image_load import ImageLoad
from image.image_preparation import ImagePreparation
from classification.image_classifier import ImageClassifier
from classification.image_saver import ImageSaver
from clustering.image_cluster import ImageCluster

def main():
    image_loader = ImageLoad(IMAGE_DIR)
    image_preparation = ImagePreparation()
    classifier = ImageClassifier()
    image_saver = ImageSaver(OUTPUT_DIR)
    image_cluster = ImageCluster()
    #
    images = image_loader.load_images()

    # Списки для хранения
    class_images = {
        "cats": [],
        "dogs": []
    }
    class_features = {
        "cats": [],
        "dogs": []
    }
    # Классификация изображений
    for image_path in images:
        image = image_preparation.prepare_image(image_path)
        predicted_class, probability = classifier.predict(image)
        features = classifier.extract_features(image)
        class_images[predicted_class].append(image_path)
        class_features[predicted_class].append(features)
        print(f"{image_path.name}: {predicted_class} ({probability:.2f})")

    # Кластери
    for class_name in class_images:
        if not class_features[class_name]:
            continue
        labels = image_cluster.cluster_images(
            class_features[class_name]
        )
        print(class_name, "Clusters:", set(labels))

        for image_path, label in zip(class_images[class_name], labels):
            image_saver.save_image(
                image_path,
                class_name,
                label)
    print("Processing completed!")

if __name__ == "__main__":
    main()


