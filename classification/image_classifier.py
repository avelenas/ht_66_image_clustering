import json
import numpy as np
import tensorflow as tf
from configurations import MODEL_PATH, CLASS_NAMES_PATH, IMAGE_SIZE

class ImageClassifier:
    def __init__(self):
        self.model = tf.keras.models.load_model(MODEL_PATH)
        with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as file:
            self.class_names = json.load(file)
        ###
        dummy_input = tf.zeros((1, *IMAGE_SIZE, 3))
        self.model(dummy_input, training=False)
        # Пошук шару
        # feature_layer = None
        # for layer in self.model.layers:
        #     if isinstance(layer, tf.keras.layers.GlobalAveragePooling2D):
        #         feature_layer = layer
        #         break
        # if feature_layer is None:
        #     raise ValueError("GlobalAveragePooling2D layer not found")
        feature_layer = None

        for layer in self.model.layers:
            if isinstance(layer, tf.keras.layers.MaxPooling2D):
                feature_layer = layer

        if feature_layer is None:
            raise ValueError("MaxPooling2D layer not found")
        # отримуємо ознаки віж моделі
        self.feature_model = tf.keras.Model(
            inputs=self.model.inputs,
            outputs=feature_layer.output)


    def predict(self, image):
        image = np.expand_dims(image, axis=0)
        prediction = self.model.predict(image, verbose=0)
        probability = float(prediction[0][0])
        class_index = 1 if probability >= 0.5 else 0
        predicted_class = self.class_names[class_index]
        return predicted_class, probability

    def extract_features(self, image):
        image = np.expand_dims(image, axis=0)
        features = self.feature_model.predict(image, verbose=0)
        features = features.reshape(features.shape[0], -1)
        return features[0]






