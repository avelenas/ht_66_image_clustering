import shutil
from pathlib import Path

class ImageSaver:
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)

    def save_image(self, image_path, class_name, cluster_label):
        if cluster_label == -1:
            folder_name = "noise"
        else:
            folder_name = f"cluster_{cluster_label}"
        # output_folder = self.output_dir / class_name / folder_name
        if class_name == "cats":
            class_name = "cat"
        elif class_name == "dogs":
            class_name = "dog"
        output_folder = self.output_dir / class_name / folder_name
        output_folder.mkdir(parents=True, exist_ok=True)
        destination = output_folder / Path(image_path).name
        shutil.copy2(image_path, destination)


