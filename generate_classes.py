import json
from tensorflow.keras.preprocessing.image import ImageDataGenerator

DATASET_PATH = r"dataset/archive (2)/PlantVillage Dataset/color"

datagen = ImageDataGenerator(rescale=1./255)

data = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical",
    shuffle=False
)

classes = {
    str(v): k
    for k, v in data.class_indices.items()
}

with open("classes.json", "w") as f:
    json.dump(classes, f, indent=4)

print("classes.json created successfully!")
