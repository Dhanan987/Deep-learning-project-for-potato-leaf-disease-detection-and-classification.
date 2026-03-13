import os
import shutil
import random

# PATHS
source_dir = "preprocessed"
target_dir = "Dataset_partion"

classes = [
    "Potato___Early_blight",
    "Potato___healthy",
    "Potato___Late_blight"
]
split_ratio = {
    "train": 0.70,
    "val": 0.20,
    "test": 0.10
}
# CREATE FOLDERS
for split in split_ratio:
    for cls in classes:
        os.makedirs(os.path.join(target_dir, split, cls), exist_ok=True)


# SPLIT DATA
random.seed(42)
for cls in classes:
    cls_path = os.path.join(source_dir, cls)
    images = os.listdir(cls_path)
    random.shuffle(images)

    total = len(images)
    train_end = int(total * split_ratio["train"])
    val_end = train_end + int(total * split_ratio["val"])

    train_images = images[:train_end]
    val_images = images[train_end:val_end]
    test_images = images[val_end:]

    for img in train_images:
        shutil.copy(
            os.path.join(cls_path, img),
            os.path.join(target_dir, "train", cls, img)
        )

    for img in val_images:
        shutil.copy(
            os.path.join(cls_path, img),
            os.path.join(target_dir, "val", cls, img)
        )
    for img in test_images:
        shutil.copy(
            os.path.join(cls_path, img),
            os.path.join(target_dir, "test", cls, img)
        )
    print(f"{cls}: Total={total}, 
            Train={len(train_images)}, 
            Val={len(val_images)}, 
            Test={len(test_images)}")

print("\n Dataset split completed successfully!")
