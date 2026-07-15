import os
import random
import shutil

# Ganti sesuai nama folder hasil extract
SOURCE_DIR = "flowers"

# Folder output
DEST_DIR = "dataset"

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

random.seed(42)

classes = [d for d in os.listdir(SOURCE_DIR)
           if os.path.isdir(os.path.join(SOURCE_DIR, d))]

for cls in classes:

    images = os.listdir(os.path.join(SOURCE_DIR, cls))
    random.shuffle(images)

    total = len(images)

    train_end = int(total * TRAIN_RATIO)
    val_end = train_end + int(total * VAL_RATIO)

    train_imgs = images[:train_end]
    val_imgs = images[train_end:val_end]
    test_imgs = images[val_end:]

    for split in ["train", "validation", "test"]:
        os.makedirs(os.path.join(DEST_DIR, split, cls), exist_ok=True)

    for img in train_imgs:
        shutil.copy(
            os.path.join(SOURCE_DIR, cls, img),
            os.path.join(DEST_DIR, "train", cls, img)
        )

    for img in val_imgs:
        shutil.copy(
            os.path.join(SOURCE_DIR, cls, img),
            os.path.join(DEST_DIR, "validation", cls, img)
        )

    for img in test_imgs:
        shutil.copy(
            os.path.join(SOURCE_DIR, cls, img),
            os.path.join(DEST_DIR, "test", cls, img)
        )

print("Dataset berhasil dibagi!")