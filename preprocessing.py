import os
import cv2
import numpy as np

input_folder = "Image"
output_folder = "Preprocessed"
size = (224, 224)

# create output folder
os.makedirs(output_folder, exist_ok=True)

# check input folder
if not os.path.exists(input_folder):
    print("Input folder not found")
    exit()

# get class folders
classes = os.listdir(input_folder)

for class_name in classes:
    input_path = os.path.join(input_folder, class_name)
    output_path = os.path.join(output_folder, class_name)

    # check if it is folder
    if not os.path.isdir(input_path):
        continue

    # create output class folder
    os.makedirs(output_path, exist_ok=True)

    count = 0

    for file_name in os.listdir(input_path):
        file_path = os.path.join(input_path, file_name)

        # read image
        img = cv2.imread(file_path)

        # skip broken image
        if img is None:
            print("Skipped image:", file_name)
            continue

        # resize image
        img = cv2.resize(img, size)

        # denoise image
        img = cv2.GaussianBlur(img, (5, 5), 0)

        # normalize image
        img = img / 255.0

        # convert again for saving
        img = (img * 255).astype(np.uint8)

        # save image
        save_path = os.path.join(output_path, file_name)
        cv2.imwrite(save_path, img)

        count = count + 1

    print(class_name, "done:", count, "images processed")

print("All preprocessing completed successfully")
