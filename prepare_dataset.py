import os
import glob
import cv2
import shutil
import random
from tqdm import tqdm

def process_dataset(source_dir, output_dir, split_ratio=0.8):
    images_dir = os.path.join(source_dir, "images")
    labels_dir = os.path.join(source_dir, "labels")

    train_img_dir = os.path.join(output_dir, "images", "train")
    val_img_dir = os.path.join(output_dir, "images", "val")
    train_lbl_dir = os.path.join(output_dir, "labels", "train")
    val_lbl_dir = os.path.join(output_dir, "labels", "val")

    os.makedirs(train_img_dir, exist_ok=True)
    os.makedirs(val_img_dir, exist_ok=True)
    os.makedirs(train_lbl_dir, exist_ok=True)
    os.makedirs(val_lbl_dir, exist_ok=True)

    # find all subdirectories in images
    subdirs = [d for d in os.listdir(images_dir) if os.path.isdir(os.path.join(images_dir, d))]

    all_data = []

    for subdir in subdirs:
        img_subdir_path = os.path.join(images_dir, subdir)
        lbl_subdir_path = os.path.join(labels_dir, subdir)

        if not os.path.exists(lbl_subdir_path):
            continue

        images = [f for f in os.listdir(img_subdir_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        for img_name in images:
            img_path = os.path.join(img_subdir_path, img_name)
            txt_name = os.path.splitext(img_name)[0] + ".txt"
            txt_path = os.path.join(lbl_subdir_path, txt_name)

            if os.path.exists(txt_path):
                # use a unique name in case of overlaps across subdirs
                unique_name = f"{subdir}_{img_name}"
                all_data.append((img_path, txt_path, unique_name))

    random.shuffle(all_data)
    train_count = int(len(all_data) * split_ratio)
    
    for i, (img_path, txt_path, unique_name) in enumerate(tqdm(all_data, desc="Processing dataset")):
        is_train = i < train_count

        out_img_dir = train_img_dir if is_train else val_img_dir
        out_lbl_dir = train_lbl_dir if is_train else val_lbl_dir

        out_img_path = os.path.join(out_img_dir, unique_name)
        
        # Read image to get dimensions
        img = cv2.imread(img_path)
        if img is None:
            continue
        
        h, w, _ = img.shape

        # Read txt
        with open(txt_path, 'r') as f:
            lines = f.read().splitlines()

        if not lines:
            continue

        # The first line is supposedly the number of boxes.
        try:
            num_boxes = int(lines[0].strip())
        except ValueError:
            # If not an int, maybe it doesn't have the explicit count.
            num_boxes = len(lines)
            start_idx = 0
        else:
            start_idx = 1
        
        # Write to new txt path
        txt_out_name = os.path.splitext(unique_name)[0] + ".txt"
        out_lbl_path = os.path.join(out_lbl_dir, txt_out_name)
        
        valid_boxes = False
        with open(out_lbl_path, 'w') as out_f:
            for line in lines[start_idx:]:
                parts = line.strip().split()
                if len(parts) >= 4:
                    try:
                        x1, y1, x2, y2 = map(float, parts[:4])
                        # convert to YOLO
                        x_center = (x1 + x2) / 2.0 / w
                        y_center = (y1 + y2) / 2.0 / h
                        width = (x2 - x1) / w
                        height = (y2 - y1) / h
                        
                        # clamp
                        x_center = max(0, min(1, x_center))
                        y_center = max(0, min(1, y_center))
                        width = max(0, min(1, width))
                        height = max(0, min(1, height))

                        out_f.write(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                        valid_boxes = True
                    except ValueError:
                        continue
        
        if valid_boxes:
            shutil.copy(img_path, out_img_path)
        else:
            os.remove(out_lbl_path)

    # create data.yaml
    yaml_path = os.path.join(output_dir, "dataset.yaml")
    with open(yaml_path, 'w') as f:
        f.write(f"train: {os.path.abspath(train_img_dir).replace(chr(92), '/')}\n")
        f.write(f"val: {os.path.abspath(val_img_dir).replace(chr(92), '/')}\n\n")
        f.write("nc: 1\n")
        f.write("names: ['license_plate']\n")
        
    print(f"Dataset prepared! Total training images: {train_count}, Total val images: {len(all_data) - train_count}")

if __name__ == '__main__':
    source = r"c:\Users\Mohamed\Desktop\projects\one\License_Plates_of_Algeria_Dataset-master\License_Plates_of_Algeria_Dataset-master\Detector"
    output = r"c:\Users\Mohamed\Desktop\projects\one\yolo_dataset"
    process_dataset(source, output)
