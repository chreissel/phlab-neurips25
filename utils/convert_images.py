import torch
import torch.nn as nn
from torchvision.transforms import v2
from PIL import Image
import os
from tqdm import tqdm
import sys

def convert_image(file_path):
    with open(file_path,"rb") as f:
        im = Image.open(f).convert("RGB")
    im_tens = v2.PILToTensor()(im)
    return im_tens

def convert_imagenette(source,target):
    subdirs = os.listdir(source)
    for subdir in subdirs:
        print(f"processing {subdir}")
        files = [f for f in os.listdir(f"{source}/{subdir}") if f.split(".")[-1].lower() in ["jpeg", "jpg", "png"]]
        os.makedirs(f"{target}/{subdir}", exist_ok=True)
        for file in tqdm(files):
            file_path = f"{source}/{subdir}/{file}"
            file_name = ".".join(file.split(".")[:-1])  # Get the file name without extension
            try:
                im_tens = convert_image(file_path)
                save_path = f"{target}/{subdir}/{file_name}.pt"
                torch.save(im_tens, save_path)
            except Exception as e:
                print(f"Error processing {file_path}: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python convert_images.py <source_directory> <target_directory>")
        sys.exit(1)

    source_directory = sys.argv[1]
    target_directory = sys.argv[2]

    convert_imagenette(source_directory, target_directory)