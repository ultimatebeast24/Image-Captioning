import os
from PIL import Image
from caption import describe
from embed import embed_and_store

def process_archive(archive_folder):
    for root, dirs, files in os.walk(archive_folder):
        for filename in files:
            if filename.lower().endswith((".png", ".jpg", ".jpeg")):
                path = os.path.join(root, filename)
                img = Image.open(path).convert("RGB")

                caption = describe(img)
                print(caption, path)

                embed_and_store(caption, path)

process_archive("archive")
