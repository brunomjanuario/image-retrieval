import os
import csv
import torch
import clip
from PIL import Image

# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Path to your image folder
image_folder = "images"  # ← put all your images here
output_csv = "image_embeddings.csv"

# Prepare list to hold image vectors
image_embeddings = []

# Loop through images in the folder
for filename in os.listdir(image_folder):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        filepath = os.path.join(image_folder, filename)
        try:
            image = preprocess(Image.open(filepath)).unsqueeze(0).to(device)
            with torch.no_grad():
                embedding = model.encode_image(image).squeeze().cpu().tolist()
            image_embeddings.append([filename] + embedding)
            print(f"✅ Processed: {filename}")
        except Exception as e:
            print(f"❌ Error with {filename}: {e}")

# Save to CSV
header = ["image_id"] + [f"dim_{i}" for i in range(512)]
with open(output_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(image_embeddings)

print(f"\n🎉 Done! Embeddings saved to: {output_csv}")