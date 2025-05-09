import csv
import torch
import clip
import numpy as np

def retrieval_images(prompt):
    # Load CLIP model   
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    # Load embeddings from CSV
    image_embeddings = []
    image_ids = []
    with open("image_embeddings.csv", "r") as f:
        reader = csv.reader(f)
        headers = next(reader)
        for row in reader:
            image_ids.append(row[0])
            embedding = list(map(float, row[1:]))
            image_embeddings.append(embedding)

    image_embeddings = np.array(image_embeddings)

    # Get user prompt
    text_prompt = prompt

    # Encode text
    with torch.no_grad():
        text_tokens = clip.tokenize([text_prompt]).to(device)
        text_embedding = model.encode_text(text_tokens).cpu().numpy()[0]

    # Normalize both embeddings for cosine similarity
    image_embeddings_norm = image_embeddings / np.linalg.norm(image_embeddings, axis=1, keepdims=True)
    text_embedding_norm = text_embedding / np.linalg.norm(text_embedding)

    # Compute cosine similarities
    similarities = image_embeddings_norm @ text_embedding_norm

    # Get top match(es)
    top_k = 9
    top_indices = similarities.argsort()[-top_k:][::-1]

    results = []

    print(f"\nTop {top_k} matches for: \"{text_prompt}\"")
    for idx in top_indices:
        print(f"{image_ids[idx]} (score: {similarities[idx]:.4f})")
        results.append({
            "image_id": image_ids[idx],
            "score": float(similarities[idx])  # make sure it's JSON-serializable
        })
    
    return results
