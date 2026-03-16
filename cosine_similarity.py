# cosine_similarity.py
import numpy as np
import voyageai
import os
from dotenv import load_dotenv

load_dotenv() # Load environment variables from .env file)

vo = voyageai.Client(api_key =os.getenv("pa-iV0XlqCoiJob_RhJWed3ZbkzGOKhLOAIbYm304E2L7W")) # uses VOYAGE_API_KEY env var


def get_embedding(text):
    text = text.replace("\n", " ")
    return np.array(vo.embed([text], model="voyage-3.5", input_type="document").embeddings[0])

def cosine_similarity(text1, text2):
    vec1 = get_embedding(text1).flatten()
    vec2 = get_embedding(text2).flatten()
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))