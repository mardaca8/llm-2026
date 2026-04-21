def chunk_text(text,chunk_size):
    chunks = []
    for i in range(0,len(text),chunk_size):
        chunk = text[i:i+chunk_size]
        chunks.append(chunk)
        
    return chunks

text = open("3_lab/HW3_2/text.txt").read()

chunks = chunk_text(text, 300)

import numpy as np
import pandas as pd
import re
from collections import Counter

def simple_vectorizer(chunks):
    # tokenize and build vocabulary
    all_words = re.findall(r'\w+', " ".join(chunks).lower())
    vocab = sorted(list(set(all_words)))
    word_to_idx = {word: i for i, word in enumerate(vocab)}
    
    # transform chunks into vectors
    vectors = []
    for chunk in chunks:
        vec = np.zeros(len(vocab))
        counts = Counter(re.findall(r'\w+', chunk.lower()))
        for word, count in counts.items():
            if word in word_to_idx:
                vec[word_to_idx[word]] = count
        vectors.append(vec)
        
    return vectors, vocab


vectors, vocabulary = simple_vectorizer(chunks)

df = pd.DataFrame({
    "text_chunk": chunks,
    "vector": vectors
})

print(df.head())

from numpy.linalg import norm

def transform(chunks, vocab):
    word_to_idx = {word: i for i, word in enumerate(vocab)}
    vectors = []
    for chunk in chunks:
        vec = np.zeros(len(vocab))
        counts = Counter(re.findall(r'\w+', chunk.lower()))
        for word, count in counts.items():
            if word in word_to_idx:
                vec[word_to_idx[word]] = count
        vectors.append(vec)
    return vectors

for i in range(3):
    add_text = open(f"3_lab/HW3_2/add_text_{i}.txt").read()
    add_chunks = chunk_text(add_text, 300)
    
    add_vectors = transform(add_chunks, vocabulary)
    
    print(f"\n--- Results for add_text_{i}.txt ---")
    
    for j, a_vec in enumerate(add_vectors):
        best_dot_sim = -1
        best_dot_chunk = ""
        best_cos_sim = -1
        best_cos_chunk = ""
        
        for k, o_vec in enumerate(df["vector"]):
            # dot similarity
            dot_sim = np.dot(a_vec, o_vec)
            if dot_sim > best_dot_sim:
                best_dot_sim = dot_sim
                best_dot_chunk = df["text_chunk"].iloc[k]
                
            # cosine similarity
            denom = (norm(a_vec) * norm(o_vec))
            cos_sim = dot_sim / denom if denom != 0 else 0
            if cos_sim > best_cos_sim:
                best_cos_sim = cos_sim
                best_cos_chunk = df["text_chunk"].iloc[k]
        
        print(f"\nAdd Chunk {j}: {add_chunks[j][:50]}...")
        print(f"Most similar (Dot): {best_dot_chunk[:50]}... (Score: {best_dot_sim})")
        print(f"Most similar (Cosine): {best_cos_chunk[:50]}... (Score: {best_cos_sim:.4f})")
