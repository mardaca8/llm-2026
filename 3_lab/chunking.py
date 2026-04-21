

def chunk_text(text,chunk_size):
    chunks = []
    for i in range(0,len(text),chunk_size):
        chunk = text[i:i+chunk_size]
        chunks.append(chunk)
        
    return chunks


text = "xdddddd"

chunks = chunk_text(text,2)

print(chunks)