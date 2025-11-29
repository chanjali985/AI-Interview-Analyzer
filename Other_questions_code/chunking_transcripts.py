from transformers import AutoTokenizer

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def chunk_transcript(text, chunk_size=400, overlap=80):
    # Convert text to token IDs
    tokens = tokenizer.encode(text, add_special_tokens=False)

    chunks = []
    start = 0

    # Slide window across tokens
    while start < len(tokens):
        end = start + chunk_size
        chunk_tokens = tokens[start:end]

        # Decode token chunk back to text
        chunk_text = tokenizer.decode(chunk_tokens)
        chunks.append(chunk_text)

        # Move forward by chunk_size - overlap
        start += (chunk_size - overlap)

    return chunks


# Example
text =   (
    "This is a sample long transcript used for testing the chunking logic. " * 200
)
chunks = chunk_transcript(text)

for i, c in enumerate(chunks, 1):
    print(f"\n--- Chunk {i} ---\n{c}\n")
