def chunk_sequence(seq, chunk_size):
    """Split sequence into smaller chunks."""
    return [seq[i:i+chunk_size] for i in range(0, len(seq), chunk_size)]

#text_chunks = chunk_sequence(text_tokens, 512)  # Example chunk size = 512
