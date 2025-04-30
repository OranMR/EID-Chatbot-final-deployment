import os
import faiss
import numpy as np

def find_index_files(folder, exts=(".index", ".faiss")):
    """Return sorted list of files in `folder` ending with any of the given exts."""
    return sorted(
        os.path.join(folder, fn)
        for fn in os.listdir(folder)
        if fn.lower().endswith(exts)
    )

def load_indices(paths):
    """Read all indices from disk."""
    indices = []
    for p in paths:
        print(f"Loading index from {p!r} …")
        idx = faiss.read_index(p)
        indices.append(idx)
    return indices

def merge_indices_flat(indices):
    """
    Given a list of indices (all same dimension), reconstruct their vectors
    and add them into one big IndexFlatL2.
    """
    if not indices:
        raise ValueError("No indices to merge")
    d = indices[0].d
    # create a new flat L2 index
    combined = faiss.IndexFlatL2(d)
    total = 0
    for idx in indices:
        if idx.d != d:
            raise ValueError("Dimension mismatch: "
                             f"{idx.d} vs expected {d}")
        n = idx.ntotal
        print(f" → extracting {n} vectors from index (d={d})")
        try:
            # batch‐reconstruct if available
            vecs = idx.reconstruct_n(0, n)
        except AttributeError:
            # fallback to one‐by‐one
            vecs = np.vstack([idx.reconstruct(i) for i in range(n)])
        combined.add(vecs)
        total += n
    print(f"Total vectors merged: {total}")
    return combined

def main():
    # Manually set the input folder and output file path
    input_folder = r""
    
    # Create output directory if it doesn't exist
    output_dir = r""  
    os.makedirs(output_dir, exist_ok=True)
    
    output_index = os.path.join(output_dir, "merged_index.faiss")
    
    files = find_index_files(input_folder)
    if not files:
        print("No FAISS index files found in", input_folder)
    else:
        indices = load_indices(files)
        merged = merge_indices_flat(indices)
        faiss.write_index(merged, output_index)
        print(f"Merged index written to {output_index!r}")

if __name__ == "__main__":
    main()