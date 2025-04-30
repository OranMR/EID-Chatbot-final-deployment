import os

def find_text_files(folder, ext=".txt"):
    """Return sorted list of files in `folder` ending with given extension."""
    return sorted(
        os.path.join(folder, fn)
        for fn in os.listdir(folder)
        if fn.lower().endswith(ext)
    )

def combine_text_files(input_files, output_file):
    """Combine multiple text files into one."""
    total_lines = 0
    
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for file_path in input_files:
            print(f"Processing {file_path!r} ...")
            try:
                with open(file_path, 'r', encoding='utf-8') as infile:
                    content = infile.read()
                    outfile.write(content)
                    
                    # Count lines
                    lines = content.count('\n') + 1
                    total_lines += lines
                    print(f" → Added {lines} lines from file")
                    
                    # Add a separator between files
                    outfile.write("\n\n")
                    
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
    
    print(f"Total lines combined: {total_lines}")
    return total_lines

def main():
    # Manually set the input folder and output file path
    input_folder = r""
    
    # Create output directory if it doesn't exist
    output_dir = r""
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, "combined_text.txt")
    
    files = find_text_files(input_folder)
    if not files:
        print("No text files found in", input_folder)
    else:
        line_count = combine_text_files(files, output_file)
        print(f"Combined text file written to {output_file!r} with {line_count} lines")

if __name__ == "__main__":
    main()