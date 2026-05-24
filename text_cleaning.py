import os
import glob
from config.config import Config

# List of noise strings to remove from Markdown files
TEXT_TO_REMOVE = [
    "<!-- image -->",
    "<!-- formula-not-decoded -->",
]

def clean_markdown_files(input_dir, output_dir):
    """
    Reads all .md files from input_dir, removes noise strings, 
    and saves cleaned versions to output_dir.
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Find all Markdown files
    list_file = glob.glob(os.path.join(input_dir, "*.md"))

    if not list_file:
        print(f"[WARN] No .md files found in {input_dir}")
        return

    print(f"[INFO] Found {len(list_file)} files. Starting cleanup...\n")

    for file_path in list_file:
        filename = os.path.basename(file_path)
        new_file_path = os.path.join(output_dir, filename)

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Remove noise strings
            for text in TEXT_TO_REMOVE:
                content = content.replace(text, "")

            # Clean up trailing whitespace and extra newlines
            cleaned_lines = [line.rstrip() for line in content.splitlines()]
            cleaned_content = "\n".join(cleaned_lines).strip()

            with open(new_file_path, "w", encoding="utf-8") as f:
                f.write(cleaned_content)

            print(f"    ✅ Cleaned: {filename}")

        except Exception as e:
            print(f"    ❌ Failed to clean {filename}: {e}")

    print("\n[DONE] All files have been cleaned and saved to:", os.path.abspath(output_dir))

def main():
    # Use paths from Config
    input_dir = Config.PROCESSED_DATA_PATH
    output_dir = Config.DATA_FOLDER_PATH

    if not os.path.exists(input_dir):
        print(f"[ERROR] Input directory {input_dir} does not exist.")
        return

    clean_markdown_files(input_dir, output_dir)

if __name__ == "__main__":
    main()
