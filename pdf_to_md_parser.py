import os
import time
from pathlib import Path
from docling.document_converter import DocumentConverter
from config import Config

def batch_convert_pdfs(input_dir, output_dir):
    """
    Converts all PDF files in input_dir to Markdown files in output_dir using Docling.

    Args:
        input_dir (str or Path): Path to the folder containing raw PDF files.
        output_dir (str or Path): Path to the folder where .md files will be saved.
    """

    # Ensure inputs are Path objects
    input_path_obj = Path(input_dir)
    output_path_obj = Path(output_dir)

    # 1. Create Output Directory (if it doesn't exist)
    output_path_obj.mkdir(parents=True, exist_ok=True)

    # 2. Check for PDF files
    pdf_files = list(input_path_obj.glob("*.pdf"))

    if not pdf_files:
        print(f"[WARN] No PDF files found in {input_path_obj}")
        return

    print(f"[INFO] Found {len(pdf_files)} PDF files to process.")
    print(f"[INFO] Reading from: {input_path_obj.absolute()}")
    print(f"[INFO] Saving to:   {output_path_obj.absolute()}")

    # 3. Initialize Docling Converter
    print("[INFO] Loading Docling model...")
    # This might take a moment to download models if running for the first time
    converter = DocumentConverter()

    # 4. Process Loop
    start_total = time.time()

    for i, pdf_path in enumerate(pdf_files):
        try:
            print(f"--- Processing {i+1}/{len(pdf_files)}: {pdf_path.name} ---")

            # A. Convert the document
            result = converter.convert(pdf_path)

            # B. Export to Markdown text
            md_content = result.document.export_to_markdown()

            # C. Determine Output Filename
            output_filename = f"{pdf_path.stem}.md"
            output_file_path = output_path_obj / output_filename

            # D. Save to File
            with open(output_file_path, "w", encoding="utf-8") as f:
                f.write(md_content)

            print(f"    ✅ Saved to: {output_file_path.name}")

        except Exception as e:
            print(f"    ❌ Failed to convert {pdf_path.name}")
            print(f"       Error: {e}")

    duration = time.time() - start_total
    print(f"\n[DONE] Processed {len(pdf_files)} files in {duration:.2f} seconds.")

def main():
    # Use paths from Config
    input_dir = Config.RAW_DATA_PATH
    output_dir = Config.PROCESSED_DATA_PATH

    # Check if input directory exists
    if not os.path.exists(input_dir):
        print(f"[ERROR] Input directory {input_dir} does not exist.")
        # Attempt to create the knowledge-base/raw directory if it's missing
        # but warn the user.
        print(f"[INFO] Please place your PDF files in {os.path.abspath(input_dir)}")
        return

    batch_convert_pdfs(input_dir, output_dir)

if __name__ == "__main__":
    main()
