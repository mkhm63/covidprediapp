# src/extract_text.py
import os
import fitz  # PyMuPDF

def extract_pdfs_to_txt(input_dir="data/pdfs", output_dir="output/texts"):
    os.makedirs(output_dir, exist_ok=True)
    
    for filename in os.listdir(input_dir):
        if filename.endswith(".pdf"):
            path = os.path.join(input_dir, filename)
            doc = fitz.open(path)
            text = ""
            for page in doc:
                text += page.get_text()

            output_path = os.path.join(output_dir, filename.replace(".pdf", ".txt"))
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(text)
            print(f"✅ Extracted {filename} → {output_path}")