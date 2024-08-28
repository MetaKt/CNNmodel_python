import os
import fitz  # PyMuPDF

# Function to convert a single PDF to images
def pdf_to_images(pdf_path, output_folder):
    pdf_document = fitz.open(pdf_path)
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]

    for page_number in range(len(pdf_document)):
        page = pdf_document.load_page(page_number)
        pix = page.get_pixmap()
        image_path = os.path.join(output_folder, f'{pdf_name}_page_{page_number + 1}.jpg')
        pix.save(image_path)
        print(f'Saved: {image_path}')

# Function to process all PDFs in a folder
def process_pdfs_in_folder(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    for file_name in os.listdir(input_folder):
        if file_name.endswith('.pdf'):
            pdf_path = os.path.join(input_folder, file_name)
            pdf_to_images(pdf_path, output_folder)

# Example usage
input_folder = r'./dataset\for_train'
output_folder = r'./data_for_train' 

process_pdfs_in_folder(input_folder, output_folder)
