import os
import fitz  # PyMuPDF
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img

# Load the trained model
model = tf.keras.models.load_model('convolutional_model2.h5') #current best model convolutional_model2.h5

def convert_pdf_to_images(pdf_path, output_folder):
    doc = fitz.open(pdf_path)
    image_paths = []
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        pix = page.get_pixmap()
        image_path = os.path.join(output_folder, f'{os.path.basename(pdf_path).replace(".pdf", "")}_{page_num}.jpg')
        pix.save(image_path)
        image_paths.append(image_path)
    return image_paths

def preprocess_image(image_path):
    image = load_img(image_path, target_size=(128, 128))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0  # Normalize the image
    print(image.size)
    return image

def classify_image(image_path, model):
    # Preprocess the image
    image = preprocess_image(image_path)
    print(image.shape)
    
    # Predict the category
    prediction = model.predict(image)
    category = np.argmax(prediction, axis=1)[0]
    confidence = prediction[0][category] * 100  # Get the confidence percentage
    
    # Map category index to label
    category_labels = {0: 'OR', 1: 'invoice', 2: 'payment'}
    category_label = category_labels[category]
    
    return category_label, confidence

def classify_pdfs_in_folder(pdf_folder, model, output_folder):
    for pdf_file in os.listdir(pdf_folder):
        if pdf_file.endswith('.pdf'):
            pdf_path = os.path.join(pdf_folder, pdf_file)
            image_paths = convert_pdf_to_images(pdf_path, output_folder)
            
            for image_path in image_paths:
                category_label, confidence = classify_image(image_path, model)
                print(f'PDF: {pdf_file}, Image: {image_path} --> Type: {category_label}, Confidence: {confidence:.2f}%')
                
# Example usage
pdf_folder = r'./data_for_test\pdf'
output_folder = r'./data_for_test\converted_img'
classify_pdfs_in_folder(pdf_folder, model, output_folder)
