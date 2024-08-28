# Convolutional Model (CNN) in Python
## เกี่ยวกับ Repos นี้
Repos นี้เป็นส่วนหนึ่งของโปรเจค ai คัดแยกประเภทเอกสารและ ai ตรวจจับด้านบัตรประชาชน โดยในโปรเจคนี้จะใช้ python (3.11.9) tensorflow และ keras เวอร์ชั่น 2 ในการสร้างโมเดล และสามารถรันได้ด้วยทั้ง python script และผ่าน IDE

ในการเทรนโมเดลของทั้งสองโปรเจคข้างต้นจะใช้โครงสร้างโมเดลแบบเดียวกัน และผลลัพธ์โมเดลที่เทรนแล้วจะออกมาในรูปแบบไฟล์ .h5  แต่จะแตกต่างกันตรง dataset ที่นำมาใช้เทรน 

เมื่อเทรนโมเดลสำเร็จแล้ว สามารถนำไฟล์ .h5 ซึ่งข้างในจะเป็นข้อมูลทั้งโครงสร้างและ parameters ต่างๆมาปรับใช้ได้ตามต้องการ (การปรับใช้กับ Java จะอยู่ใน Repos DocumentClassifier_java และ idcardRotation)

## Dataset Preparation
อันดับแรกก่อนการสร้างและเทรนโมเดลคือการเตรียม dataset โดยโฟลเดอร์สำหรับ dataset จะต้องมีสองโฟลเดอร์
- data_for_train
- data_for_test
  
### pdf to img Program
ในเมื่อการใช้กับโมเดลนั้นจะต้องใช้รูปภาพ .jpg แต่ไฟล์เอกสารที่มีอยู่ อยู่ในรูปแบบ .pdf จึงต้องเริ่มจากการแปลงไฟล์ .pdf ให้เป็น .jpg 
โดยในโปรแกรมนี้เริ่มจาก
- การเซ็ต path ใน input_folder ให้เป็น path ของโฟลเดอร์ที่เก็บไฟล์ .pdf ของเอกสารต่างๆที่ต้องการจะนำมาเทรน และ
- เซ็ต path ใน output_folder ให้เป็น path ของโฟลเดอร์ data_for_train ซึ่งคือโฟลเดอร์ที่เก็บไฟล์ .jpg ของเอกสารต่างๆที่ต้องการจะนำมาเทรน
  
```
# Example usage
input_folder = r'./dataset\for_train'
output_folder = r'./data_for_train' 

process_pdfs_in_folder(input_folder, output_folder)

```
  
โปรแกรมจะทำการ loop ไปใน input_folder แปลงจาก .pdf เป็น .jpg และไปเก็บไว้ที่ output_folder ซึ่งคือโฟลเดอร์ data_for_train 


### data_for_train
ในโฟลเดอร์นี้เป็นพื้นที่เก็บไฟล์ .jpg ของเอกสารต่างๆ หลังจากโปรแกรม pdf_to_img.py ได้รันและนำรูปที่แปลงแล้วมาเก็บไว้ในนี้ จากนั้นให้สร้างโฟลเดอร์ย่อยอีกตามจำนวนประเภทของเอกสารที่มี
เช่นในตัวอย่างมีเอกสารทั้งหมดสามประเภท invoice OR และ payment จึงเป็นโฟลเดอร์ data_for_train ที่มีโฟลเดอร์ย่อยอีกสามโฟลเดอร์
- data_for_train
    - invoice
        - รูปภาพเอกสาร invoice.jpg ...
        - รูปภาพเอกสาร invoice.jpg ...
        - รูปภาพเอกสาร invoice.jpg ...
    - OR
        - รูปภาพเอกสาร OR.jpg...
        - รูปภาพเอกสาร OR.jpg...
        - รูปภาพเอกสาร OR.jpg...
    - payment
        - รูปภาพเอกสาร payment.jpg...
        - รูปภาพเอกสาร payment.jpg...
        - รูปภาพเอกสาร payment.jpg...


### data_for_test
ในโฟลเดอร์นี้เป็นพื้นที่เก็บไฟล์ .pdf และ .jpg ของเอกสารต่างๆที่ต้องการจะนำมาจำแนกประเภท โฟลเดอร์นี้ **ไม่จำเป็น** ต้องแยกประเภทเหมือนกับ data_for_train 

**ไฟล์เอกสาร .pdf ที่ต้องการจะจำแนกประเภทให้นำมาใส่ในโฟลเดอร์นี้ data_for_test\pdf**

โดยโฟลเดอร์ data_for_test นี้จะประกอบไปด้วยสองโฟลเดอร์ย่อย converted_img และ pdf
- data_for_test
    - converted_img
    - pdf
        - รูปภาพเอกสาร payment.pdf...
        - รูปภาพเอกสาร OR.pdf...
        - รูปภาพเอกสาร OR.pdf...
        - รูปภาพเอกสาร invoice.pdf ...
        - รูปภาพเอกสาร payment.pdf...
        - รูปภาพเอกสาร invoice.pdf ...

โฟลเดอร์ pdf มีไว้เพื่อเก็บเอกสาร .pdf ต่างๆที่ต้องการจะนำมาจำแนกประเภท

โฟลเดอร์ converted_img มีไว้เพื่อเก็บรูปที่แปลงจาก .pdf เป็น .jpg โดยโปรแกรม Model Usage Program ก่อนจะนำเข้าโมเดลเพื่อจำแนกประเภท 


## CNN Training Program
โปรแกรมนี้เป็นส่วนของการสร้างและเทรนโมเดล โดยจะแยกเป็น 5 ส่วนหลักๆ 

ก่อนที่จะใช้โปรแกรมนี้ ควรจะ pip install tensorflow (keras) เวอร์ชั่น 2, numpy, matplotlib, และ pillow 

### Importing Dataset
ในส่วนนี้จะเป็นการเตรียมและแบ่ง dataset เป็นสองตัว train และ validate โดยจะแบ่ง 80% ของภาพใน data_for_train ไว้สำหรับ train และ 20% ของภาพใน data_for_train ไว้สำหรับ validate เพื่อเช็คอีกรอบในแต่ละ epochs
```
#Dataset----------------------------------------------------------------------------------------------------------
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
train_generator = train_datagen.flow_from_directory(
    r'./data_for_train',  # Replace with the path to your dataset
    target_size=(128, 128),  # Resize images to 128x128 pixels
    batch_size=32,
    class_mode='categorical',
    subset='training'
)
validation_generator = train_datagen.flow_from_directory(
    r'./data_for_train',  # Replace with the path to your dataset
    target_size=(128, 128),  # Resize images to 128x128 pixels
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)
```
### Define CNN Model
ในส่วนนี้จะเป็นการสร้างโครงสร้างของตัวโมเดล โดยจะมีสองชั้นหลักๆคือ ชั้นของ convolution และชั้น dense โดยส่วนที่สำคัญที่สุดคือ input shape และ ชั้น dense ชั้นสุดท้าย (softmax layer) 

input shape ในชั้นแรกสุดกับ array รูปภาพที่เราจะนำมาเข้าโมเดลจะต้องเหมือนกัน ([1,128,128,3]) สามารถเช็คได้ด้วย model.summary() และเลขจำนวน node ใน dense ชั้นสุดท้ายจะต้องเท่ากับจำนวนประเภทของผลลัพธ์ที่มี ซึ่งในที่นี้เป็น 3 เพราะมีเอกสารอยู่ 3 ประเภท

```
#Define CNN Model-------------------------------------------------------------------------------------------------
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dense(3, activation='softmax')
])

model.summary()
```

### Compile the Model 
ในส่วนนี้จะเป็นตัวกำหนดว่าจะใช้ optimizer ตัวไหน คิด loss ด้วย function แบบไหน และมี metrics เป็นอะไร โดยในที่นี้เลือกใช้เป็น optimizer Nadam เนื่องด้วย tools deeplearning4j ในภาษา Java ใช้ loss function เป็นแบบ categorical crossentropy และมี metrics เป็น accuracy

```
# Compile the model-----------------------------------------------------------------------------------------------
model.compile(optimizer='Nadam', loss='categorical_crossentropy', metrics=['accuracy']) 
```


### Train the Model
ในส่วนนี้จะเป็นขั้นตอนการเทรนโมเดล โดยมี parameters เป็น dataset train และ validate และจำนวน epochs (จำนวนครั้งที่โมเดลจะได้เห็น dataset train) 

ในแต่ละรอบของ epoch โมเดลจะเห็น dataset train แล้วเช็คกับ labels เพื่อคิด loss function และ เช็คอีกทีกับ dataset validate ที่ไม่เคยเห็นเมื่อตอนเทรนเพื่อเช็ค overfitting โดยค่าที่ออกมา accuracy และ val_accuracy ไม่ควรจะห่างกันจนเห็นได้ชัด 

```
# Train the model-------------------------------------------------------------------------------------------------
model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    epochs=20 #Epochs is adjustable
)
```

### Save the Model
ส่วนสุดท้ายคือการเชฟโครงสร้างและ parameters ต่างๆของโมเดลในไฟล์ .h5 เพื่อนำไปปรับใช้ต่อ

```
# Save the model-------------------------------------------------------------------------------------------------
model.save('convolutional_model2.h5')
print("Model saved")
```



## Model Usage Program in Python
โปรแกรมนี้เป็นการนำโมเดลที่เทรนแล้วในไฟล์ .h5 มาใช้ โดยโปรแกรม python นี้เป็นต้นแบบของ ทั้ง DocumentClassifier_java และ idcardRotation ที่นำมาแปลและปรับเป็น Java คู่กับ deeplearning4j ในการ import โมเดลไปใช้

เริ่มจากการ import โมเดล .h5 
```
# Load the trained model
model = tf.keras.models.load_model('convolutional_model2.h5') #current best model convolutional_model2.h5
```

ต่อมาคือการเซ็ต path ของ pdf_folder ให้เป็น data_for_test\pdf และ output_folder ให้เป็น data_for_test\converted_img 

```
# Example usage
pdf_folder = r'./data_for_test\pdf'
output_folder = r'./data_for_test\converted_img'
classify_pdfs_in_folder(pdf_folder, model, output_folder)
```

### classify_pdfs_in_folder(pdf_folder, model, output_folder)
ฟังก์ชั่นนี้ มี parameters :
- pdf_folder : path ของ data_for_test\pdf
- model : model ที่ import เข้ามา
- output_folder : path ของ data_for_test\converted_img

จะนำไฟล์ .pdf แต่ละอันในโฟลเดอร์ data_for_test\pdf มาแปลงให้เป็นรูปภาพก่อน แล้วนำไปเก็บไว้ใน output_folder หรือ data_for_test\converted_img จากนั้น แต่ละรูปจะถูกนำไปเข้าโมเดลเพื่อจำแนกประเภท โดยผลลัพธ์จะออกมาในรูปแบบ text ที่บอก ชื่อไฟล์ pdf, path ของรูปที่แปลง, ประเภทที่โมเดลได้จำแนก และ ค่าความน่าจะเป็น 

```
def classify_pdfs_in_folder(pdf_folder, model, output_folder):
    for pdf_file in os.listdir(pdf_folder):
        if pdf_file.endswith('.pdf'):
            pdf_path = os.path.join(pdf_folder, pdf_file)
            image_paths = convert_pdf_to_images(pdf_path, output_folder)
            
            for image_path in image_paths:
                category_label, confidence = classify_image(image_path, model)
                print(f'PDF: {pdf_file}, Image: {image_path} --> Type: {category_label}, Confidence: {confidence:.2f}%'
```

### convert_pdf_to_images(pdf_path, output_folder)

Parameters :
- pdf_path : path ของไฟล์ .pdf เดี่ยวๆ
- output_folder : path ของ data_for_test\converted_img

ฟังก์ชั่นนี้จะรับ parameters ต่อมาจาก classify_pdfs_in_folder() และทำการแปลง .pdf เป็น .jpg แล้วนำไปเก็บไว้ที่ data_for_test\converted_img

```
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
```


### classify_image(image_path, model) และ  preprocess_image(image_path)
classify_image() parameters :
- image_path : path ของ รูป .jpg เดี่ยวๆ
- model : model ที่ import เข้ามา

ฟังก์ชั่นนี้คือฟังก์ชั่นหลักที่รับ parameters ต่อมาจาก convert_pdf_to_images() และ classify_pdfs_in_folder() แล้วนำแต่ละภาพมาผ่าน preprocess_image() เพื่อปรับเป็น array และเตรียมพร้อมนำเข้าสู่โมเดล

เมื่อนำเข้าโมเดลแล้ว ผลลัพธ์จะถูกนำไปคำนวณค่าความน่าจะเป็น และจำแนกเป็นประเภทไหนตาม category_labels

**หากมีการเพิ่มเติมประเภทของผลลัพธ์ category_labels ก็ควรจะเปลี่ยนตามด้วยเช่นกัน โดยสามารถยึดหลักได้ตามโฟลเดอร์ย่อยใน data_for_train** ซึ่งในตอนนี้มี 3 ตัวนั้นคือ invoice, OR, และ payment 


```
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
```

## โปรเจคบัตรประชาชน
โปรเจคตรวจจับและหมุนบัตรประชาชนให้ถูกต้องก็ได้ต้นแบบมาจากโปรเจค โดยที่แตกต่างคือ 
- ไม่ต้องทำการแปลง pdf เป็น jpg
- dataset ที่นำมาทั้ง train และ test
- จำนวนประเภทที่จะจำแนก จาก 3 ผลลัพธ์เป็น 4ผลลัพธ์ คือ ไม่ต้องหมุน หมุนซ้าย 90องศา หมุนขวา 90องศา และสุดท้าย หมุน 180องศา ซึ่งต้องเพิ่มโฟลเดอร์ย่อยใน data_for_train, เปลี่ยน parameter ใน dense ชั้นสุดท้าย, และเพิ่มคู่ label กับ index ใน category_labels 











