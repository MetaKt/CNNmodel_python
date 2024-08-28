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
ในเมื่อการเทรนนั้นจะต้องใช้รูปภาพ .jpg แต่ไฟล์เอกสารที่มีอยู่ อยู่ในรูปแบบ .pdf จึงต้องเริ่มจากการแปลงไฟล์ .pdf ให้เป็น .jpg 
โดยในโปรแกรมนี้เริ่มจาก
- การเซ็ต path ใน input_folder ให้เป็น path ของโฟลเดอร์ที่เก็บไฟล์ .pdf ของเอกสารต่างๆที่ต้องการจะนำมาเทรน และ
- เซ็ต path ใน output_folder ให้เป็น path ของโฟลเดอร์ data_for_train ซึ่งคือโฟลเดอร์ที่เก็บไฟล์ .jpg ของเอกสารต่างๆที่ต้องการจะนำมาเทรน
  
โปรแกรมจะทำการ loop ไปใน input_folder แปลงจาก .pdf เป็น .jpg และไปเก็บไว้ที่ output_folder ซึ่งคือโฟลเดอร์ data_for_train 


### data_for_train
ในโฟลเดอร์นี้เป็นพื้นที่เก็บไฟล์ .jpg ของเอกสารต่างๆ หลังจากโปรแกรม pdf_to_img.py ได้รันและนำรูปที่แปลงแล้วมาเก็บไว้ในนี้ ให้สร้างโฟลเดอร์ย่อยอีกตามจำนวนประเภทของเอกสารที่มี
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
โฟลเดอร์นี้สามารถทั้งนำมาเช็คโมเดลที่เทรนแล้ว และทั้งนำมาเก็บเอกสาร .pdf ต่างๆที่ต้องการจะจำแนกด้วยโมเดลที่เทรนแล้วจริงๆ 

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

input shape ในชั้นแรกสุดกับ array รูปภาพที่เราจะนำมาเข้าโมเดลจะต้องเหมือนกัน ([1,128,128,3]) สามารถเช็คได้ด้วย model.summary() และเลขจำนวน node ใน dense ชั้นสุดท้ายจะต้องเท่ากับจำนวนประเภทของผลลัพธ์ที่มี ซึ่งในมี่นี้เป็น 3 เพราะมีเอกสารอยู่ 3 ประเภท

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
model.compile(optimizer='Nadam', loss='categorical_crossentropy', metrics=['accuracy']) #optimizer='Nadam'
```


### Train the Model
ในส่วนนี้จะเป็นขั้นตอนการเทรนโมเดล โดยมี parameters เป็น dataset train และ validate และจำนวน epochs (จำนวนครั้งที่โมเดลจะได้เห็น dataset train) 

ในแต่ละรอบของ epoch โมเดลจะเห็น dataset train แล้วเช็คกับ labels เพื่อคิด loss function และ เช็คอีกทีกับ dataset validate ที่ไม่เคยเห็นมาตอนเทรนเพื่อเช็ค overfitting โดยค่าทีออกมา accuracy และ val_accuracy ไม่ควรจะห่างกันจนเห็นได้ชัด 

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
