##This is the convolutional model using tf.keras with optimizer adam           keras 2.14.0
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

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

# Compile the model-----------------------------------------------------------------------------------------------
model.compile(optimizer='Nadam', loss='categorical_crossentropy', metrics=['accuracy']) #optimizer='Nadam'



# Train the model-------------------------------------------------------------------------------------------------
model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    epochs=5 #Epochs is adjustable
)

# # Save the model-------------------------------------------------------------------------------------------------
# model.save('convolutional_model2.h5')
# print("Model saved")
