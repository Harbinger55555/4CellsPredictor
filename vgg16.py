from helper import plot_graphs
import os
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG16
from keras import models
from keras import layers


base_dir = './datasets'
train_dir = os.path.join(base_dir, 'TRAIN')
test_dir = os.path.join(base_dir, 'TEST')

train_datagen = ImageDataGenerator(rescale=1./255,
                                   validation_split=0.2)
test_datagen = ImageDataGenerator(rescale=1./255)

# Default class_mode is categorical and data is shuffled.
gen_args = {'directory':train_dir,
            'target_size':(150,150),
            'batch_size':20}
train_generator = train_datagen.flow_from_directory(**gen_args,
                                                    subset='training')
validation_generator = train_datagen.flow_from_directory(**gen_args,
                                                         subset='validation')


# Make the model.
model = models.Sequential()

# Using a pretrained VGG16 conv_base.
conv_base = VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=(150, 150, 3)
)

# Freeze the conv_base layers.
conv_base.trainable = False

model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(100, activation='relu'))
model.add(layers.Dense(100, activation='relu'))
model.add(layers.Dense(100, activation='relu'))
model.add(layers.Dense(4, activation='softmax'))

model.compile(
    loss='categorical_crossentropy',
    optimizer='rmsprop',
    metrics=['acc']
)

history = model.fit_generator(
    train_generator,
    steps_per_epoch=100,
    epochs=30,
    validation_data=validation_generator,
    validation_steps=50
)

plot_graphs(history)