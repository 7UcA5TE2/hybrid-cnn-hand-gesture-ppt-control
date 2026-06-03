import tensorflow as tf
from tensorflow.keras import layers, models, regularizers

def build_model(input_shape=(224, 224, 3), num_classes=7):

    model = models.Sequential()

    # Normalization inside model
    model.add(layers.Rescaling(1./255, input_shape=input_shape))

    # Block 1
    model.add(layers.Conv2D(32, (3,3), padding='same',
                            kernel_regularizer=regularizers.l2(0.0001)))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(layers.MaxPooling2D((2,2)))

    # Block 2
    model.add(layers.Conv2D(64, (3,3), padding='same',
                            kernel_regularizer=regularizers.l2(0.0001)))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(layers.MaxPooling2D((2,2)))

    # Block 3
    model.add(layers.Conv2D(128, (3,3), padding='same',
                            kernel_regularizer=regularizers.l2(0.0001)))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(layers.MaxPooling2D((2,2)))

    # Block 4
    model.add(layers.Conv2D(256, (3,3), padding='same',
                            kernel_regularizer=regularizers.l2(0.0001)))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(layers.MaxPooling2D((2,2)))

    model.add(layers.GlobalAveragePooling2D())

    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.5))

    model.add(layers.Dense(num_classes, activation='softmax'))

    return model