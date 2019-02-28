from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.python.keras.applications import VGG16
from tensorflow.python.keras.optimizers import Adam


class Neural:
    def launch_neural(self):
        vgg16_net = VGG16(weights='imagenet', include_top=False, input_shape=(161, 161, 3))
        vgg16_net.trainable = False

        model = Sequential()
        model.add(vgg16_net)
        # Добавляем в модель новый классификатор
        model.add(Flatten())
        model.add(Dense(256))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(2))
        model.add(Activation('softmax'))

        model.compile(loss='categorical_crossentropy',
                      optimizer=Adam(lr=1e-5),
                      metrics=['accuracy'])
        model.load_weights("mnist_model.h5")
        return model
