import numpy as np
from tensorflow.python.keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input

from MyAudioLib import Audio
from NeuralLib import Neural

model = Neural().launch_neural()

while True:
    s = input('Нажмите s для записи и q  для выхода\n')
    if (s == 's'):
        record = Audio()
        record.record_audio_in_file()
        record.creation_spectrogram()
        img = image.load_img('grad/grad-1.png', target_size=(161, 161))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x) / 255
        preds = model.predict(x)
        classes = ['град быстрый', 'град медленный']
        print(classes[np.argmax(preds)])
    elif (s == 'q'):
        break
    else:
        continue
