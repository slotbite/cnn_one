import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model

# debe ser la misma al train
longitud, altura = 150, 150

# establece los directorios
modelo = './modelo/modelo.h5'
pesos_modelo = './modelo/pesos.h5'

# diremoas que nuestra red se cargara con sus pesos
cnn = load_model(modelo)
cnn.load_weights(pesos_modelo)

# esta funcion recibira el nombre de una imagen y nos dira a la la categoria con cierta precision


def predict(file):
 # carga la imagen
    x = load_img(file, target_size=(longitud, altura))

    # convierte  la imagen a un vector
    x = img_to_array(x)

    # agrega una dimencion en el eje 0 para que sea compateble con los datos entrenados
    x = np.expand_dims(x, axis=0)

    # llamamos la red y para hacer la preciccion
    array = cnn.predict(x)  # [[1,0,0]]

    # como es de 2 dimenciones solo queremos la dim en donde esta la prediccion
    result = array[0]  # [1,0,0]

    # la respuesta es igual al indice donde esta el valor mayor
    answer = np.argmax(result)
 # {'gato': 0, 'gorila': 1, 'perro': 2}
# OJO ACA REVISAR QUE COINCIDA CON LA SALIDA DEL TRAIN.PY
# print(entrenamiento_generador.class_indices)
    if answer == 0:
        print("CNN prediction: gato")
    elif answer == 1:
        print("CNN prediction: perro")
    # elif answer == 2:
    #    print("CNN prediction: nada")

    return answer


# SUBIR TUS FOTOSA  LA CARPETA  Y REVISAR
predict("dog1.jpg")
predict("dog2.jpg")
predict("dog3.jpg")
predict("cat.jpg")

predict("dog1.jpg")
predict("dog2.jpg")
predict("dog3.jpg")
predict("cat.jpg")

# voila
