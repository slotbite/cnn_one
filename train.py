# Entrenando una cnn con 2 clases y ~1000 ejemplos por cada una
# en la carpeta data estan separados los datos para entrenamiento de los de prueba

import sys
import os
# se utiliza keras dentro de tensorflow
# preprocesa la imagenes que le entregaremos al algoritmo
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras import optimizers
# permite crear redes neuronales secuenciales
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dropout, Flatten, Dense, Activation
# capas para las convoluciones y poolling
from tensorflow.python.keras.layers import Convolution2D, MaxPooling2D
# ayuda a cerrar una session de keras si se esta ejecutando en 2 plano
from tensorflow.python.keras import backend as K

K.clear_session()

# directorios de datos
data_entrenamiento = './data/entrenamiento'
data_validacion = './data/validacion'

# colocar la direccion completa si tira error del directorio
#data_entrenamiento = 'C:/Users/cvargasa/Documents/IMG/BIG DATA/CNN/data/entrenamiento'
#data_validacion = 'C:/Users/cvargasa/Documents/IMG/BIG DATA/CNN/data/validacion'

"""
Parameters
"""
# numero de veces a iterar sobre todo el entrenamiento
# coorrer con 1 para asegurar que genera guarda el  modelo
# default 20
epocas = 1
# tamaño de la imagenes estos tamaños deberan ser respetados en el predict.py
longitud, altura = 150, 150
# numero de imagenes a procesar en cada uno de los pasos
batch_size = 32
# numero de veces que se procesa la info en una epocas
pasos = 1000
# al final de cada epoca se corren 300 n con el set de validacion para aprender
validation_steps = 300
# numero de filtros a aplicar por cada convolucion
filtrosConv1 = 32
filtrosConv2 = 64
# altura 3 long 3
tamano_filtro1 = (3, 3)
tamano_filtro2 = (2, 2)
# tamaño del fitro en el max polling
tamano_pool = (2, 2)

# numero de categorias  , claese , (gato , perro , gorillla)
clases = 2
# learning rate ,menor es mejor
lr = 0.0004

# Preparamos nuestras imagenes
# preprocesamiento de las imagenes
# reescalar imagenes : tranforma el largo de un pixel de 0 a 1 para optimizar
# shear rnage : genera imagenes inclinadas para cubrir angulos
# zoom_range : aplica rangos de zoom
# horizontla_flip ; inviente las imagenes para aprender direcionalidad
# TODO  agragar mas filtros

entrenamiento_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

# para los datos de validacion solo se reescalara,
# debido que para la validacion las imagenes deben ser tal cual son
test_datagen = ImageDataGenerator(rescale=1. / 255)

# dentro del directorio abrira todas las carpetas y
# procesara  a la determinada altura de manera categorica
entrenamiento_generador = entrenamiento_datagen.flow_from_directory(
    data_entrenamiento,
    target_size=(altura, longitud),
    batch_size=batch_size,
    class_mode='categorical')

validacion_generador = test_datagen.flow_from_directory(
    data_validacion,
    target_size=(altura, longitud),
    batch_size=batch_size,
    class_mode='categorical')

# nos dice como asigno las clases dentro del vector
print(entrenamiento_generador.class_indices)

# crear red convolucional
# red de tipo secuencial , varias capias apiladas
cnn = Sequential()

# primara capa
# aplicando convolucion , filtros , formas , fx relu , input shape solo para la primera capa ,3 es de los canales
cnn.add(Convolution2D(filtrosConv1, tamano_filtro1, padding="same",
                      input_shape=(longitud, altura, 3), activation='relu'))
cnn.add(MaxPooling2D(pool_size=tamano_pool))

# segunda capa
cnn.add(Convolution2D(filtrosConv2, tamano_filtro2, padding="same"))
cnn.add(MaxPooling2D(pool_size=tamano_pool))


# clasificacion
# esa imagen profunda  se convierte a una imagen plana
cnn.add(Flatten())
# despues de aplanar , se manda a una capa normal con 256 neuronas
cnn.add(Dense(256, activation='relu'))
# a la capa denza se le apagaran el 50 % de las neuronas
# para que aprenda caminos alternos
cnn.add(Dropout(0.5))
# ultima capa con 3 neuronas , donde se realiza la clasificacion
cnn.add(Dense(clases, activation='softmax'))

# parametros para optimizar  durante el entrenamiento
# funcion de perdida categorica
# optimizador : 0.004
# metrict : % de que tan bien esta aprendiendo la red neuronal
cnn.compile(loss='categorical_crossentropy',
            optimizer=optimizers.Adam(lr=lr),
            metrics=['accuracy'])

# las imagenes de entrenamiento se alimentaran
# con las de entrenamiento
cnn.fit_generator(
    entrenamiento_generador,
    steps_per_epoch=pasos,
    epochs=epocas,
    validation_data=validacion_generador,
    validation_steps=validation_steps)

# guardar el modelo en un archivo para no estar entrenandolo cada vez
# sada
target_dir = './modelo/'
if not os.path.exists(target_dir):
    os.mkdir(target_dir)
  # estructura del modelo
cnn.save('./modelo/modelo.h5')
# pesos de cada
cnn.save_weights('./modelo/pesos.h5')
