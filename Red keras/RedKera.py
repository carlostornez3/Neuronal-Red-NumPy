import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import RMSprop, SGD, Adam
from tensorflow.keras import activations
from keras.callbacks import ModelCheckpoint, EarlyStopping
import time
import matplotlib.pyplot as plt

#Para escribir logs en wandb
#pip install wandb
#wandb login

learning_rate = 0.001
epochs = 300
batch_size = 120

#import wandb
#from wandb.keras import WandbCallback
#wandb.init(project="tensor1")
#wandb.config.learning_rate = learning_rate
#wandb.config.epochs = epochs
#wandb.config.batch_size = batch_size
#wandb.config.patito = "cuacCuac"
###################
import mlflow
mlflow.tensorflow.autolog()

dataset=mnist.load_data()



(x_train, y_train), (x_test, y_test) = dataset
data_number=10000
x_train = x_train[:data_number]
y_train = y_train[:data_number]
x_test = x_test[:data_number]
y_test = y_test[:data_number]
print(x_train.shape)
x_trainv = x_train.reshape(data_number, 784)
x_testv = x_test.reshape(data_number, 784)

x_trainv = x_trainv.astype('float32')
x_testv = x_testv.astype('float32')

x_trainv /= 255  # x_trainv = x_trainv/255
x_testv /= 255



num_classes=10
y_trainc = keras.utils.to_categorical(y_train, num_classes)
y_testc = keras.utils.to_categorical(y_test, num_classes)



model = Sequential()
model.add(Dense(60, activation='relu6', input_shape=(784,)))
model.add(Dense(50, activation='relu'))
model.add(Dense(30, activation='tanh'))
model.add(Dense(20, activation='sigmoid'))
#model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))

model.summary()
earlystop = EarlyStopping(monitor='val_loss',mode='min',restore_best_weights=False,patience=20,verbose=1)
start_time = time.time()
model.compile(loss='binary_crossentropy',optimizer=Adam(learning_rate=learning_rate,beta_1=0.9,beta_2=0.99, epsilon=10e-09),metrics=['accuracy'])

history = model.fit(x_trainv, y_trainc,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_testv, y_testc)
                    )

end_time = time.time()
execution_time = end_time - start_time
print(f"El tiempo de ejecución fue de {execution_time} segundos.")
plt.plot(history.history['loss'],label="Loss")
plt.plot(history.history['val_loss'],label="Val_Loss")
plt.legend()
plt.figure()
plt.plot(history.history['accuracy'], label="Accuracy")
plt.plot(history.history['val_accuracy'], label="val_Accuracy")
plt.legend()
plt.show()

score = model.evaluate(x_testv, y_testc, verbose=1)




print(score)
a=model.predict(x_testv)
print(a[1])
print("resultado correcto:")
print(y_testc[1])

#Para guardar el modelo en disco
model.save("red.h5")
exit()
#para cargar la red:
#modelo_cargado = tf.keras.models.load_model('red.h5')