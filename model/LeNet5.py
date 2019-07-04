from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras import backend as K
from keras.utils import plot_model
import keras
import matplotlib.pyplot as plt

KERNAL_SIZE = (5,5)
INPUT_SHAPE = (28,28,1)
POOL_SIZE = (2,2)
NUM_CLASSES = 10
BATCH_SIZE = 16
EPOCHS = 5

img_rows, img_cols = 28,28

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train[0:5000,:]
y_train = y_train[0:5000]



if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

y_train = keras.utils.to_categorical(y_train, NUM_CLASSES)
y_test = keras.utils.to_categorical(y_test, NUM_CLASSES)
print('labels, categorical: ',y_train.shape, y_test.shape)


model = Sequential()
model.add(Convolution2D(20,kernel_size=(5,5),padding='same',activation='relu',input_shape=INPUT_SHAPE))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(50,kernel_size=KERNAL_SIZE,padding='same',activation='relu'))
model.add(Flatten()) #拉成一维
model.add(Dense(500, activation='relu'))
model.add(Dense(NUM_CLASSES, activation='softmax'))

def precision(y_true, y_pred):
    """Precision metric. Only computes a batch-wise average of precision.
-    Computes the precision, a metric for multi-label classification of
-    how many selected items are relevant.
-    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def recall(y_true, y_pred):
    """Recall metric.
-    Only computes a batch-wise average of recall.
-    Computes the recall, a metric for multi-label classification of
-    how many relevant items are selected.
-    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def fbeta_score(y_true, y_pred, beta=1):

    """Computes the F score.

-    """
    if beta < 0:
        raise ValueError('The lowest choosable beta is zero (only precision).')

    # If there are no true positives, fix the F score at 0 like sklearn.
    if K.sum(K.round(K.clip(y_true, 0, 1))) == 0:

        return 0
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    bb = beta ** 2
    fbeta_score = (1 + bb) * (p * r) / (bb * p + r + K.epsilon())
    return fbeta_score

def fmeasure(y_true, y_pred):
    """Computes the f-measure, the harmonic mean of precision and recall.
    Here it is only computed as a batch-wise average, not globally.
    """
    return fbeta_score(y_true, y_pred, beta=1)

# 该函数将画出模型结构图，并保存成图片：
plot_model(model, to_file='../picture/model.png',show_shapes=True)


model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=[fmeasure,recall,precision])

#  调用回调函数保存模型
# filepath="weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
filepath="../save/weights-improvement-{epoch:02d}.h5"
checkpointer1 = keras.callbacks.ModelCheckpoint(filepath,
                               monitor='loss',
                               verbose=1,
                               save_best_only=False,
                               save_weights_only=True,
                               mode='auto',
                               period=5)

# 调用回调函数终止训练
checkpointer2 = keras.callbacks.EarlyStopping(monitor='loss',
                                              patience=3,#训练不下降之后经过几个阶段停止
                                              verbose=1,
                                              mode='auto')


# 调用回调函数更改学习率

# def scheduler(epoch):
#     if epoch % 100 == 0 and epoch != 0:
#         lr = K.get_value(model.optimizer.lr)
#         K.set_value(model.optimizer.lr, lr * 0.1)
#         print("lr changed to {}".format(lr * 0.1))
#     return K.get_value(model.optimizer.lr)
#
# reduce_lr = keras.callbacks.LearningRateScheduler(scheduler)


# 当评价指标不在提升时，减少学习率
# reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
#                                               factor=0.1,
#                                               patience=10,
#                                               verbose=1,
#                                               mode='auto',
#                                               epsilon=0.0001,
#                                               cooldown=0,
#                                               min_lr=0)



# TensorBoard
tensroboad = keras.callbacks.TensorBoard(log_dir='../logs',
                            histogram_freq=1,
                            write_graph=False,
                            write_images=True,
                            embeddings_freq=0,
                            embeddings_layer_names=None,
                            embeddings_metadata=None)


history = model.fit(x_train,
                    y_train,
                    validation_split=0.1,
                    epochs = EPOCHS,
                    batch_size=BATCH_SIZE,
                    callbacks=[checkpointer1,checkpointer2,tensroboad]
                    )

with open('../save/log_history.txt','w') as f:
    f.write(str(history.history))

history_dict = history.history
history_dict.keys()

epochs = range(1, len(history_dict['loss']) + 1)
plt.plot(epochs, history_dict['loss'], 'b',label='loss')
plt.plot(epochs, history_dict['precision'], 'g',label='precision')
plt.xlabel('Epochs')
plt.grid()
plt.legend(loc=1)
plt.savefig("../picture/show_loss_precision.jpg")
plt.show()

# test

# score = model.evaluate(x_test, y_test, verbose=0)
# print('Test loss:', score[0])
# print('Test accuracy:', score[1])