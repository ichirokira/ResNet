import cv2
import numpy as np
from tensorflow import keras
from processor import AspectAwearProcessor, MeanProcessor, PatchProcessor, CropProcessor
from callbacks import EpochCheckpoint, TrainMonitor
from resnet import ResNet
from sklearn.preprocessing import LabelBinarizer
import configs as cfg
print("[INFO] Loading Dataset .................")
(trainX, trainY), (testX, testY) = keras.datasets.cifar10.load_data()

trainX = trainX.astype("float32")
testX = testX.astype("float32")

mean = np.mean(trainX, axis=0)
trainX -= mean
testX -= mean

le = LabelBinarizer()
trainY = le.fit_transform(trainY)
testY = le.fit_transform(testY)


aug = keras.preprocessing.image.ImageDataGenerator(width_shift_range=0.2, height_shift_range=0.2, shear_range=1.5, fill_mode="nearest", horizontal_flip=True)
callbacks = [EpochCheckpoint(cfg.CHECKPOINT_OUTPUT, every=5, startAt=cfg.START_EPOCH),TrainMonitor(cfg.FIG_PATH, cfg.JSON_PATH, startAt= cfg.START_EPOCH)]

print("[INFO] Loading and Compiling Model...............")
if cfg.MODEL is None:
    model = ResNet.build(32,32,3,10,(9,9,9),(64,64,128,256))
    opt = keras.optimizers.SGD(lr = 1e-1)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=['accuracy'])
else:
    model = keras.models.load_model(cfg.MODEL)
    print("[INFO] Old learning rate {}".format(keras.backend.get_value(model.optimizer.lr)))
    keras.backend.set_value(model.optimizer.lr, cfg.LEARNING_RATE)
    print("[INFO] New learning rate {}".format(keras.backend.get_value(model.optimizer.lr)))

print("[INFO] Traing model .............................")
model.fit_generator( aug.flow(trainX, trainY, batch_size=cfg.BATCH_SIZE), validation_data=(testX, testY), epochs = cfg.NUM_EPOCH, steps_per_epoch = len(trainX) // cfg.BATCH_SIZE, callbacks = callbacks, verbose= 1)



