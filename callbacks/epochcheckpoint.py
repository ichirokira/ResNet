from tensorflow.keras.callbacks import Callback
import os

class EpochCheckpoint(Callback):
    def __init__(self, output, every, startAt):
        super(EpochCheckpoint, self).__init__()
        self.output = output
        self.every = every
        self.intEpoch = startAt
    def on_epoch_end(self, epoch, log={}):
        if (self.intEpoch + 1) % self.every == 0:
            p = os.path.sep.join([self.output, "epoch_{}_{}.hdf5".format(self.intEpoch+1, log['val_accuracy'])])
            self.model.save(p, overwrite=True)
        self.intEpoch += 1
        
