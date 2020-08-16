from tensorflow.keras.callbacks import BaseLogger
import matplotlib.pyplot as plt
import numpy as np
import json
import os

class TrainMonitor(BaseLogger):

    def __init__(self, figPath,jsonPath, startAt=0):
        super(TrainMonitor, self).__init__()
        self.jsonPath = jsonPath
        self.figPath = figPath
        self.startAt = startAt
        
    def on_train_begin(self, log={}):
        self.H = {}
        if os.path.exists(self.jsonPath):
            self.H = json.loads( open(self.jsonPath).read())
            if self.startAt > 0:
                for k in self.H.keys():
                    self.H[k] = self.H[k][:self.startAt]
    def on_epoch_end(self, epoch, log={}):
        for (k,v) in log.items():
            l = self.H.get(k, [])
            l.append(float(v))
            self.H[k] = l 
        if os.path.exists(self.jsonPath):
            f = open(self.jsonPath, "w")
            f.write(json.dumps(self.H))
            f.close()
        if len(self.H['loss']) > 0:
            N = np.arange(len(self.H['loss']))
            plt.style.use("ggplot")
            plt.figure()
            plt.plot(N, self.H["loss"], label='Train Loss')
            plt.plot(N, self.H['val_loss'], label="Val Loss")
            plt.plot(N, self.H['accuracy'], label='Train Accuracy')
            plt.plot(N, self.H['val_accuracy'], label='Val Accuracy')
            plt.xlabel("#Epoch")
            plt.ylabel("Loss/Accracy ")
            plt.title("Loos and Accuracy Figure on Epoch {}".format(len(self.H['loss'])))
            plt.legend()
            plt.savefig(self.figPath)