import cv2
import numpy as np
import imutils
class AspectAwearProcessor:
    def __init__(self, width, height, inter = cv2.INTER_AREA):
        self.width = width
        self.height = height
        self.inter = inter
    def process(self, image):
        h,w = image.shape[:2]
        dW = 0
        dH = 0
        if w > h:
            img = imutils.resize(image, height= self.height, inter = self.inter)
            h,w = img.shape[:2]
            dW = int(0.5*(w - self.width))
        else:
            img = imutils.resize(image, width = self.width, inter = self.inter)
            h,w = img.shape[:2]
            dH = int(0.5*(h - self.height))
        h,w  = img.shape[:2]
        image = img[dH: h - dH, dW:w-dW]
        image = cv2.resize(image, (self.width, self.height), interpolation = self.inter)
        return image        

