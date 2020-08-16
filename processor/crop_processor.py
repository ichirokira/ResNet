import cv2
import numpy as np
class CropProcessor:
    def __init__(self, width, height, inter=cv2.INTER_AREA):
        self.width = width
        self.height = height
        self.inter = inter
    def process(self, image):
        h,w = image.shape[:2]
        coords = [[0, 0, self.width, self.height],
                [0,h-self.height, self.width, h],
                [w-self.width, 0, w, self.height],
                [w-self.width, h-self.height, w, h]]
        dW = int((w-self.width)*0.5)
        dH = int((h-self.height)*0.5)
        coords.append([dW,dH, w-dW, h-dH])
        crop = []
        for startX, startY, endX, endY in coords:
            img = image[startY:endY, startX:endX]
            img = cv2.resize(img, (self.width, self.height), interpolation = self.inter)
            crop.append(img)
        return np.array(crop)
