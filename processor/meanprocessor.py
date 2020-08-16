import cv2
class MeanProcessor:
    def __init__(self, rMean, bMean, gMean):
        self.rMean = rMean
        self.bMean = bMean
        self.gMean = gMean
    def process(self, image):
        b,g,r = cv2.split(image.astype("float32"))
        b -= self.bMean
        g -= self.gMean
        r -= self.rMean
        image = cv2.merge([b,g,r])
        return image