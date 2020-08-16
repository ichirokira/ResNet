from sklearn.feature_extraction.image import extract_patches_2d
class PatchProcessor:
    def __init__(self, width, height):
        self.width = width
        self.height = height
    def process(self, image):
        return extract_patches_2d( image, (self.height, self.width),max_patches=1)[0]