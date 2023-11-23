import cv2


class CannyDetector:
    def __call__(self, img, low_threshold, high_threshold, apertureSize):
        return cv2.Canny(img, low_threshold, high_threshold, apertureSize=apertureSize)
