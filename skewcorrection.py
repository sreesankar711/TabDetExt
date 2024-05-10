import os
import numpy as np
import cv2
from PIL import Image


def get_rotation_angle(input_image):
    input_image = np.array(input_image)  
    gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)
    thresh = cv2.threshold(gray, 0, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(thresh > 0))
    
    angle = cv2.minAreaRect(coords)[-1]
    
    if angle >= 45:
        angle = -90 + angle
            
    return angle


image = Image.open(image_path)
(h, w) = image.size
center = (w // 2, h // 2) 
M = cv2.getRotationMatrix2D(center, -get_rotation_angle(image_path)[0], 1.0) 
rotated = cv2.warpAffine(np.array(image), M, (w, h),flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
rotated = Image.fromarray(rotated)
