import cv2
import carla
from CDetectLane import CDetectLane
import os

def test_function():
    print("Current working directory:", os.getcwd())
    img = cv2.imread(r"./PracaMagisterska/test.png")
    if img is None:
        print("Error: Image not found or unable to load.")
        return  # Exit the function if image is not loaded
    cv2.imshow("test", img)
    CDetectLane(img).detect_lines()
    cv2.imwrite("out.png", img)
    pass

test_function()    
    
        