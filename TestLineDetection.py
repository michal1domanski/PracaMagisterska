import cv2
import carla
from CDetectLane import CDetectLane
import os
import time

#GLOBAL for test_speed function
ARRAY = []

def test_function():
    img = cv2.imread("PracaMagisterska/test.png")
    if img is None:
        print("Error: Image not found or unable to load.")
        return  # Exit the function if image is not loaded
    t1 = time.time()
    img = CDetectLane().detect_lines(img)
    t2 = time.time()
    cv2.imwrite("out/out.png", img)
    ARRAY.insert(-1,(t2-t1))

def test_speed(iterations):
    for i in range(iterations):
        test_function()   
    print("Average: ", sum(ARRAY) / len(ARRAY))

def run_single_time():
    test_function()

#run_single_time()
test_speed(100)