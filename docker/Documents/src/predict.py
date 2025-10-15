import ultralytics
from ultralytics import YOLO
import cv2
import os
import time
import numpy as np

from detector import Detector


detector = Detector('Documents/save/train2/weights/best.pt')

while True:
    is_ok, input = detector.get_frame()
    frame = input

    if not is_ok:
        continue

    results = detector.get_results(input, 'cuda', 0.6)

    for result in results:
        boxes = result.boxes
        masks = result.masks
        frame = detector.draw_all(frame, result)
        
    cv2.imshow('frame', frame)

    key = cv2.waitKey(1) & 0xFF

    # Press "q" to exit
    if key == ord('q'):
        print("Exitting...")
        break

    

detector.release()
cv2.destroyAllWindows()