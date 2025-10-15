import cv2
import os
import time
import uuid

path = os.path.abspath(os.path.join(os.getcwd(), 'Documents', 'images'))
print(path)

capture = cv2.VideoCapture(0)

while True:
    is_ok, frame = capture.read()

    cv2.imshow('frame', frame)

    key = cv2.waitKey(1) & 0xFF

    # "c" to capture
    if key == ord('c'):
        # Create a random name for each image
        image_name = os.path.join(path, "image" + "-" + '{}.png'.format(str(uuid.uuid1())))

        cv2.imwrite(image_name, frame)
        time.sleep(0.1)

        print("Capturing Image!")

    # "q" to exit
    elif key == ord('q'):
        print("Exitting...")
        break

capture.release()
cv2.destroyAllWindows()