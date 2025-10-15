import ultralytics
from ultralytics import YOLO
from ultralytics.utils.ops import scale_image
import cv2
import os
import time
import numpy as np

class Detector:

    def __init__(self, model_path, capture_index=0):
        self.model = YOLO(model_path)
        self.capture = cv2.VideoCapture(capture_index)

    def release(self):
        self.capture.release()

    def get_frame(self) -> tuple[bool, cv2.Mat]:
        return self.capture.read()
    
    def get_results(self, frame, device='cuda', conf_thresh=0.9):
        return self.model.predict(frame, device=device, conf=conf_thresh)

    def is_result_empty(self, result):
        # If there's nothing in boxes then it's safe to say that there's no result
        return result.boxes.__len__() == 0
    
    def draw_crosshair(self, frame, point, length=40) -> cv2.Mat:
        # Create a copy of frame for output
        output = frame

        # point is (x, y)
        x = point[0]
        y = point[1]

        # horizontal, x +/- length/2
        output = cv2.line(output, (round(x - (length / 2)), round(y)), (round(x + (length / 2)), round(y)), (0, 0, 255), 2)
        # vertical, y +/- length/2
        output = cv2.line(output, (round(x), round(y - (length / 2))), (round(x), round(y + (length / 2))), (0, 0, 255), 2)

        return output
    
    def get_center(self, box: np.ndarray): 
        # If the box has no coordinates, do nothing
        if(box.__len__() == 0):
            return (0, 0)
        
        # box is in the format of [x1, y1, x2, y2]
        x1 = box[0].item()
        y1 = box[1].item()
        x2 = box[2].item()
        y2 = box[3].item()

        # Average between each ends
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        return (center_x, center_y)

    def draw_box(self, frame, box: np.ndarray) -> cv2.Mat:
        # If the box has no coordinates, do nothing
        if(box.__len__() == 0):
            return frame
        
        # box is in the format of [x1, y1, x2, y2]
        x1 = box[0].item()
        y1 = box[1].item()
        x2 = box[2].item()
        y2 = box[3].item()

        # Make the rectangle with the coordinates and round since opencv doesn't like decimals
        return cv2.rectangle(frame, (round(x1), round(y1)), (round(x2), round(y2)), (255, 0, 0), 2)
    
    def put_confidence(self, frame, box, confidence):
        # Create a copy of frame to output
        output = frame

        # Top Left corner
        x = box[0].item()
        y = box[1].item()

        # Round since opencv doesn't like decimals
        corner = (round(x), round(y))

        # Overlay
        output = cv2.putText(output, str(round(confidence, 2)), corner, cv2.FONT_HERSHEY_PLAIN, 4, (255, 0, 0), 4)

        return output
    
    def put_id(self, frame, box, id):
        # Create a copy of frame to output
        output = frame

        # Bottom left corner + a lot of padding
        x = box[0].item()
        y = box[3].item() + 80

        # Round since opencv doesn't like decimals
        corner = (round(x), round(y))

        # Overlay
        output = cv2.putText(output, 'id: ' + str(id), corner, cv2.FONT_HERSHEY_PLAIN, 4, (255, 0, 0), 4)

        return output
    
    def put_name(self, frame, box, name):
        # Create a copy of frame to output
        output = frame

        # Bottom Left corner + padding
        x = box[0].item() # x1
        y = box[3].item() + 40 #y2

        # Round since opencv doesn't like decimals
        corner = (round(x), round(y))

        # Overlay the text
        output = cv2.putText(output, name, corner, cv2.FONT_HERSHEY_PLAIN, 4, (255, 0, 0), 4)

        return output
    
    def get_names(self, dict: dict[int, str], classes: np.ndarray):
        # Empty array to append stuff into
        names = np.array([])
    
        # For each id that was detected
        for cls in classes:
            # Look up what name corresponds to the ID
            name = dict[round(cls.item())]

            # Append
            names = np.append(names, name)

        return names
    
    def get_ids(self, classes: np.ndarray):
        # Empty array to append stuff into
        ids = np.array([])

        # For each id that was detected
        for cls in classes:
            # Round since it's a decimal for some reason
            id = round(cls.item())

            # Append
            ids = np.append(ids, id)

        return ids

    def draw_all(self, frame, result) -> cv2.Mat:
        boxes = result.boxes
        masks = result.masks
        names = result.names
        classes = boxes.cls

        # If the result is empty then do nothing
        if(self.is_result_empty(result)):
            return frame
        
        # Make a copy for output
        output = frame
        
        # For each detection in the detections list
        for i in range(boxes.xyxy.__len__()):
            # Box for the detection
            box = boxes.xyxy[i]
            
            # Mask of the detection
            mask = masks.data.cpu().numpy()[i]

            # Confidence of this detection
            confidence = boxes.conf[i].item()

            # Name
            name = self.get_names(names, classes)[i]
            
            # ID
            id = round(self.get_ids(classes)[i])

            # Draw the bounding box
            output = self.draw_box(output, box)

            # Draw the Crosshair
            output = self.draw_crosshair(output, self.get_center(box))

            # Overlay the confidence
            output = self.put_confidence(output, box, confidence)

            # Overlay the name
            output = self.put_name(output, box, name)

            # Overlay the ID
            output = self.put_id(output, box, id)

            # Draw the mask
            output = self.draw_mask(frame, mask)

        return output
    
    def draw_mask(self, frame, mask, color=(255, 0, 0), alpha=0.3):
        # Gotta flip since resize uses (width, height) but numpy is (height, width)
        mask = cv2.resize(mask, (frame.shape[:2])[::-1])

        # Make this a (3, h, w) matrix since `frame` is (h, w, 3) 3 for BGR
        colored_mask = np.expand_dims(mask, 0).repeat(3, axis=0)

        # Move 0 axis to the end since `frame` is (h, w, 3) and `colored_mask` prior is (3, h, w)
        colored_mask = np.moveaxis(colored_mask, 0, -1)

        # Make the masked array object
        masked = np.ma.MaskedArray(frame, mask=colored_mask, fill_value=color)

        # Mask the actual frame
        overlay = masked.filled()

        # Overlay the overlay (very good english)
        return cv2.addWeighted(frame, 1 - alpha, overlay, alpha, 0)


    def draw_boxes(self, frame, xyxy: np.ndarray, with_crosshair=True) -> cv2.Mat:
        # If there's no boxes then do nothing
        if(xyxy.__len__() == 0):
            return frame
        
        # Make an output matrix
        output = frame

        # For every box in boxes
        for box in xyxy:
            # Update the output matrix with the box drawn
            output = self.draw_box(output, box)

            if with_crosshair:
                # Optionally, draw the crosshair
                output = self.draw_crosshair(output, self.get_center(box))

        return output
