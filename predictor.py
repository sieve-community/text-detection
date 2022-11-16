from typing import List
from sieve.types import FrameFetcher, SingleObject, BoundingBox, Temporal
from sieve.predictors import TemporalPredictor
from sieve.types.constants import FRAME_NUMBER, BOUNDING_BOX, CLASS, TEMPORAL
import os 
import cv2 
import easyocr

class TextDetector(TemporalPredictor):
    """
    This text detector uses EasyOCR to detect text in a frame and create a new SingleObject for each detected text.
    """

    def setup(self):
        self.reader = easyocr.Reader(['en'])
    
    def predict(self, frame: FrameFetcher) -> List[SingleObject]:
        # Get the frame array
        frame_data = frame.get_frame()
        # Convert to RGB
        frame_data = cv2.cvtColor(frame_data, cv2.COLOR_BGR2RGB)
        frame_number = frame.get_current_frame_number()

        # Resize the frame to at most 640x480
        height, width, _ = frame_data.shape
        max_height = 480
        max_width = 640
        if height > max_height or width > max_width:
            ratio = min(max_height / height, max_width / width)
            frame_data = cv2.resize(frame_data, (0, 0), fx=ratio, fy=ratio)
        else: 
            ratio = 1

        print("Predicting text on frame with shape", frame_data.shape)
        ret_dets = self.reader.readtext(frame_data)
        print("Predicted text")

        ret_objs = []

        for i in ret_dets:
            bbox_array = i[0]
            x1 = float(bbox_array[0][0])
            x2 = float(bbox_array[1][0])
            y1 = float(bbox_array[0][1])
            y2 = float(bbox_array[2][1])
            resized_bbox = {
                "x1": x1 / ratio,
                "y1": y1 / ratio,
                "x2": x2 / ratio,
                "y2": y2 / ratio
            }
            text_data = str(i[1]).lower()
            init_dict = {
                CLASS: "text",
                TEMPORAL: Temporal(**{
                    FRAME_NUMBER: frame_number,
                    BOUNDING_BOX: BoundingBox(**resized_bbox),
                    "text": text_data
                })
            }
            ret_objs.append(SingleObject(**init_dict))

        return ret_objs