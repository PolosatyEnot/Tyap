import cv2
from darkflow.net.build import TFNet
import numpy as np
from PIL import ImageGrab

option = {
    'model': 'cfg/yolov2.cfg',
    'load': 'cfg/yolov2.weights',
    'threshold': 0.15,
    'gpu': 1.0
}

tfnet = TFNet(option)
capture = cv2.VideoCapture('film.avi')

def file():
    #[capture.read() for i in range(4)]
    [capture.read() for i in range(4)]
    _, img = capture.read()
    return img

def screen():
    return np.array(ImageGrab.grab(bbox=(0,0,608,608)))

while True:
    frame = screen()
    for result in tfnet.return_predict(frame):
        if result['label'] in ['car', 'bus', 'truck']:
            tl = (result['topleft']['x'], result['topleft']['y'])
            br = (result['bottomright']['x'], result['bottomright']['y'])
            frame = cv2.rectangle(frame, tl, br, [0, 255, 0], 1)
    cv2.imshow('', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break