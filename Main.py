import cv2
from darkflow.net.build import TFNet
import numpy as np
from PIL import ImageGrab
import matplotlib.pyplot as plt

option = {
    'model': 'cfg/yolov2.cfg',
    'load': 'cfg/yolov2.weights',
    'threshold': 0.15,
    'gpu': 1.0
}

tfnet = TFNet(option)
capture = cv2.VideoCapture('film.avi')
data = []
count = 0

def file():
    [capture.read() for i in range(4)]
    _, img = capture.read()
    return img

def screen():
    return np.array(ImageGrab.grab(bbox=(0, 0, 608, 608)))

def median_filter(x, window=3):
    add = np.zeros([int((window-1)/2)])
    add = [x[0] for y in add]
    x = np.append(add, x)
    add = [x[len(x)-1] for y in add]
    x = np.append(x, add)
    x_filtered = []
    for i in range(0, len(x)-(window-1), 1):
        median = np.median(x[i:i+window])
        x_filtered.append(median)
    return x_filtered

while True:
    count = 0
    frame = screen()
    for result in tfnet.return_predict(frame):
        if result['label'] in ['car', 'bus', 'truck']:
            tl = (result['topleft']['x'], result['topleft']['y'])
            br = (result['bottomright']['x'], result['bottomright']['y'])
            frame = cv2.rectangle(frame, tl, br, [0, 255, 0], 1)
            count += 1
    frame = cv2.putText(frame, str(count), (0, 40), 2, 1, (0, 255, 255), 2)
    data.append(count)
    cv2.imshow('', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break

data1 = median_filter(data)
plt.plot(data)
plt.plot(data1)
plt.show()