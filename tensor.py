import numpy as np
import cv2
import os
from keras.models import load_model
from collections import deque

IMG_SIZE = 128

def print_results(video):
    print("Loading model ...")
    model = load_model('./model.h5')
    Q = deque(maxlen=128)
    vs = cv2.VideoCapture(video)
    (W, H) = (None, None)
    fps = vs.get(cv2.CAP_PROP_FPS)
    frame_count = int(vs.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count/fps
    print(f"This videp is {duration} seconds long and {fps}fps")
    while True:
        (grabbed, frame) = vs.read()
        ID = vs.get(1)
        if not grabbed:
            break
        try:
            if (ID % fps == 0):
                if W is None or H is None:
                    (H, W) = frame.shape[:2]
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (128, 128)).astype("float32")
                frame = frame.reshape(IMG_SIZE, IMG_SIZE, 3) / 255
                preds = model.predict(np.expand_dims(frame, axis=0))[0]
                Q.append(preds)
                print(preds)
                i = (preds > 0.56)[0]  # np.argmax(results)
                text = f"Violence: {i}"
                print('prediction:', text)
        except:
            print('ERROR')
            break

#give video url or file path
print_results('https://dm0qx8t0i9gc9.cloudfront.net/watermarks/video/msqd2XJ/videoblocks-6320aeb0e44a0755799e2501_byciolaxs__c17c896dcdd2aa293fc9fdbd8b20b752__P360.mp4')
