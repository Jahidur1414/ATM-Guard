from imutils.video import FileVideoStream
from imutils.video import FPS
import numpy as np
from imutils.video import VideoStream
import imutils
import time
import argparse
import cv2

from scripts import load_model, predictEmotion


# ap = argparse.ArgumentParser()
# ap.add_argument("-m", "--model", required=True, help="path to trained human activity recognition model")
# args = vars(ap.parse_args())

class_file = 'action_recognition_kinetics.txt'
CLASSES = open(class_file).read().strip().split("\n")
SAMPLE_DURATION = 16
SAMPLE_SIZE = 112

print("[INFO] loading human activity recognition model...")
# net = cv2.dnn.readNet(args["model"])
net = cv2.dnn.readNet('models/resnet-34_kinetics.onnx')

model = load_model()
print("[INFO] Jahidur Rahman Fahim")
print("[INFO] All Done..... System Geting Ready !")
print("[INFO] loading face detection model...")
face_net = cv2.dnn.readNetFromCaffe('face_detect/deploy.prototxt.txt',
                                    'face_detect/res10_300x300_ssd_iter_140000.caffemodel')
print("[INFO] loading eyes casecades")
eye_cascade = cv2.CascadeClassifier('models/haarcascade_eye.xml')

print("[INFO] accessing video stream...")


#--------------------------------------------------------

vs = VideoStream(src=0).start()

#----------------------------------------------------------

time.sleep(2.0)
fps = FPS().start()
fs = 0.0
while True:
    frames = []

    for i in range(0, SAMPLE_DURATION):
        frame = vs.read()
        frame = imutils.resize(frame, width=600)
        frames.append(frame)
        
    
    t1 = time.time()
    # now that our frames array is filled we can construct our blob
    blob = cv2.dnn.blobFromImages(frames, 1.0, (SAMPLE_SIZE, SAMPLE_SIZE), (114.7748, 107.7354, 99.4750), swapRB=True,
                                  crop=True)
    blob = np.transpose(blob, (1, 0, 2, 3))
    blob = np.expand_dims(blob, axis=0)

    # pass the blob through the network to obtain our human activity recognition predictions
    net.setInput(blob)
    outputs = net.forward()
    label = CLASSES[np.argmax(outputs)]

    for frame in frames:
        # face detection
        (h, w) = frame.shape[:2]
        blob_face = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        face_net.setInput(blob_face)
        detections = face_net.forward()
        nof_j = 0
        for j in range(0, detections.shape[2]):
            confidence = detections[0, 0, j, 2]
            if confidence < 0.5:
                continue
            nof_j = j
            box = detections[0, 0, j, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype('int')

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            roi_gray = gray[startY:endY, startX:endX]
            roi_color = frame[startY:endY, startX:endX]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
            # frame_name = 'roi '+ str(i)
            # cv2.imshow(frame_name, roi_gray)
            emotion = predictEmotion(model, cropped_img)
            #emotion = 'commented'
            # id, predicted-emotion
            text = str(j) + ' ' + emotion
            color = (0, 255, 0)
            font = cv2.FONT_HERSHEY_SIMPLEX
            thickness = 2

            # draw box and face id
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(frame, (startX - 1, startY - 25), (endX + 1, startY), color, -1)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, thickness)
            cv2.putText(frame, text, (startX + 5, startY - 5), font, .5, (255, 255, 255), 1)

            # eye detection
            eyes = eye_cascade.detectMultiScale(roi_gray)
            for ex, ey, ew, eh in eyes:
                roi = roi_color[ey:ey + ew, ex:ex + eh]
                rows, cols, _ = roi.shape
                gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                gray_roi = cv2.GaussianBlur(gray_roi, (7, 7), 0)
                _, threshold = cv2.threshold(gray_roi, 127, 255, cv2.THRESH_BINARY_INV)
                contours, hierachy = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
                for cnt in contours:
                    (x, y, w, h) = cv2.boundingRect(cnt)

                    cv2.rectangle(roi, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    cv2.line(roi, (x + int(w / 2), 0), (x + int(w / 2), rows), (0, 255, 0), 1)
                    cv2.line(roi, (0, y + int(h / 2)), (cols, y + int(h / 2)), (0, 255, 0), 1)
                    break
                # cv2.imshow("Threshold", threshold)
                # cv2.imshow("gray roi", gray_roi)
                # cv2.imshow("Roi", roi)
        nof = nof_j
        cv2.putText(frame, 'Activity: ' + label, (5, 43), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1, cv2.LINE_AA)
        total_faces = 'Total Faces in Frame: ' + str(nof + 1)
        cv2.putText(frame, total_faces, (5, 68), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1, cv2.LINE_AA)
        
        fs = ( fs + (1.0/(time.time()-t1)) ) / 2.0
        
        cv2.putText(frame, "FPS: {:.2f}".format(fs), (5, 18),
                          cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)
        cv2.imshow("Activity Recognition", frame)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break
    fps.update



fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()