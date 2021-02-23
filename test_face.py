import os
import time
import datetime as dt
import numpy as np
import imutils
from imutils.object_detection import non_max_suppression
import math
import shutil
import cv2
from facenet_pytorch import MTCNN
import torch

# MAX_STORAGE = 1500

### Tapo IP Cam credentials

username = "asifhaider"
password = "tapo332"
ip = "192.168.0.103"
port = "554"
stream = "stream1"

### Collection Duration in seconds
DURATION = 45

### Remote PI ftp connection
# ip_remote = "103.129.37.38"
# username_remote = "pi"
# password_remote = "system"

########################### Function to return dir size ##################################

# def get_size(path):
#     total = 0
#     for dirpath, dirnames, files in os.walk(path):
#         for i in files:
#             f = os.path.join(dirpath, i)
            
#             total += os.path.getsize(f)
#     return total


#########################################################################################

# def detect(frame):
#     cordinates, weights =  HOGCV.detectMultiScale(frame, winStride = (4, 4), padding = (8, 8), scale = 1.03)
#     orig = frame.copy()
#     person = 0
#     for x,y,w,h in cordinates:
#         cv2.rectangle(orig, (x,y), (x+w,y+h), (0,255,0), 2)
#         person += 1

#     # apply non-maxima suppression to the bounding boxes using a
# 	# fairly large overlap threshold to try to maintain overlapping
# 	# boxes that are still people
#     cordinates = np.array([[x, y, x + w, y + h] for (x, y, w, h) in cordinates])
#     pick = non_max_suppression(cordinates, probs=None, overlapThresh=0.65)
# 	# draw the final bounding boxes
#     for (xA, yA, xB, yB) in pick:
#         cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 2)

#     # cv2.putText(frame, 'Status : Detecting ', (40,40), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255,0,0), 2)
#     # cv2.putText(frame, f'Total Persons : {person-1}', (40,70), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255,0,0), 2)
#     # cv2.imshow('output', frame)

#     return [frame, pick]


###########################################################

# def cropify(image, rects):
#     cropped_img_arr = []
#     # img = imutils.resize(image, width = min(800, image.shape[1]))
#     for x,y,w,h in rects:
#         # print(x+w / y+h)
#         dim = (150, 300)
#         crop = image[y: h , x: w]
#         resized = cv2.resize(crop, dim, interpolation = cv2.INTER_AREA)
#         # newH = resized.shape[0]
#         # newW = resized.shape[1]
#         # print(newH, newW)
#         # cv2.rectangle(image, (x,y), (x+w,y+h), (0,255,0), 2)
#         cropped_img_arr.append(resized)
#     return cropped_img_arr

#################################################################################
def main():
    main_start = time.time()
    # current_dt = dt.datetime.now()
    src = "rtsp://"+username+":"+password+"@"+ip+":"+port+"/"+stream

    #############################################################################
    print(src)
    cap = cv2.VideoCapture(0)
    
    count = 0

    
    start = time.time()
    print("Starts recording")
    
    while(cap.isOpened()):
        current = time.time()
        ret, frame = cap.read()
        
        if ret == True:
            count += 1
            frame = cv2.resize(frame, (600, 400))

            #Here we are going to use the facenet detector
            boxes, conf = mtcnn.detect(frame)
            # detections = detect(frame)
            
            if conf[0] !=  None:
                for (x, y, w, h) in boxes:
                    text = f"{conf[0]*100:.2f}%"
                    x, y, w, h = int(x), int(y), int(w), int(h)
                    cv2.putText(frame, text, (x, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1,(170, 170, 170), 1)
                    cv2.rectangle(frame, (x, y), (w, h), (255, 255, 255), 1)
            
            cv2.imshow("Detections", frame)
            print(boxes)
            
            
            if math.floor(current - start) > DURATION:
                break
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        else:
            break

    cv2.destroyAllWindows()
    cap.release()
    print("Ends recording")
    #############################################################################
    
    #############################################################################
    main_end = time.time()
    print("Elapsed: "+str(math.floor(main_end - main_start))+" s")

if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    #Create the model
    mtcnn = MTCNN(keep_all=True, device=device)
    main()

