import os
import requests
import time
import datetime as dt
import cv2
import numpy as np
import imutils
from imutils.object_detection import non_max_suppression
import math
import glob
import pysftp
import shutil

MAX_STORAGE = 1500

### Tapo IP Cam credentials

username = "asifhaider"
password = "tapo332"
ip = "192.168.0.103"
port = "554"
stream = "stream1"

### Collection Duration in seconds
DURATION = 45

### Remote PI ftp connection
ip_remote = "103.129.37.38"
username_remote = "pi"
password_remote = "system"

########################### Function to return dir size ##################################

# def get_size(path):
#     total = 0
#     for dirpath, dirnames, files in os.walk(path):
#         for i in files:
#             f = os.path.join(dirpath, i)
            
#             total += os.path.getsize(f)
#     return total


#########################################################################################

def detect(frame):
    cordinates, weights =  HOGCV.detectMultiScale(frame, winStride = (4, 4), padding = (8, 8), scale = 1.03)
    orig = frame.copy()
    person = 0
    for x,y,w,h in cordinates:
        cv2.rectangle(orig, (x,y), (x+w,y+h), (0,255,0), 2)
        person += 1

    # apply non-maxima suppression to the bounding boxes using a
	# fairly large overlap threshold to try to maintain overlapping
	# boxes that are still people
    cordinates = np.array([[x, y, x + w, y + h] for (x, y, w, h) in cordinates])
    pick = non_max_suppression(cordinates, probs=None, overlapThresh=0.65)
	# draw the final bounding boxes
    for (xA, yA, xB, yB) in pick:
        cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 2)

    # cv2.putText(frame, 'Status : Detecting ', (40,40), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255,0,0), 2)
    # cv2.putText(frame, f'Total Persons : {person-1}', (40,70), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255,0,0), 2)
    # cv2.imshow('output', frame)

    return [frame, pick]


###########################################################

def cropify(image, rects):
    cropped_img_arr = []
    # img = imutils.resize(image, width = min(800, image.shape[1]))
    for x,y,w,h in rects:
        # print(x+w / y+h)
        dim = (150, 300)
        crop = image[y: h , x: w]
        resized = cv2.resize(crop, dim, interpolation = cv2.INTER_AREA)
        # newH = resized.shape[0]
        # newW = resized.shape[1]
        # print(newH, newW)
        # cv2.rectangle(image, (x,y), (x+w,y+h), (0,255,0), 2)
        cropped_img_arr.append(resized)
    return cropped_img_arr

#################################################################################
def main():
    main_start = time.time()
    # current_dt = dt.datetime.now()
    src = "rtsp://"+username+":"+password+"@"+ip+":"+port+"/"+stream

    #############################################################################
    print(src)
    cap = cv2.VideoCapture(0)
    
    count = 0

    # path = os.getcwd()
    # Make Main Data folder
    # print("Creating Directories")
    # dataDir = os.path.join(path, "data") 
    
    # make directories
    # try:
    #     os.mkdir(dataDir)
    # except OSError as error:
    #     print(error)

    # dataPath = os.path.join(dataDir, current_dt.strftime("%Y-%m-%d-%H-%M-%S"))
    # os.mkdir(dataPath)
    start = time.time()
    print("Starts recording")
    
    while(cap.isOpened()):
        current = time.time()
        ret, frame = cap.read()
        # filename_root = current_dt.strftime("%Y-%m-%d-%H-%M-%S")+str(count)
        # # filePath = os.path.join(dataPath, filename)

        # filePath =  os.path.join(dataPath, filename_root)

        if ret == True:
            # orig = frame.copy()
            count += 1
            detections = detect(frame)
            
            rects = detections[1]
            img = detections[0]
            
            cv2.imshow("Detections", img)
            print(rects)
            
            # if len(rects) > 0:
            #     cropped = cropify(orig, rects)
            #     print("Detected: ", len(cropped))
            #     inner_count = 0
                
            #     # Writing positive samples
            #     for img in cropped:
            #         # cv2.imshow("Resized "+str(count), img)
            #         inner_count += 1
            #         fileName = filePath+str(inner_count)+"-cropped.png"
            #         #if abspath is not None:
            #         try:
            #             cv2.imwrite(fileName, img)
            #         except Exception as e:
            #             print(e)

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
    # print("Compressing...")
    

    # shutil.make_archive(dataPath, 'zip', dataPath)


    # #############################################################################
    # print("Sending to Raiyan")
    # try:
    #     with pysftp.Connection(ip_remote, username=username_remote, password=password_remote) as sftp:
    #         with sftp.cd('dataPayload'):
    #             sftp.put(remotepath="",localpath=dataPath+".zip")
        
    # except Exception as e:
    #     print(e)
    # #############################################################################

    # try:
    #     # Clear Storage upto a certain limit
    #     print("Directory size: "+str(math.floor(get_size(dataDir) / (1024*1024)))+" MB")
    #     if math.floor(get_size(dataPath) / (1024*1024)) > MAX_STORAGE:
    #         print("Clearing Storage, exceeded MAX LIMIT: "+str(MAX_STORAGE))
    #         shutil.rmtree(dataPath)
    # except Exception as e:
    #     print(e)

    #############################################################################
    main_end = time.time()
    print("Elapsed: "+str(math.floor(main_end - main_start))+" s")

if __name__ == "__main__":
    HOGCV = cv2.HOGDescriptor()
    HOGCV.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    main()

