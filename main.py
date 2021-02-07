import os
import requests
import time
import datetime as dt
import cv2
import numpy as numpy
import math
import glob
import pysftp

username = "asifhaider"
password = "tapo332"
ip = "192.168.43.121"
port = "554"
stream = "stream2"

DURATION = 57


def main():
    main_start = time.time()
    current_dt = dt.datetime.now()
    src = "rtsp://"+username+":"+password+"/"+ip+":"+port+"/"+stream

    #############################################################################

    cap = cv2.VideoCapture("http://dev.ddad-bd.com/storage/campaigns/2/videos/JwCsYLw7hXrfTpC6xZVvvzfaQmFVe99xZGUGj9rw.webm")
    
    
    count = 0

    path = os.getcwd()
    dataPath = path + "/data/" + current_dt.strftime("%Y-%m-%d-%X")
    os.mkdir(dataPath)

    start = time.time()
    while(cap.isOpened()):
        current = time.time()
        ret, frame = cap.read()
        count += 1
        filename = str(count)+".jpg"
        if ret == True:
            cv2.imshow("Test",frame)
            # print(math.floor(current - start))
            cv2.imwrite(dataPath+"/"+filename, frame)
            if math.floor(current - start) > DURATION:
                break
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        else:
            break
    cv2.destroyAllWindows()
    cap.release()

    #############################################################################

    with pysftp.Connection('103.129.37.38', username="pi", password="system") as sftp:
        with sftp.cd('dataPayload'):
            sftp.put_r(remotepath="",localpath=dataPath)

    main_end = time.time()
    print(math.floor(main_end - main_start))
if __name__ == "__main__":
    main()

