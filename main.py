import os
import requests
import time
import datetime as dt
import cv2
import numpy as numpy
import math
import glob
import pysftp

### Tapo IP Cam credentials

username = "asifhaider"
password = "tapo332"
ip = "192.168.0.103"
port = "554"
stream = "stream2"

### Collection Duration
DURATION = 60

### Remote PI ftp connection
ip_remote = "103.129.37.38"
username_remote = "pi"
password_remote = "system"

###########################################################

def main():
    main_start = time.time()
    current_dt = dt.datetime.now()
    src = "rtsp://"+username+":"+password+"@"+ip+":"+port+"/"+stream

    #############################################################################
    print(src)
    cap = cv2.VideoCapture(src)
    
    count = 0

    path = os.getcwd()
    dataPath = path + "/data/" + current_dt.strftime("%Y-%m-%d-%H-%M-%S")
    os.mkdir(dataPath)

    start = time.time()
    while(cap.isOpened()):
        current = time.time()
        ret, frame = cap.read()
        count += 1
        filename = current_dt.strftime("%Y-%m-%d-%H-%M-%S")+str(count)+".jpg"
        
        
        if ret == True:
            # frame = cv2.resize(frame, (1024, 576))
            if math.floor(current - start) % 3 == 0:
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
    import shutil

    shutil.make_archive(dataPath, 'zip', dataPath)


    #############################################################################
    with pysftp.Connection(ip_remote, username=username_remote, password=password_remote) as sftp:
       with sftp.cd('dataPayload'):
           sftp.put(remotepath="",localpath=dataPath+'.zip')

    #############################################################################
    
    main_end = time.time()
    print(math.floor(main_end - main_start))
if __name__ == "__main__":
    main()

