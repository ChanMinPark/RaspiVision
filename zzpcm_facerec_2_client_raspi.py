#!/usr/bin/env python

import sys
import cv2
import numpy as np
import socket
import time

# This variables must be changed depend on your System.
face_cascade = cv2.CascadeClassifier("/home/pi/Desktop/opencv-3.0.0/data/haarcascades/haarcascade_frontalface_alt.xml")
serverip = '192.168.0.72'
serverport = 10200
# 'ip'and 'port' must be changed depend on your System.
myip = '192.168.0.81'
myport = 10100  #My Receive port

def recv_data(sock, count):
    buf = b''
    while count:
        newbuf = sock.recv(count)
        if not newbuf: return None
        buf += newbuf
        count -= len(newbuf)
    return buf


def register_proc(name):
    print("Begin register process.")

    ss = socket.socket()
    ss.connect((serverip, serverport))
    
    count = 0
    while count < 10:
        try:
            cam = cv2.VideoCapture(0)
            
            ret, frame = cam.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.2, 3)
            if len(faces) == 0:
                print "Look at the Camera squarely."
            else:
                print "took a picture #"+str(count+1)
            for (x,y,w,h) in faces:
                resized_image = cv2.resize(frame[y:y+h,x:x+w], (273, 273))
                
                encode_param=[int(cv2.IMWRITE_JPEG_QUALITY),90]
                result, imgencode = cv2.imencode('.jpg', resized_image, encode_param)
                data = np.array(imgencode)
                stringData = data.tostring()
                metaMsg = 'r/'+str(12345)+'/'+str(len(stringData)).ljust(16)+'/'+name.ljust(10)
                
                ss.send(metaMsg)
                ss.send(stringData)
                count = count+1
            del(cam)
            time.sleep(0.5)
        except KeyboardInterrupt:
            ss.close()
            sys.exit()
            
    print "Register process is end."
    ss.close()


def operation_proc():
    print("Begin operation process.")
    
    ss = socket.socket()
    rs = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    cam = cv2.VideoCapture(0)

    resultData = "Unknown"
    for count in range(1):
        try:
            print("Capture...")
            ret, frame = cam.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.2, 3)
            if len(faces) == 0:
                print "Look at the Camera squarely and then Restart the program."
            else:
                print "took a picture"
            for (x,y,w,h) in faces:
                ss.connect((serverip, serverport))
                
                resized_image = cv2.resize(frame[y:y+h,x:x+w], (273, 273))
                
                print("Encoding...")
                encode_param=[int(cv2.IMWRITE_JPEG_QUALITY),90]
                result, imgencode = cv2.imencode('.jpg', resized_image, encode_param)
                data = np.array(imgencode)
                stringData = data.tostring()
                name="none"
                metaMsg = 'o/'+str(myport)+'/'+str(len(stringData)).ljust(16)+'/'+name.ljust(10)
                
                print("Sending...")
                ss.send(metaMsg)
                ss.send(stringData)

                rs.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                rs.bind((myip, myport))
                rs.listen(True)

                con, saddr = rs.accept()
                resultData = recv_data(con, 10)
                print "You are "+resultData.strip()
                
        except KeyboardInterrupt:
            rs.close()
            ss.close()
            sys.exit()
            
    print "Operation process is end."
    rs.close()
    ss.close()
            

if __name__ == '__main__':
    menu = int(raw_input('Select main menu ( 1: Register your face, 2: Start ) : '))

    if menu == 1:
        name = raw_input('Enter your name (under 10 charcter) : ')
        if len(name) > 10:
            name = name[:10]
        register_proc(name)
    elif menu == 2:
        operation_proc()
    else:
        print "Incorrect menu. Terminate program."
