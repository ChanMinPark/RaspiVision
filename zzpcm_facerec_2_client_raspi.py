#!/usr/bin/env python

import sys
import cv2
import numpy as np
import socket
import time


face_cascade = cv2.CascadeClassifier("[change Dir]/haarcascades/haarcascade_frontalface_alt.xml")


def recv_data(sock, count):
    buf = b''
    while count:
        newbuf = sock.recv(count)
        if not newbuf: return None
        buf += newbuf
        count -= len(newbuf)
    return buf


def register_proc(name):
    #To do
    print("Begin register process.")

    # 'ip'and 'port' must be changed depend on your System.
    ip = '192.168.0.72'
    port = 10100  #My Receive port

    ss = socket.socket()
    ss.connect(('192.168.0.55', 10200))
  
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((ip, port))
    s.listen(True)

    cam = cv2.VideoCapture(0)
    cam.set(cv2.CAP_PROP_FPS, 7)

    for count in range(10):
        try:
            ret, frame = cam.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.2, 3)
            for (x,y,w,h) in faces:
                resized_image = cv2.resize(frame[y:y+h,x:x+w], (256, 256))
                
                encode_param=[int(cv2.IMWRITE_JPEG_QUALITY),90]
                result, imgencode = cv2.imencode('.jpg', resized_image, encode_param)
                data = numpy.array(imgencode)
                stringData = data.tostring()
                metaMsg = 'r/'+str(port)+str(len(stringData)).ljust(16)+name.ljust(10)
                
                ss.send(metaMsg)
                ss.send(stringData)
            
            time.sleep(1)
        except KeyboardInterrupt:
            s.close()
            ss.close()
            sys.exit()
            
    print "Register process is end."
    s.close()
    ss.close()


def operation_proc():
    #To do
    print("Begin register process.")

    # 'ip'and 'port' must be changed depend on your System.
    ip = '192.168.0.72'
    port = 10100  #My Receive port

    ss = socket.socket()
    ss.connect(('192.168.0.67', 10200))
  
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((ip, port))
    s.listen(True)

    cam = cv2.VideoCapture(0)
    cam.set(cv2.CAP_PROP_FPS, 7)

    resultData = "Unknown"
    for count in range(1):
        try:
            ret, frame = cam.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.2, 3)
            for (x,y,w,h) in faces:
                resized_image = cv2.resize(frame[y:y+h,x:x+w], (256, 256))
                
                encode_param=[int(cv2.IMWRITE_JPEG_QUALITY),90]
                result, imgencode = cv2.imencode('.jpg', resized_image, encode_param)
                data = numpy.array(imgencode)
                stringData = data.tostring()
                name="none"
                metaMsg = 'o/'+str(port)+str(len(stringData)).ljust(16)+name.ljust(10)
                
                ss.send(metaMsg)
                ss.send(stringData)

                rs = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                rs.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                rs.bind((myip, myport))
                rs.listen(True)

                con, saddr = rs.accept()
                resultData = recv_data(con, 10)
                print resultData
                
            #time.sleep(1)
        except KeyboardInterrupt:
            s.close()
            ss.close()
            sys.exit()
            
    print "Register process is end."
    s.close()
    ss.close()
            

if __name__ == '__main__':
    menu = int(raw_input('Select main menu ( 1: Register your face, 2: Start ) : '))

    if menu == 1:
        name = raw_input('Enter your name : ')
        if len(name) > 10:
            name = name[:10]
        register_proc(name)
    elif menu == 2:
        operation_proc()
    else:
        print "Incorrect menu. Terminate program."

    
  
