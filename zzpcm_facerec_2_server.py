#!/usr/bin/python
from facerec.feature import Fisherfaces
from facerec.classifier import NearestNeighbor
from facerec.model import PredictableModel
from PIL import Image
import numpy as np
import socket
import sys, os
import time
import cv2
import multiprocessing
import random

#'path' must be changed depend on your System.
path = '/home/pcm/drvs/facedatabase/'

#Load Image in Face Database
def read_images(path, sz=(256,256)):
    """Reads the images in a given folder, resizes images on the fly if size is given.
    Args:
        path: Path to a folder with subfolders representing the subjects (persons).
        sz: A tuple with the size Resizes 
    Returns:
        A list [X,y]
            X: The images, which is a Python list of numpy arrays.
            y: The corresponding labels (the unique number of the subject, person) in a Python list.
    """
    c = 0
    X,y = [], []
    folder_names = []
    
    for dirname, dirnames, filenames in os.walk(path):
        for subdirname in dirnames:
            folder_names.append(subdirname)
            subject_path = os.path.join(dirname, subdirname)
            for filename in os.listdir(subject_path):
                try:
                    im = cv2.imread(os.path.join(subject_path, filename), cv2.IMREAD_GRAYSCALE)
                    
                    # resize to given size (if given)
                    if (sz is not None):
                        im = cv2.resize(im, sz)
                    X.append(np.asarray(im, dtype=np.uint8))
                    y.append(c)
                except IOError, (errno, strerror):
                    print "I/O error({0}): {1}".format(errno, strerror)
                except:
                    print "Unexpected error:", sys.exc_info()[0]
                    raise
            c = c+1
    return [X,y,folder_names]

def recv_Data(sock, count):
    buf = b''
    while count:
        newbuf = sock.recv(count)
        if not newbuf: return None
        buf += newbuf
        count -= len(newbuf)
    return buf

def save_FaceDB(name, img):
    if not os.path.exists(path+name): os.makedirs(path+name)
    cv2.imwrite(path+name+'/'+str(random.randrange(100,1000))+'.jpg',img)
    
def checkFace(origin_img):
    #To do
    model = PredictableModel(Fisherfaces(), NearestNeighbor())
    #face_cascade = cv2.CascadeClassifier("/home/pi/Desktop/opencv-3.0.0/data/haarcascades/haarcascade_frontalface_alt.xml")
    
    result_name = 'unknown'
    
    [X,y,subject_names] = read_images(path)
    list_of_labels = list(xrange(max(y)+1))
    subject_dictionary = dict(zip(list_of_labels, subject_names))
    model.compute(X,y)

    #gray = cv2.cvtColor(origin_img, cv2.COLOR_BGR2GRAY)
    #faces = face_cascade.detectMultiScale(gray, 1.2, 3)

    #for (x,y,w,h) in faces:        
        #sampleImage = gray[y:y+h, x:x+w]
        #sampleImage = cv2.resize(sampleImage, (256,256))
    sampleImage = cv2.resize(origin_img, (256,256))

        
    [ predicted_label, generic_classifier_output] = model.predict(sampleImage)
    print [ predicted_label, generic_classifier_output]
        
    if int(generic_classifier_output['distances']) <=  700:
        result_name = str(subject_dictionary[predicted_label])

    return result_name



#TCP_IP = 'localhost'  #remote ip
#TCP_PORT = 10100      #remote port

#'myip'and 'myport' must be changed depend on your System.
myip = '192.168.0.55'
myport = 10200

rs = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
rs.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
rs.bind((myip, myport))
rs.listen(True)

#ss.send('request/'+str(myport))
while True:
    try:
        conn, addr = rs.accept()


        for count in range(10):
            metaData = recv_Data(conn,35)

            menu = metaData[:1]
            client_port = int(metaData[2:7])
            length = int(metaData[8:24])
            user_name = metaData[25:]

            stringData = recv_Data(conn, length)
            data = np.fromstring(stringData, dtype='uint8')
            decimg=cv2.imdecode(data,1)
            
            if menu=='r':
                print "Request is accepted : Register"
                save_FaceDB(user_name, decimg)
                print "Save "+user_name+"image #"+str(count+1)
            else:
                print "Request is accepted : Operation"
                who = checkFace(decimg)
                print("Result Face : "+who)
                ss = socket.socket()
                ss.connect((addr[0], client_port))
                ss.send(who.ljust(10))
                ss.close()
                break
    
    except KeyboardInterrupt:
        rs.close()
        sys.exit()

rs.close()
