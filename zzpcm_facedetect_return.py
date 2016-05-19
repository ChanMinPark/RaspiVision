#!/usr/bin/env python

import sys, getopt

import cv2
import numpy

#from video import create_capture
from common import clock, draw_str

import socket

help_message = '''
USAGE: zzpcm_facedetect_return.py --mode (local/network) [--sdir <save_dir>] [--sname <save_name>]
Default : local mode, ../saveImage/, tempImage.jpg
'''

def detect(img, cascade):
    #call by facedetect
    rects = cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=4, minSize=(30, 30),
                                     flags=cv2.CASCADE_SCALE_IMAGE)
    if len(rects) == 0:
        return []
    rects[:,2:] += rects[:,:2]
    return rects

def facedetect(img):
  #Local, Network
  resultflag = 0
  
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  gray = cv2.equalizeHist(gray)
  cascade = cv2.CascadeClassifier("../../data/haarcascades/haarcascade_frontalface_alt.xml")
  
  rects = detect(gray, cascade)
  if len(rects)!=0:
    resultflag=1
  
  return resultflag

def saveImage(img, s_dir, s_name):
  #Local
  if s_dir[-1:] != "/":
    s_dir = s_dir+'/'
  print("Save Directory and File name : "+s_dir+s_name)
  cv2.imwrite(s_dir+s_name, img)

def recv_data(sock, count):
    buf = b''
    while count:
        newbuf = sock.recv(count)
        if not newbuf: return None
        buf += newbuf
        count -= len(newbuf)
    return buf


def localMode(s_dir, s_name):
  #To do
  print("Local Mode On")
  
  cam = cv2.VideoCapture(0)
  cam.set(cv2.CAP_PROP_FPS, 7)
  ret, img = cam.read()
  
  result = facedetect(img)
  
  if result==1:
    print("Save the Image.")
    saveImage(img, s_dir, s_name)
  else:
    print("There is no face. Do nothing.")

def networkMode():
  #To do
  print("Network Mode On")
  
  ip = 'localhost'
  port = 10100  #Request-receive port of Camera Device
  
  s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
  s.bind((ip, port))
  s.listen(True)
  
  while True:
    try:
      conn, addr = s.accept()
      print("Request is accepted :")
      print(addr)
      msg = recv_data(conn, 13)
      print(", msg: "+msg)
      
      if msg[:7] == 'request':
        cam = cv2.VideoCapture(0)
        cam.set(cv2.CAP_PROP_FPS, 7)
        ret, img = cam.read()
        
        encode_param=[int(cv2.IMWRITE_JPEG_QUALITY),90]
        if facedetect(img)==1:
          result, imgencode = cv2.imencode('.jpg', img, encode_param)
          data = numpy.array(imgencode)
          stringData = data.tostring()
          print("Send Image")
        else:
          stringData = 'noface'
          print("Send nothing.")
        
        ss = socket.socket()
        ss.connect((addr[0],int(msg[8:])))
        ss.send(str(len(stringData)).ljust(16))
        ss.send(stringData)
        
        del(cam)
    except KeyboardInterrupt:
      s.close()
      ss.close()
      sys.exit()

if __name__ == '__main__':
  print help_message
  
  opts, args = getopt.getopt(sys.argv[1:], '', ['mode=', 'sdir=', 'sname='])
  opt = dict(opts)
  running_mode = opt.get('--mode', 'local')
  save_dir = opt.get('--sdir', '../saveImage/')
  save_name = opt.get('--sname', 'tempImage.jpg')
  
  if running_mode =='local':
    localMode(save_dir, save_name)
  elif running_mode == 'network':
    networkMode()
  else:
    print("Please type correct mode.")
  
