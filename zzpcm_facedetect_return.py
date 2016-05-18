#!/usr/bin/env python

import sys, getopt

import cv2
import numpy as np

#from video import create_capture
from common import clock, draw_str

help_message = '''
USAGE: zzpcm_facedetect_return.py --mode (local/network) [--sdir <save_dir>] [--sname <save_name>]
'''

def detect(img, cascade):
    rects = cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=4, minSize=(30, 30),
                                     flags=cv2.CASCADE_SCALE_IMAGE)
    if len(rects) == 0:
        return []
    rects[:,2:] += rects[:,:2]
    return rects

def facedetect(img):
  #To do
  resultflag = 0
  
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  gray = cv2.equalizeHist(gray)
  cascade = cv2.CascadeClassifier("../../data/haarcascades/haarcascade_frontalface_alt.xml")
  
  rects = detect(gray, cascade)
  if len(rects)!=0:
    resultflag=1
  
  return resultflag

def saveImage(img, s_dir, s_name):
  #To do
  if s_dir[-1:] != "/":
    s_dir = s_dir+'/'
  print("Save Directory and File name : "+s_dir+s_name)
  cv2.imwrite(s_dir+s_name, img)

def localMode(s_dir, s_name):
  #To do
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


if __name__ == '__main__':
  print help_message
  
  opts, args = getopt.getopt(sys.argv[1:], '', ['mode=', 'sdir=', 'sname='])
  args = dict(args)
  running_mode = args.get('--mode', 'local')
  save_dir = args.get('--sdir', '../saveImage/')
  save_name = args.get('--sname', 'tempImage.jpg')
  
  if running_mode =='local':
    localMode(save_dir, save_name)
  elif running_mode == 'network':
    networkMode()
  else:
    print("Please type correct mode.")
  
