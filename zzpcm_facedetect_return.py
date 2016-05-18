#!/usr/bin/env python

import sys, getopt

import cv2
import numpy as np

from video import create_capture
from common import clock, draw_str

help_message = '''
USAGE: zzpcm_facedetect_return.py --mode (local/network) [--sdir <save_dir>] [--sname <save_name>]
'''


def localMode():
  #To do

def networkMode():
  #To do


if __name == '__main__':
  print help_message
  
  args = getopt.getopt(sys.argv[1:], '', ['mode=', 'sdir=', 'sname='])
  running_mode = args.get('--mode', 'local')
  save_dir = args.get('--sdir', '../saveImage/')
  save_name = args.get('--sname', 'tempImage.jpg')
  
  if running_mode =='local':
    localMode()
  elif running_mode == 'network':
    networkMode()
  else:
    print("Please type correct mode.")
  
