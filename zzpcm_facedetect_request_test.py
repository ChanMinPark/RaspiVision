#!/usr/bin/python
import socket
import cv2
import numpy

def recv_Data(sock, count):
    buf = b''
    while count:
        newbuf = sock.recv(count)
        if not newbuf: return None
        buf += newbuf
        count -= len(newbuf)
    return buf

TCP_IP = 'localhost'  #remote ip
TCP_PORT = 10100      #remote port

myip = 'localhost'
myport = 10200

ss = socket.socket()
ss.connect((TCP_IP, TCP_PORT))
rs = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
rs.bind((myip, myport))
rs.listen(True)

ss.send('request/'+str(myport))

conn, addr = rs.accept()

length = recvall(conn,16)
stringData = recvall(conn, int(length))
data = numpy.fromstring(stringData, dtype='uint8')
ss.close()
rs.close()

decimg=cv2.imdecode(data,1)
cv2.imshow('SERVER',decimg)
cv2.waitKey(0)
cv2.destroyAllWindows() 
