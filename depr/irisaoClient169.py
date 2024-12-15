# -*- coding: utf-8 -*-
"""
Created on Fri Apr  2 11:23:33 2021

@author: UCSCL
"""

import zmq
import numpy as np
import time

port="5557"
context = zmq.Context()
socket = context.socket(zmq.REQ)
socket.connect("tcp://128.114.23.114:%s" % port)

def applySegment(segment, z, xgrad, ygrad):
    socket.send(np.array([segment, z, xgrad, ygrad]).astype(np.float32));
    msg=socket.recv()

def applyMode(mode, amplitude):
    socket.send(np.array([mode, amplitude]).astype(np.float32));
    msg=socket.recv()

def initMirror():
    socket.send(np.array([1]).astype(np.float32));
    msg=socket.recv()

def releaseMirror():
    socket.send(np.array([0]).astype(np.float32));
    msg=socket.recv()

def stopServer():
    socket.send(np.array([2]).astype(np.float32));
    msg=socket.recv()

def pistonAll(pauseTime):
    # piston each segment
    for k in range(1, 168):
        print('Pushing on segment '+str(k))
        applySegment(k, 0.2, 0, 0)
        time.sleep(pauseTime)
        applySegment(k, -0.2, 0, 0)
        time.sleep(pauseTime)
        applySegment(k, 0., 0, 0)

