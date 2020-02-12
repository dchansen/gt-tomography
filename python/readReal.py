# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 11:31:46 2011

@author: Dae
"""
from __future__ import print_function
import numpy as np

class RealFile:
    def __init__(self,filename,dims,dtype=np.float32):
         self.ft = open(filename,'wb')
         self.dims = dims
         np.array(len(dims),dtype=np.uint32).tofile(self.ft)
         self.dtype = dtype
         self.closed = False
         self.elemsAdded = 0
         for i in range(len(dims)):
             np.array(dims[i],dtype=np.uint32).tofile(self.ft)
        
    def append(self,a):
         np.array(a,dtype=self.dtype).tofile(self.ft)
         self.elemsAdded += np.prod(a.shape)
    def close(self):
        if (self.elemsAdded != np.prod(self.dims)):
            print( 'Warning, number of elements in shape ', np.prod(self.dims), ' does not match number of elements added ',self.elemsAdded)
        self.close = True
        
        self.ft.close()
    def header(self,dims):
        if len(dims) != len(self.dims):
            print( 'Warning: New dimensions has different length than old dimension. Aborting!')
            return
        self.dims = dims
        self.ft.seek(0)
        np.array(len(dims),dtype=np.uint32).tofile(self.ft)
        for i in range(len(dims)):
             np.array(dims[i],dtype=np.uint32).tofile(self.ft)
        self.ft.seek(0,2)
        
def readReal(f):
    if( type(f) == type('s')):
        ft = open(f,'rb')
    else:   
        ft = f
        
    dims = np.fromfile(ft,dtype=np.uint32,count=1)
    print("dims",dims)
    shape= np.fromfile(ft,dtype=np.uint32,count=dims[0])
    curPos = ft.tell()
    ft.seek(0,2)
    size = ft.tell()-curPos
    ft.seek(curPos)

    if (size > np.prod(shape)*4):
        array = np.fromfile(ft,dtype=np.complex64)
    else:
        array = np.fromfile(ft,dtype=np.float32)
    array = array.reshape(shape[::-1])
    if( type(f) == 'str'):
            ft.close()
    
    return array
def saveReal(a,f,dtype=np.float32):
    if( type(f) == type('s')):
        ft = open(f,'wb')
    else:   
        ft = f
    s = a.shape
    np.array(len(s),dtype=np.uint32).tofile(ft)
    s = s[::-1]
    for i in range(len(s)):
        np.array(s[i],dtype=np.uint32).tofile(ft)
    np.array(a,dtype=dtype).tofile(ft)
    if( type(f) == 'str'):
            ft.close()
