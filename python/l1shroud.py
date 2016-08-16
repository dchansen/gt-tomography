# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 07:28:44 2015

@author: David
"""
from __future__ import division
from numpy import *
import tables
#import hndreader
from sklearn.decomposition import PCA 
from scipy.ndimage.filters import gaussian_filter, uniform_filter

def updateF(p,alpha,sigma):
    pabs =sqrt(p[:,:,0]**2+p[:,:,1]**2)
    return p/(1+alpha*sigma)/maximum(1,pabs[...,newaxis]/(1+alpha*sigma))
def updateF2(p,alpha,sigma):
    
    return p/(1+alpha*sigma)/maximum(1,abs(p/(1+alpha*sigma)))
#def updateG(u,g,tau,L):
#    return (u+tau*L*g)/(1+tau*L)
 
def updateG(u,g,tau,L):
    res = zeros(shape(u))
    res[u-g > tau*L] = u[u-g > tau*L]-tau*L
    res[u-g < tau*L] = u[u-g < tau*L]+tau*L
    res[abs(u-g) <= tau*L] = g[abs(u-g) <= tau*L]
    return res
def gradient_x(v):
    """Calculates the x component of the gradient transform of v"""
    res = zeros(shape(v),dtype=float64)
    res[:-1,:] = v[1:,:]-v[:-1,:]
    
    return res
def gradient_x_H(v):
    """ Calculates the transpose of the x component of the gradient transform of v"""
    res = zeros(shape(v),dtype=float64)
    res[0,:]= -v[0,:]
    res[1:-1,:] = v[:-2,:]-v[1:-1,:]
    res[-1,:] = v[-2,:]
    
    return res


def gradient_y(v):
    """ Calculates the y component of the gradient transform of v"""
    return transpose(gradient_x(transpose(v)))
def gradient_y_H(v):
    """ Calculates the transpose of the y component of the gradient transform of v"""
    return transpose(gradient_x_H(transpose(v)))


def identity(v): 
    """ Calculates the identity transform of v"""
    return array(v)
    
def identity_H(v):
    """ Calculates the transpose of the identity transform of v"""
    return identity(v)

def grad(v):
    s =shape(v)
    
    s2 = (s[0],s[1],2)
    res = zeros(s2)
    res[:-1,:,0] = v[1:,:]-v[:-1,:]
    res[:,:-1,1] = v[:,1:]-v[:,:-1]
    return res
def grad_H(v):
    return gradient_x_H(v[:,:,0])+gradient_y_H(v[:,:,1])
def l1denoise(x0, Lambda = 1, it = 100):
    x=x0

    L = 4

    xn = x
    xold = x
    tau=0.25
    theta=0.5
    gamma=0.35*Lambda
    sigma = 0.1/(L*tau)
    #sigma = 1
    y = grad(x)
    xold = array(x)
    for i in range(it):
        gx = grad(x)
        print sum((x-x0)**2)*Lambda+sum(sqrt(sum(gx**2,axis=2)))
        y = updateF(y+sigma*gx,0,sigma)
        xn = updateG(xn-tau*grad_H(y),x0,tau,Lambda)
        x = xn+theta*(xn-xold)
        xold = xn
    return x

def adaptive_filt(image):
    """Based on 'Robust breathing signal extraction from cone beam CT projections based on adaptive and global optimization techniques'
    Chao et al, PMB 2016"""
    ig = gaussian_filter(image,sigma=10)
    mean_image = uniform_filter(image,size=10)
    istd = sqrt(uniform_filter((mean_image-image)**2,size=10))
    im = mean(istd)
    return (image-ig)/(istd+im)+mean(image)

def save_binning(binning,nbins,filename,exclude=[]):

    binning2 = array(binning)
    for proj in exclude:
        binning2[proj] = nbins+1 #Exclude some projections completely

    f = tables.openFile(filename, mode="w")
    root = f.getNode("/")
    index = array(range(len(binning2)),dtype=uint32)
    for i in range(nbins):

        arr = f.createArray(root,"bin_"+str(i+1),array(index[binning2==i],dtype=uint32))
    f.createArray(root,"numBins",array([nbins],dtype=uint32))
    f.createArray(root,"binning_dataformat_version",array([1],dtype=uint32))

    f.close()


def pca_extract_signal(shroud,W=40):


    pca = PCA(n_components=5)
    pca.fit(transpose(shroud))
    primary = pca.components_[0,:]
    figure()
    plot(primary,label="primary")
    plot(pca.components_[1,:],label="secondary")
    legend()

    window = transpose(shroud[:,:W])
    pca.fit(window)
    primary = pca.components_[0,:]



    print shape(primary),norm(primary)

    signal = []

    for k in range(W//2):
        signal.append(dot(window[k,:],primary)/norm(window[k,:]))

    for i in range(1,nproj-W):
        window = transpose(dproj2[:,i:i+W])
        pca.fit(window)
        V = pca.components_;
        cc = abs(dot(primary,V[0,:]))
        m = 0
        for k in range(1,5):
            c2 = abs(dot(primary,V[k,:]))
            if (c2 > cc):
                cc = c2
                m = k
        m=0
        if dot(primary,V[m,:]) > 0:
            primary = V[m,:]
        else:
            primary = -V[m,:]
        #primary = V[m,:]/norm(V[m,:])
        frame = window[W//2,:]/norm(window[W//2,:])
        signal.append(dot(frame,primary))

    for k in range(nproj-W//2,nproj):

        signal.append(dot(dproj2[:,k],primary)/norm(dproj2[:,k]))

    signal = array(signal)
    return signal

def clean_signal(signal,nvals=10):
    from scipy.fftpack import rfft,rfftfreq,irfft
    signal2 = signal-mean(signal)
    fsignal = rfft(signal2)
    pabs = abs(fsignal)
    freq= rfftfreq(len(signal),d=60.0/len(signal))
    fLow = array(pabs)
    fLow[freq > 0.15 ] = 0
    fHigh = array(pabs)
    fHigh[freq < 0.15] = 0

    output = zeros_like(fsignal)
    fLowS = sort(fLow)
    output[fLow > fLowS[-nvals]] = fsignal[fLow > fLowS[-nvals]]
    fHighS = sort(fHigh)
    output[fHigh > fHighS[-nvals]] = fsignal[fHigh > fHighS[-nvals]]

    figure()
    plot(freq,pabs)
    return irfft(output)

def find_move(v1,v2,L=10):
    v1t = v1[L:-L]
    cur_min = norm(v1-v2,ord=1)
    b = 0
    for k in range(-L,L+1):
        vl2 = np.roll(v2,k,axis=0)[L:-L]

        knorm = norm(v1t-vl2,ord=1)
        print cur_min,knorm,k
        if (knorm < cur_min):
            b = -k
            cur_min = knorm
    return b

def extract_signal(shroud):
    b = zeros(shroud.shape[1])
    print shroud.shape
    for i in range(1,shroud.shape[1]):
        blocal = find_move(shroud[:,i-1],shroud[:,i])

        b[i] = b[i-1]+blocal
    return b


import matplotlib.pyplot as plt
import glob
import time

from imshow2 import imshow2 
#tstart = time.clock()
#files = glob.glob("*.hnd")
#files.sort()
#
#proj,head = hndreader.readHnd(files[0])
#projdims = [len(files),head.sizeY,head.sizeX]
#
#projections = zeros(projdims,dtype=float64)
#projections[0,:,:] = proj
#
#from scipy.ndimage.filters import gaussian_filter
#for i in range(1,len(files)):
#    proj,_ = hndreader.readHnd(files[i])
#    projections[i,:,:]=proj
#
#print "Loaded projections in", time.clock()-tstart, "seconds"
#projections =projections[:,8:-8,8:-8]
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("projections")
parser.add_argument("--output",metavar="f",dest="output",help="Output filename",default="binning.hdf5")
parser.add_argument("--breathing_cycle",dest="W",help="Approximate number of projections per breathing cycle",type=int,
                    default=40)
args = parser.parse_args()


ptab = tables.openFile(args.projections)
projections = array(ptab.getNode("/projections"),dtype=float32)

from scipy.ndimage.interpolation import zoom
#projections = zoom(projections,[1.0,0.25,0.25])


s = shape(projections)
print s


#projections = projections[5:,...]
#projections = log(projections[5:,...])
#projections[isnan(projections)] = 0
import numpy

from pylab import *
imshow2(projections[100,:,:],cmap="gray")

s = shape(projections)
nproj = s[0]

from scipy.ndimage.filters import sobel


bsignal = sum(projections[:,60:75,40:55],axis=(1,2))

bsignal /= amax(bsignal)
bsignal -= amin(bsignal)
figure()
imshow2(projections[10,:,:],cmap="gray")
#dproj = transpose(sum(projections[:,1:,:]-projections[:,:-1,:],2))

#diffproj[abs(diffproj) < 1.5] = 0
#diffproj[diffproj >= 0.8] = 1
print "Creating Amsterdam shroud"
diffproj = (sobel(projections,axis=1))
dproj = transpose(sum(diffproj,axis=2))
print "Done"
figure()
imshow2(diffproj[10,...],cmap="gray")
figure()

dproj2 =dproj
dproj2 -= amin(dproj2)
dproj2 /= amax(dproj2)
imshow2(array(dproj2),cmap="gray")
#background = l1denoise(dproj2,Lambda=0.2,it=500)
#dproj2 -= background
dproj2 = adaptive_filt(dproj2)
dproj2 = dproj2[:310,:]
dproj2 = l1denoise(dproj2,Lambda=1.0,it=500)

#figure()
#imshow2(background,cmap="gray")
figure()
imshow2(dproj2,cmap="gray")

#dproj2 = background



        
signal = pca_extract_signal(dproj2,W=args.W)
signal2 = extract_signal(dproj2)
save("signal.npy",signal)
save("signal2.npy",signal)
figure()
plot(signal)
plot(signal2)

##Binning

signal -= amin(signal)
signal /= amax(signal)
signal -= mean(signal)
x = linspace(0,len(signal),len(signal))

nbins =10 

binning = zeros([len(signal)],dtype=int)

binning = array((signal*(nbins/2+2)),dtype=int)
binning[binning == amax(binning)] = nbins-1

previous = binning[0]
import seaborn as sb
markers = ["o","v","^","<",">","1","2","3","4","*"]
# k = 0
# while(binning[k] == previous):
#     k += 1
# if (binning[k] > previous):
#     inhale = True
# else:
#     inhale = False
#
# for i in range(len(binning)):
#
#     if (binning[i] != previous):
#         if (binning[i] > previous):
#             inhale = True
#         else:
#             inhale = False
#         previous = binning[i]
#     if (not inhale) and binning[i] != (nbins-1) and binning[i] != 0:
#         binning[i] += nbins/2-2
#
#
#

#
# c = sb.color_palette("RdYlBu",10)
# print c
# #plt.rcParams["axes.color_cycle"]=c
# #sb.set_palette(c)
# #with plt.style.context("ggplot"):
#
# with sb.color_palette("RdBu_r",10):
#     figure()
#     #sb.set_palette(c)
#     plot(signal)
#     for i in range(nbins):
#         plot(x[binning==i],signal[binning==i],markers[i],label="Bin " +str(i))
#     print binning
#     legend()
#
# figure()
# sb.countplot(binning)
# save_binning(binning,nbins,"testbinning.hdf5")


from scipy.interpolate import UnivariateSpline
from scipy.signal import wiener,argrelmax,argrelmin,periodogram,welch,savgol_filter

#signal2 = wiener(signal,12,0.1)
#sapprox = UnivariateSpline(x,signal2,s=0.05,k=4)

#der= sapprox.derivative()
#roots = der.roots()


#if (signal[0] < sapprox(roots[0])):
#    inhale_peaks = roots[::2]
#else:
#    inhale_peaks = roots[1::2]
osignal = signal
#signal = savgol_filter(signal,81,2)
signal = wiener(signal,mysize=10)
#signal = clean_signal(signal)
order =15 
inhale_peaks = ravel(array(argrelmax(signal,order=order)))
exhale_peaks = ravel(array(argrelmin(signal,order=order)))
figure()
plot(osignal)
#plot(wsignal)
plot(signal)
plot(inhale_peaks,signal[inhale_peaks],"bo")
plot(exhale_peaks,signal[exhale_peaks],"ro")
plot(bsignal,"g")

figure()
imshow(dproj2,cmap="gray")
plot(signal*200+450,linewidth=5)
"""
k_in=0
k_out =0

if (inhale_peaks[0] > exhale_peaks[0]):
    peak2 = exhale_peaks[0]
    peak1 = 2*inhale_peaks[0]-inhale_peaks[1]
    inhale = False

else:
    peak2 = inhale_peaks[0]
    peak1 = 2*exhale_peaks[0]-exhale_peaks[1]
    inhale = True

for i in range(len(binning)):
    if (i > peak2):
        peak1 = peak2
        if inhale:
            inhale = False
            k_in += 1
            if (k_out == len(exhale_peaks)):
                peak2 = 2*exhale_peaks[-1]-exhale_peaks[-2]
            else:
                peak2 = exhale_peaks[k_out]
        else:
            inhale = True
            k_out += 1
            if (k_in == len(inhale_peaks)):
                peak2 = 2*inhale_peaks[-1]-inhale_peaks[-2]
            else:
                peak2 = inhale_peaks[k_in]


    binning[i] = floor(nbins/2*(i-peak1)/float(peak2-peak1+1))+nbins/2*inhale
"""
k = 0

binning = zeros([len(signal)],dtype=int)
inhale_peaks2= zeros(len(inhale_peaks)+2)
inhale_peaks2[:-2] = inhale_peaks  
inhale_peaks2[-2] = 2*inhale_peaks[-1]-inhale_peaks[-2]
inhale_peaks2[-1] = 2*inhale_peaks[0]-inhale_peaks[1]
inhale_peaks= inhale_peaks2
for i in range(len(binning)):
    if (i > inhale_peaks[k]):
        k+=1
        if (k >= len(inhale_peaks)):
            k = len(inhale_peaks)-1

    binning[i] = (2*nbins*(i-inhale_peaks[k-1])/(inhale_peaks[k]-inhale_peaks[k-1]))
print binning

for i in range(len(binning)):
    binning[i] = (binning[i]+1)/2
binning[binning==nbins] = 0
#with plt.style.context("ggplot"):
with sb.color_palette("RdYlBu",10):
    figure()
    
    #sb.set_palette(c)
    plot(signal)
    for i in range(nbins):
        plot(x[binning==i],signal[binning==i],markers[i],label="Bin " +str(i))
    legend()    
figure()    
sb.countplot(binning)

#excludes = range(1,len(binning),2)
excludes = []

save_binning(binning,nbins,args.output)



show()



#from pylab import imshow,show,figure
#imshow(transpose(dproj),cmap="gray")
    #figure()
#imshow(transpose(dproj2),cmap="gray")
#show()
