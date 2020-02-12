# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 07:28:44 2015

@author: David
"""
from __future__ import division, print_function
from numpy import *
import tables
from sklearn.decomposition import PCA
from scipy.ndimage.filters import gaussian_filter, uniform_filter,median_filter
import scipy.signal as spsignal
from skimage.restoration import denoise_nl_means
import subprocess
import tempfile
from readReal import saveReal,readReal
import shutil
def updateF(p, alpha, sigma):
    pabs = sqrt(p[:, :, 0] ** 2 + p[:, :, 1] ** 2)
    return p / (1 + alpha * sigma) / maximum(1, pabs[..., newaxis] / (1 + alpha * sigma))


def updateF2(p, alpha, sigma):
    return p / (1 + alpha * sigma) / maximum(1, abs(p / (1 + alpha * sigma)))


# def updateG(u,g,tau,L):
#    return (u+tau*L*g)/(1+tau*L)

def updateG(u, g, tau, L):
    res = zeros(shape(u))
    res[u - g > tau * L] = u[u - g > tau * L] - tau * L
    res[u - g < tau * L] = u[u - g < tau * L] + tau * L
    res[abs(u - g) <= tau * L] = g[abs(u - g) <= tau * L]
    return res


def gradient_x(v):
    """Calculates the x component of the gradient transform of v"""
    res = zeros(shape(v), dtype=float64)
    res[:-1, :] = v[1:, :] - v[:-1, :]

    return res


def gradient_x_H(v):
    """ Calculates the transpose of the x component of the gradient transform of v"""
    res = zeros(shape(v), dtype=float64)
    res[0, :] = -v[0, :]
    res[1:-1, :] = v[:-2, :] - v[1:-1, :]
    res[-1, :] = v[-2, :]

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
    s = shape(v)

    s2 = (s[0], s[1], 2)
    res = zeros(s2)
    res[:-1, :, 0] = v[1:, :] - v[:-1, :]
    res[:, :-1, 1] = v[:, 1:] - v[:, :-1]
    return res


def grad_H(v):
    return gradient_x_H(v[:, :, 0]) + gradient_y_H(v[:, :, 1])

def poisson_denoise(img,noise=1):

    path = tempfile.mkdtemp()

    img_file = path + "/img.real"
    output_file = path + "/denoised.real"
    saveReal(img,img_file)
    print(["/home/dch/gt-tomography/build/RelWithDebInfo/denoise_NLM2DPoisson", "--input",img_file,"--output",output_file,"--noise",str(noise)])

    completed = subprocess.run(["/home/dch/gt-tomography/build/RelWithDebInfo/denoise_NLM2DPoisson", "--input",img_file,"--output",output_file,"--noise",str(noise)],check=True)
    print(output_file)
    result = readReal(output_file)
    shutil.rmtree(path)
    return result


def l1denoise(x0, Lambda=1, it=100):
    x = x0

    L = 4

    xn = x
    xold = x
    tau = 0.25
    theta = 0.5
    gamma = 0.35 * Lambda
    sigma = 0.1 / (L * tau)
    # sigma = 1
    y = grad(x)
    xold = array(x)
    for i in range(it):
        gx = grad(x)
        print(sum((x - x0) ** 2) * Lambda + sum(sqrt(sum(gx ** 2, axis=2))))
        y = updateF(y + sigma * gx, 0, sigma)
        xn = updateG(xn - tau * grad_H(y), x0, tau, Lambda)
        x = xn + theta * (xn - xold)
        xold = xn
    return x


def adaptive_filt(image):
    """Based on 'Robust breathing signal extraction from cone beam CT projections based on adaptive and global optimization techniques'
    Chao et al, PMB 2016"""
    ig = gaussian_filter(image, sigma=10)
    mean_image = uniform_filter(image, size=10)
    istd = sqrt(uniform_filter((mean_image - image) ** 2, size=10))
    im = mean(istd)
    return (image - ig) / (istd + im) + mean(image)


def save_binning(binning, nbins, filename, exclude=[]):
    binning2 = array(binning)
    for proj in exclude:
        binning2[proj] = nbins + 1  # Exclude some projections completely

    f = tables.open_file(filename, mode="w")
    root = f.get_node("/")
    index = array(range(len(binning2)), dtype=uint32)
    for i in range(nbins):
        arr = f.create_array(root, "bin_" + str(i + 1), array(index[binning2 == i], dtype=uint32))
    f.create_array(root, "numBins", array([nbins], dtype=uint32))
    f.create_array(root, "binning_dataformat_version", array([1], dtype=uint32))

    f.close()




def pca_extract_signal(shroud, W=40):
    n_components = 5
    pca = PCA(n_components=n_components)
    pca.fit(transpose(shroud))
    primary = pca.components_[0, :]
    figure()
    plot(primary, label="primary")
    plot(pca.components_[1, :], label="secondary")
    legend()

    window = transpose(shroud[:, :W])
    pca.fit(window)
    primary = pca.components_[0, :]

    print(shape(primary), norm(primary))

    signal = []

    for k in range(W // 2):
        signal.append(dot(window[k, :], primary) / norm(window[k, :]))

    for i in range(1, nproj - W):
        window = transpose(dproj2[:, i:i + W])
        pca.fit(window)
        V = pca.components_;
        cc = abs(dot(primary, V[0, :]))
        m = 0
        for k in range(1, n_components):
            c2 = abs(dot(primary, V[k, :]))
            if (c2 > cc):
                cc = c2
                m = k
        m = 0
        if dot(primary, V[m, :]) > 0:
            primary = V[m, :]
        else:
            primary = -V[m, :]
        # primary = V[m,:]/norm(V[m,:])
        frame = window[W // 2, :] / norm(window[W // 2, :])
        signal.append(dot(frame, primary))

    for k in range(nproj - W // 2, nproj):
        signal.append(dot(dproj2[:, k], primary) / norm(dproj2[:, k]))

    signal = array(signal)
    return signal

def create_binning2(signal,nbins):
    from scipy.signal import argrelmax,argrelmin
    import seaborn as sb

    x = linspace(0, len(signal), len(signal))
    order = 15
    inhale_peaks = ravel(array(argrelmax(signal, order=order)))
    exhale_peaks = ravel(array(argrelmin(signal, order=order)))
    figure()
    plot(signal)
    # plot(wsignal)
    plot(signal)
    plot(inhale_peaks, signal[inhale_peaks], "bo")
    plot(exhale_peaks, signal[exhale_peaks], "ro")

    show()
    inhale_peaks = np.append(inhale_peaks, [2*inhale_peaks[-1]-inhale_peaks[-2], 2*inhale_peaks[0]-inhale_peaks[1]])
    exhale_peaks = np.append(exhale_peaks,[2*exhale_peaks[-1]-exhale_peaks[-2], 2*exhale_peaks[0]-exhale_peaks[1]])
    binningf = zeros(len(signal))
    ki = 0
    ke = 0
    for i in range(len(signal)):
        if i > inhale_peaks[ki]:
            ki += 1
        if i > exhale_peaks[ke]:
            ke += 1
        if (inhale_peaks[ki] < exhale_peaks[ke]):
            binningf[i] = (inhale_peaks[ki]-i)/(inhale_peaks[ki]-exhale_peaks[ke-1])*0.5
        else:
            binningf[i] = (exhale_peaks[ke]-i)/(exhale_peaks[ke]-inhale_peaks[ki-1])*0.5+0.5


    binningf = (binningf+1.0/nbins)%1
    plot(binningf)

    binning = np.array(binningf*nbins,dtype=int)
    # with plt.style.context("ggplot"):
    with sb.color_palette("RdYlBu", nbins):
        figure()

        # sb.set_palette(c)
        plot(signal)
        for i in range(nbins):
            plot(x[binning == i], signal[binning == i], "o", label="Bin " + str(i))
        legend()
    figure()
    sb.countplot(binning)
    return binning

def create_binning(signal,nbins):
    from scipy.signal import argrelmax,argrelmin
    import seaborn as sb

    x = linspace(0, len(signal), len(signal))
    order = 15
    inhale_peaks = ravel(array(argrelmax(signal, order=order)))
    exhale_peaks = ravel(array(argrelmin(signal, order=order)))
    figure()
    plot(signal)
    # plot(wsignal)
    plot(signal)
    plot(inhale_peaks, signal[inhale_peaks], "bo")
    plot(exhale_peaks, signal[exhale_peaks], "ro")
    plot(bsignal, "g")
    if args.curve_path is not None:
        save(args.curve_path,signal)
    # fig = figure()
    # imshow(dproj2, cmap="gray")
    # plot(signal * 200 + 350, linewidth=5)
    # if args.debug_image is not None:
    #     fig.savefig(args.debug_image)
    k = 0

    binning = zeros([len(signal)], dtype=int)
    inhale_peaks2 = zeros(len(inhale_peaks) + 2)
    inhale_peaks2[:-2] = inhale_peaks
    inhale_peaks2[-2] = 2 * inhale_peaks[-1] - inhale_peaks[-2]
    inhale_peaks2[-1] = 2 * inhale_peaks[0] - inhale_peaks[1]
    inhale_peaks = inhale_peaks2
    for i in range(len(binning)):
        if (i > inhale_peaks[k]):
            k += 1
            if (k >= len(inhale_peaks)):
                k = len(inhale_peaks) - 1

        binning[i] = (2 * nbins * (i - inhale_peaks[k - 1]) / (inhale_peaks[k] - inhale_peaks[k - 1]))
    print(binning)

    for i in range(len(binning)):
        binning[i] = (binning[i] + 1) / 2
    binning[binning == nbins] = 0
    # with plt.style.context("ggplot"):
    with sb.color_palette("RdYlBu", nbins):
        figure()

        # sb.set_palette(c)
        plot(signal)
        for i in range(nbins):
            plot(x[binning == i], signal[binning == i], "o", label="Bin " + str(i))
        legend()
    figure()
    sb.countplot(binning)
    return binning

    # excludes = range(1,len(binning),2)


def clean_signal(signal, nvals=10):
    from scipy.fftpack import rfft, rfftfreq, irfft
    signal2 = signal - mean(signal)
    fsignal = rfft(signal2)
    pabs = abs(fsignal)
    freq = rfftfreq(len(signal), d=60.0 / len(signal))
    fLow = array(pabs)
    fLow[freq > 0.15] = 0
    fHigh = array(pabs)
    fHigh[freq < 0.15] = 0

    output = zeros_like(fsignal)
    fLowS = sort(fLow)
    output[fLow > fLowS[-nvals]] = fsignal[fLow > fLowS[-nvals]]
    fHighS = sort(fHigh)
    output[fHigh > fHighS[-nvals]] = fsignal[fHigh > fHighS[-nvals]]

    figure()
    plot(freq, pabs)
    return irfft(output)

def find_move(v1,v2):
    return np.argmax(spsignal.correlate(v1,v2))
#
# def find_move(v1, v2, L=15,metric=signal.correlate):
#     v1t = v1[L:-L]
#     # cur_min = norm(v1 - v2, ord=1)
#     cur_max = metric(v1,v2)
#     b = 0
#     for k in range(-L, L + 1):
#         vl2 = np.roll(v2, k, axis=0)[L:-L]
#
#         knorm = metric(v1,vl2)
#         if (knorm > cur_max):
#             b = -k
#             cur_max = knorm
#     return b


def extract_signal(shroud):

    # fshroud = fftpack.fft(shroud,axis=0)
    b = zeros(shroud.shape[1])
    size = shroud.shape[0]
    print(shroud.shape)
    for i in range(1, shroud.shape[1]-10):
        # blocal = np.argmax(np.abs(fftpack.ifft(-np.conjugate(fshroud[:,i])*fshroud[:,i-1])))
        blocal = np.argmax(spsignal.correlate(shroud[:,i-1],shroud[10:-10,i+10],mode="valid"))-10
        print(blocal)
        b[i] = b[i - 1] + blocal
    return b


import matplotlib.pyplot as plt
import glob
import time

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("projections")
parser.add_argument("--output", metavar="f", dest="output", help="Output filename", default="binning.hdf5")
parser.add_argument("--nbins", metavar="n", dest="nbins", help="Number of bins", default=10, type=int)
parser.add_argument("--breathing_cycle","-W", dest="W", help="Approximate number of projections per breathing cycle",
                    type=int,
                    default=40)
parser.add_argument("--silent",action='count')
parser.add_argument("--curve_path",help="Location to store breathing curve")
parser.add_argument("--debug_image",help="Save the Amsterdam shroud with the overlaid breathing curve at the described "
                                         "location. Mainly useful for debugging")


args = parser.parse_args()

ptab = tables.open_file(args.projections)
projections = array(ptab.get_node("/projections"), dtype=float32)
# for p in projections:
#     p[:] = median_filter(p,size=7)
# projections = poisson_denoise(projections,noise=2)
# projections = median_filter(projections,size=(1,7,7))
from scipy.ndimage.interpolation import zoom

# projections = zoom(projections,[1.0,0.25,0.25])


s = shape(projections)
print(s)

# projections = projections[5:,...]
# projections = log(projections[5:,...])
# projections[isnan(projections)] = 0
import numpy

from pylab import *

imshow(projections[100, :, :], cmap="gray")

s = shape(projections)
nproj = s[0]

from scipy.ndimage.filters import sobel

# bsignal = sum(projections[:,60:75,40:55],axis=(1,2))
bsignal = sum(projections, axis=(1, 2))

bsignal = np.concatenate((bsignal,bsignal[::-1]))
bsignal = np.roll(bsignal,nproj//2)

bsignal = spsignal.medfilt(bsignal,kernel_size=5)
gauss = spsignal.gaussian(100,10)
gauss /= np.sum(gauss)
# bsignal = bsignal-spsignal.convolve(bsignal,)
# bsignal = bsignal-spsignal.convolve(bsignal,gauss,mode='same')

gauss = spsignal.gaussian(10,4)
# bsignal = spsignal.convolve(bsignal,gauss,mode='same')
# bartlett = spsignal.bartlett(11)
# bsignal = spsignal.convolve(bsignal,bartlett,mode='same')
figure()
plot(bsignal)
# bsignal = np.imag(spsignal.hilbert(bsignal))
# bsignal = spsignal.detrend(bsignal)

bsignal = np.roll(bsignal,-nproj//2)[:nproj]

bsignal -= mean(bsignal)
bsignal /= std(bsignal)

#

diffproj = (sobel(projections, axis=1))
dproj = transpose(sum(diffproj, axis=2))
dproj = adaptive_filt(dproj)
figure()
imshow(dproj,cmap="gray")
title("Raw AMS")
plot(bsignal*50+400,linewidth=4)
if args.debug_image is not None:
    savefig(args.debug_image)
if args.curve_path is not None:
    save(args.curve_path,bsignal)


binning = create_binning2(bsignal,args.nbins)
save_binning(binning, args.nbins, args.output)
if args.silent is None:
    show()
#
# figure()
# imshow(projections[10, :, :], cmap="gray")
# # dproj = transpose(sum(projections[:,1:,:]-projections[:,:-1,:],2))
#
# # diffproj[abs(diffproj) < 1.5] = 0
# # diffproj[diffproj >= 0.8] = 1
# print("Creating Amsterdam shroud")

# print("Done")
#
# figure()
#
# dproj2 = dproj
# dproj2 -= amin(dproj2)
# dproj2 /= amax(dproj2)
#
# # background = l1denoise(dproj2,Lambda=0.2,it=500)
# # dproj2 -= background
#
# # dproj2 = sobel(dproj2,axis=1)
# dproj2 = adaptive_filt(dproj2)
#
# # dproj2 = denoise_nl_means(dproj2,multichannel=False,h=0.5)
# # dproj2 = median_filter(dproj2,size=(7,7))
# figure()
# imshow(dproj2,cmap="gray")
# title("Denoised NL")
# #dproj2 = dproj2[150:-100, :]
# denoised = l1denoise(dproj2, Lambda=1.0, it=500)
# # figure()
# # imshow(denoised,cmap="gray")
# # title("Denoised")
#
# dproj2 = denoised
#
# # figure()
# # imshow(background,cmap="gray")
# figure()
# imshow(dproj2, cmap="gray")
# title("Exctractred PCA")
#
# # dproj2 = background
#
#
# save("ams.npy",dproj2)
#
# signal = pca_extract_signal(dproj2, W=args.W)
# signal2 = extract_signal(dproj2)
# save("signal.npy", signal)
# save("signal2.npy", signal2)
# figure()
# plot(signal,label="PCA")
# plot(signal2,label="Reg")
# legend()
# ##Binning
#
# signal -= amin(signal)
# signal /= amax(signal)
# signal -= mean(signal)
# x = linspace(0, len(signal), len(signal))
#
# nbins = args.nbins
#
# binning = zeros([len(signal)], dtype=int)
#
# binning = array((signal * (nbins / 2 + 2)), dtype=int)
# binning[binning == amax(binning)] = nbins - 1
#
# previous = binning[0]
# import seaborn as sb
#
# markers = ["o", "v", "^", "<", ">", "1", "2", "3", "4", "*"]
# # k = 0
# # while(binning[k] == previous):
# #     k += 1
# # if (binning[k] > previous):
# #     inhale = True
# # else:
# #     inhale = False
# #
# # for i in range(len(binning)):
# #
# #     if (binning[i] != previous):
# #         if (binning[i] > previous):
# #             inhale = True
# #         else:
# #             inhale = False
# #         previous = binning[i]
# #     if (not inhale) and binning[i] != (nbins-1) and binning[i] != 0:
# #         binning[i] += nbins/2-2
# #
# #
# #
#
# #
# # c = sb.color_palette("RdYlBu",10)
# # print c
# # #plt.rcParams["axes.color_cycle"]=c
# # #sb.set_palette(c)
# # #with plt.style.context("ggplot"):
# #
# # with sb.color_palette("RdBu_r",10):
# #     figure()
# #     #sb.set_palette(c)
# #     plot(signal)
# #     for i in range(nbins):
# #         plot(x[binning==i],signal[binning==i],markers[i],label="Bin " +str(i))
# #     print binning
# #     legend()
# #
# # figure()
# # sb.countplot(binning)
# # save_binning(binning,nbins,"testbinning.hdf5")
#
#
#
# from scipy.interpolate import UnivariateSpline
# from scipy.signal import wiener, argrelmax, argrelmin, periodogram, welch, savgol_filter
#
# # signal2 = wiener(signal,12,0.1)
# # sapprox = UnivariateSpline(x,signal2,s=0.05,k=4)
#
# # der= sapprox.derivative()
# # roots = der.roots()
#
#
# # if (signal[0] < sapprox(roots[0])):
# #    inhale_peaks = roots[::2]
# # else:
# #    inhale_peaks = roots[1::2]
# osignal = signal
# # signal = savgol_filter(signal,81,2)
# signal = wiener(signal, mysize=20)
# # signal = clean_signal(signal)
#
# """
# k_in=0
# k_out =0
#
# if (inhale_peaks[0] > exhale_peaks[0]):
#     peak2 = exhale_peaks[0]
#     peak1 = 2*inhale_peaks[0]-inhale_peaks[1]
#     inhale = False
#
# else:
#     peak2 = inhale_peaks[0]
#     peak1 = 2*exhale_peaks[0]-exhale_peaks[1]
#     inhale = True
#
# for i in range(len(binning)):
#     if (i > peak2):
#         peak1 = peak2
#         if inhale:
#             inhale = False
#             k_in += 1
#             if (k_out == len(exhale_peaks)):
#                 peak2 = 2*exhale_peaks[-1]-exhale_peaks[-2]
#             else:
#                 peak2 = exhale_peaks[k_out]
#         else:
#             inhale = True
#             k_out += 1
#             if (k_in == len(inhale_peaks)):
#                 peak2 = 2*inhale_peaks[-1]-inhale_peaks[-2]
#             else:
#                 peak2 = inhale_peaks[k_in]
#
#
#     binning[i] = floor(nbins/2*(i-peak1)/float(peak2-peak1+1))+nbins/2*inhale
# """
# k = 0
#
# binning = zeros([len(signal)], dtype=int)
# inhale_peaks2 = zeros(len(inhale_peaks) + 2)
# inhale_peaks2[:-2] = inhale_peaks
# inhale_peaks2[-2] = 2 * inhale_peaks[-1] - inhale_peaks[-2]
# inhale_peaks2[-1] = 2 * inhale_peaks[0] - inhale_peaks[1]
# inhale_peaks = inhale_peaks2
# for i in range(len(binning)):
#     if (i > inhale_peaks[k]):
#         k += 1
#         if (k >= len(inhale_peaks)):
#             k = len(inhale_peaks) - 1
#
#     binning[i] = (2 * nbins * (i - inhale_peaks[k - 1]) / (inhale_peaks[k] - inhale_peaks[k - 1]))
# print(binning)
#
# for i in range(len(binning)):
#     binning[i] = (binning[i] + 1) / 2
# binning[binning == nbins] = 0
# # with plt.style.context("ggplot"):
# with sb.color_palette("RdYlBu", nbins):
#     figure()
#
#     # sb.set_palette(c)
#     plot(signal)
#     for i in range(nbins):
#         plot(x[binning == i], signal[binning == i], "o", label="Bin " + str(i))
#     legend()
# figure()
# sb.countplot(binning)
#
# # excludes = range(1,len(binning),2)
# excludes = []
#
# save_binning(binning, nbins, args.output)
# if args.silent is None:
#     show()



# from pylab import imshow,show,figure
# imshow(transpose(dproj),cmap="gray")
# figure()
# imshow(transpose(dproj2),cmap="gray")
# show()
