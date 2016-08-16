
import xml.etree.ElementTree as ET
import collections
import numpy as np
from scipy.signal import fftconvolve
from scipy.ndimage.interpolation import  zoom
import pylab
def parse_scatter(scatter_file):
    ns = {'varian': 'http://baden.varian.com/cr.xsd'}
    scatter = collections.namedtuple("ScatterModel", "Thickness A alpha beta sigma1 B sigma2 gamma MagFactor")
    tree = ET.parse(scatter_file)

    root = tree.getroot()

    scatter_models = root.find("varian:CalibrationResults",ns).find("varian:ObjectScatterModels",ns)

    scatterers = []
    for scatter_model in scatter_models.findall("varian:ObjectScatterModel",ns):

        thickness =  float(scatter_model.find('varian:Thickness',ns).text)
        OSF = scatter_model.find("varian:ObjectScatterFit",ns)

        A = float(OSF.find('varian:A', ns).text)
        B = float(OSF.find('varian:B', ns).text)
        alpha = float(OSF.find('varian:alpha', ns).text)
        beta = float(OSF.find('varian:beta', ns).text)
        sigma1 = float(OSF.find('varian:sigma1', ns).text)
        sigma2 = float(OSF.find('varian:sigma2', ns).text)
        gamma = float(OSF.find('varian:gamma', ns).text)
        MagFactor = float(OSF.find('varian:MagFactor', ns).text)


        scatterers.append(scatter(thickness,A,alpha,beta,sigma1,B,sigma2,gamma,MagFactor))

    return scatterers


def calculate_scatter(scatterModel,I0,IP,thickness,mask):
    logI0IP = np.log(I0 / IP)
    logI0IP[logI0IP < 0] = 0
    amplitude_factor = scatterModel.A*(IP/I0)**scatterModel.alpha*logI0IP**scatterModel.beta*mask


    s= np.shape(I0)
    xx,yy = np.mgrid[:s[0],:s[1]]
    xx -= s[0]/2
    yy -= s[1]/2

    g = np.exp(-(xx**2+yy**2)/(2*scatterModel.sigma1**2))+scatterModel.B*np.exp(-(xx**2+yy**2)/(2*scatterModel.sigma2**2))

    scatter = (1-scatterModel.gamma*thickness)*fftconvolve(amplitude_factor*IP,g,mode="same")+\
              scatterModel.gamma*fftconvolve(amplitude_factor*thickness*IP,g,mode="same")


    return scatter




def adaptive_correct_scatter(scatterModels,I0,IP,pixelsize):

    s = np.shape(I0)

    new_pixelsize = (4.0,4.0)
    new_imagesize = (10.0,26.0)
    print s,new_imagesize
    I0_small = zoom(I0,new_imagesize)
    IP_small = zoom(IP, new_imagesize)

    scatter_old = np.zeros_like(I0_small)

    mu = 0.02
    step_size = 0.6
    t = np.log(I0_small/IP_small)/mu

    pylab.imshow(t,cmap="viridis")
    pylab.show()

    for k in range(1):
        total_scatter = np.zeros_like(I0_small)

        for i in range(len(scatterModels)-1):
            thickness1 = scatterModels[i].Thickness
            thickness2 = scatterModels[i+1].Thickness
            mask = np.logical_and(t > thickness1, t < thickness2)
            total_scatter += calculate_scatter(scatterModels[i],I0_small,IP_small,t,mask)

        total_scatter += calculate_scatter(scatterModels[i],I0_small,IP_small,t,t > scatterModels[-1].Thickness)


        IP_small += step_size*(scatter_old-total_scatter)
        scatter_old = total_scatter

    s2 = np.shape(I0_small)
    scatter = zoom(scatter_old,(float(s[0])/s2[0],float(s[1])/s2[1]))

    SF = np.minimum(scatter/IP,0.95)
    print "SF",np.shape(SF),np.shape(scatter)
    result = IP*(1.0-SF)

    pylab.imshow(IP, cmap="viridis")
    pylab.figure()
    pylab.imshow(scatter, cmap="viridis")
    pylab.figure()
    pylab.imshow(SF, cmap="viridis")
    pylab.figure()
    pylab.imshow(result,cmap="viridis")
    pylab.show()
    return result

