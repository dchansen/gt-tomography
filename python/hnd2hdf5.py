import hndreader
import tables
import glob
from pylab import *
import argparse
from scipy.ndimage.filters import uniform_filter,gaussian_filter
import subprocess
import tempfile
import shutil
from readReal import saveReal,readReal
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
parser = argparse.ArgumentParser(description="Converts hnd files to hdf5 files")
parser.add_argument("scandir")
parser.add_argument("--airscan",metavar="a",dest="airscan",help="Directory containing the airscan")
parser.add_argument("--output",dest="output")
parser.add_argument("--denoise",type=float)
args = parser.parse_args()

files = glob.glob(args.scandir +"/*.hnd")
files.sort()
print("Files", files)
#
proj,head = hndreader.readHnd(files[0])
print("mAs",head.xrayMA)
cutoff = 8

proj = array(proj,dtype=float64)[cutoff:-cutoff,cutoff:-cutoff]
proj[proj == 0] = 1
projdims = [len(files),proj.shape[0],proj.shape[1]]
#
angles = [head.ctProjectionAngle]
offsetx = [head.IDUPosLat]
offsety = [head.IDUPosLng]
airfiles = []
mAs = [head.ctNormChamber]


if args.airscan:
    airfiles = glob.glob(args.airscan+"/*.hnd")
    airfiles.sort()
    airscan,airhead = hndreader.readHnd(airfiles[-1])
    airscan = array(airscan,dtype=float64)[cutoff:-cutoff,cutoff:-cutoff]
    airmAs = airhead.ctNormChamber
    # proj /= airscan * mAs / airmAs



# proj = -log(proj)
projections = zeros(projdims,dtype=float32)
projections[0,:,:] = proj
print("Max",amax(proj),"Mean",mean(proj),"Min",amin(proj))

for i in range(1,len(files)):
    proj,head = hndreader.readHnd(files[i])
    proj = array(proj,dtype=float32)[cutoff:-cutoff,cutoff:-cutoff]

    mAs.append(head.ctNormChamber)
    # if args.airscan:
    #     proj /= airscan * mAs / airmAs
    # else:
    #     proj /= amax(proj)
    # proj = -log(proj)

    #proj[proj < 0] = 0

    # print(i,"Max",amax(proj),"Mean",mean(proj),"Min",amin(proj))



    projections[i,:,:]=proj
    angles.append(head.ctProjectionAngle)
    offsetx.append(head.IDUPosLat)

    offsety.append(head.IDUPosLng)

if (args.denoise is not  None):
    projections = poisson_denoise(projections,args.denoise)

projections[projections == 0] = 1

projections /= airscan
projections /= array(mAs)[:,np.newaxis,np.newaxis]/airmAs
projections = -log(projections)

print(projections.shape)
import scipy.ndimage as nd
from scipy import signal
tmp2 = projections
tmp = np.sum(tmp2,axis=(2,1))

tmp /= np.std(tmp)
tmp -= signal.medfilt(tmp,7) 
tmp = np.abs(tmp)


toKeep = tmp < 0.05
projections = projections[toKeep]


offsetx = np.array(offsetx)
offsety = np.array(offsety)
angles = np.array(angles)

offsetx = offsetx[toKeep]
offsety = offsety[toKeep]
angles = angles[toKeep]              


outfile = args.output

#projections /= amax(projections)
#projections = -log(projections)
f = tables.open_file(outfile,"w")
root = f.get_node("/")
f.create_array(root,"SAD",array([head.SAD*10],dtype=float32))
f.create_array(root,"SDD",array([(head.SAD-head.IDUPosVrt)*10],dtype=float32))
f.create_array(root,"angles",array(angles,dtype=float32))
f.create_array(root,"geometry_dataformat_version",array([2],dtype=int32))
f.create_array(root,"offsetx",array(offsetx,dtype=float32))
f.create_array(root,"offsety",array(offsety,dtype=float32))
#
f.create_array(root,"projection_dataformat_version",array([1],dtype=int32))

f.create_array(root,"projections",projections)
f.create_array(root,"spacing",array([head.IDUResolutionX,head.IDUResolutionY],dtype=float32))
