import hndreader
import tables
import glob
from pylab import *
import argparse
from scipy.ndimage.filters import uniform_filter,gaussian_filter

parser = argparse.ArgumentParser(description="Converts hnd files to hdf5 files")
parser.add_argument("scandir")
parser.add_argument("--airscan",metavar="a",dest="airscan",help="Directory containing the airscan")

args = parser.parse_args()

files = glob.glob(args.scandir +"/*.hnd")
files.sort()
#
proj,head = hndreader.readHnd(files[0])
print "mAs",head.xrayMA

proj = array(proj,dtype=float64)
projdims = [len(files),head.sizeY,head.sizeX]
#
angles = [head.ctProjectionAngle]
offsetx = [head.IDUPosLat]
offsety = [head.IDUPosLng]
airfiles = []
mAs = head.ctNormChamber

if args.airscan:
    airfiles = glob.glob(args.airscan+"/*.hnd")
    airfiles.sort()
    airscan,airhead = hndreader.readHnd(airfiles[-1])
    airscan = array(airscan,dtype=float64)
    airmAs = airhead.ctNormChamber
    proj /= airscan * mAs / airmAs
else:
    proj /= amax(proj)

batman = proj >= 0
if any(batman):
    b = uniform_filter(proj, size=5)
    proj[batman] = b[batman]

proj = -log(proj)
projections = zeros(projdims,dtype=float32)
projections[0,:,:] = proj
print "Max",amax(proj),"Mean",mean(proj),"Min",amin(proj)
for i in range(1,len(files)):
    proj,head = hndreader.readHnd(files[i])
    proj = array(proj,dtype=float64)

    batman = proj >= 0
    if any(batman):
        b = uniform_filter(proj,size=5)
        proj[batman] = b[batman]

    mAs = head.ctNormChamber
    if args.airscan:
        proj /= airscan * mAs / airmAs
    else:
        proj /= amax(proj)
    proj = -log(proj)
    proj[proj < 0] = 0

    print "Max",amax(proj),"Mean",mean(proj),"Min",amin(proj)
    projections[i,:,:]=proj
    angles.append(head.ctProjectionAngle)
    offsetx.append(head.IDUPosLat)

    offsety.append(head.IDUPosLng)
    print angles[-1],offsetx[-1],offsety[-1]
outfile = "projections.hdf5"

#projections /= amax(projections)
#projections = -log(projections)
f = tables.openFile(outfile,"w")
root = f.getNode("/")
f.createArray(root,"SAD",array([head.SAD*10],dtype=float32))
f.createArray(root,"SDD",array([(head.SAD-head.IDUPosVrt)*10],dtype=float32))
f.createArray(root,"angles",array(angles,dtype=float32))
f.createArray(root,"geometry_dataformat_version",array([2],dtype=int32))
f.createArray(root,"offsetx",array(offsetx,dtype=float32))
f.createArray(root,"offsety",array(offsety,dtype=float32))
#
f.createArray(root,"projection_dataformat_version",array([1],dtype=int32))

f.createArray(root,"projections",projections)
f.createArray(root,"spacing",array([head.IDUResolutionX,head.IDUResolutionY],dtype=float32))
