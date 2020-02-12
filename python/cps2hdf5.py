import hndreader
import tables
import glob
from pylab import *
import argparse
from imshow2 import imshow2

parser = argparse.ArgumentParser(description="Converts hnd files to hdf5 files")
parser.add_argument("cps_file")
parser.add_argument("--airscan",metavar="a",dest="airscan",help="Directory containing the airscan")
args = parser.parse_args()

#
projs,heads = hndreader.readCps(args.cps_file)
print "mAs",heads[0].xrayMA
projs = array(projs,dtype=float32)
#
angles = []
offsetx = []
offsety = []
for head in heads:
    angles.append(head.ctProjectionAngle)
    offsetx.append(head.IDUPosLat)
    offsety.append(head.IDUPosLng)
airfiles = []
if args.airscan:
    airfiles = glob.glob(args.airscan+"/*.hnd")
    airfiles.sort()

    airscan,_ = hndreader.readHnd(airfiles[0])
    airscan = array(airscan,dtype=float32)
    for airfile in airfiles[1:]:
        airscanTmp,_ = hndreader.readHnd(airfile)
        airscan += airscanTmp
    airscan /= len(airfiles)
if len(airfiles) > 0:
    projs /= airscan
else:
    projs /= amax(projs)

projs = -log(projs)
print "Projs contains NaN:",any(isnan(projs)),"Infs:",any(isinf(projs))
projs[isinf(projs)] = 0
outfile = "projections.hdf5"

head = heads[0]
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

f.createArray(root,"projections",projs)
f.createArray(root,"spacing",array([head.IDUResolutionX,head.IDUResolutionY],dtype=float32))
