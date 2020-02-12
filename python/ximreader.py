
import tables
import glob
from pylab import *
import argparse
from imshow2 import imshow2
import varian_utils
import pyximport; pyximport.install()
import ximImageReader


parser = argparse.ArgumentParser(description="Converts hnd files to hdf5 files")
parser.add_argument("scandir")
parser.add_argument("--airscan",metavar="a",dest="airscan",help="Directory containing the airscan")
parser.add_argument("--output",metavar="f",dest="output",help="Output filename",default="projections.hdf5")
parser.add_argument("--skip",metavar="s",dest="skip",type=int,default=1,help="Step between projections (2 would skip every second)")

parser.add_argument("--scatter_calibration",dest="scattercalibration",help="XML file containing the scatter calibration")
args = parser.parse_args()

files = glob.glob(args.scandir +"/*.xim")
files.sort()
skip = args.skip
files = files[::skip]
#
proj,head = ximImageReader.readXim(files[0])
proj[proj == 0] = 1
print head
proj = array(proj,dtype=float64)
projdims = [len(files),proj.shape[0],proj.shape[1]]


if args.scattercalibration:
    scattercalibration = varian_utils.parse_scatter(args.scattercalibration)

#
angles = [head["GantryRtn"]]
offsetx = [head["KVDetectorVrt"]]
offsety = [head["KVDetectorLat"]]
airfiles = []

#mAs = head["KVMilliAmperes"]*head["KVMilliSeconds"]
mAs = head["KVNormChamber"]
print "Proj",amin(proj),amax(proj)
if args.airscan:
    airscan,airhead = ximImageReader.readXim(args.airscan)
    airscan[airscan == 0] = 1
    airscan = array(airscan,dtype=float64)
    #airmAs = airhead["KVMilliAmperes"]*airhead["KVMilliSeconds"]
    airmAs = airhead["KVNormChamber"]
    print "Airscan",amin(airscan),amax(airscan)

    imshow2(airscan,cmap="gray")
    figure()
    proj2 = array(proj)
    imshow2(proj2,cmap="gray")

    if (args.scattercalibration):
        proj = varian_utils.adaptive_correct_scatter(scattercalibration,airscan*mAs/airmAs,proj,head["PixelHeight"]*10)


    proj /= airscan*mAs/airmAs
else:
    proj /= amax(proj)
proj = -log(proj)
figure()
imshow2(proj,cmap="gray")
show()

projections = zeros(projdims,dtype=float32)
projections[0,:,:] = proj
print "Max",amax(proj),"Mean",mean(proj),"Min",amin(proj)
for i in range(1,len(files)):

    proj,head = ximImageReader.readXim(files[i])
    if (proj == None):
        break
    proj[proj == 0] = 1
    proj = array(proj,dtype=float64)


    #mAs = head["KVMilliAmperes"]*head["KVMilliSeconds"]
    mAs = head["KVNormChamber"]
    print "mAs",mAs,airmAs
    if args.airscan:
        if (args.scattercalibration):
            proj = varian_utils.adaptive_correct_scatter(scattercalibration, airscan * mAs / airmAs, proj,head["PixelHeight"]*10)
        proj /= airscan*mAs/airmAs
    else:
        proj /= amax(proj)
    proj = -log(proj)

    print "Max",amax(proj),"Mean",mean(proj),"Min",amin(proj),files[i]
    projections[i,:,:]=proj
    angles.append(-head["GantryRtn"])
    offsetx.append(head["KVDetectorLat"]*10)
    offsety.append(head["KVDetectorLng"]*10)
    print angles[-1],offsetx[-1],offsety[-1]


#projections /= amax(projections)
#projections = -log(projections)
f = tables.openFile(args.output,"w")
root = f.getNode("/")
f.createArray(root,"SAD",array([head["KVSourceVrt"]*10],dtype=float32))
f.createArray(root,"SDD",array([(head["KVSourceVrt"]-head["KVDetectorVrt"])*10],dtype=float32))
f.createArray(root,"angles",array(angles,dtype=float32))
f.createArray(root,"geometry_dataformat_version",array([2],dtype=int32))
f.createArray(root,"offsetx",array(offsetx,dtype=float32))
f.createArray(root,"offsety",array(offsety,dtype=float32))
#
f.createArray(root,"projection_dataformat_version",array([1],dtype=int32))

f.createArray(root,"projections",projections)
f.createArray(root,"spacing",array([head["PixelHeight"]*10,head["PixelWidth"]*10],dtype=float32))
