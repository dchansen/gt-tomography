import numpy
import struct
from collections import namedtuple
import pyximport; pyximport.install()
import hndImageReader

def readHnd(filename):
    totbytes = 120+10*4+41*8
    HndHeader = namedtuple("HndHeader","filetype filelength checksumSpec checksum creationDate creationTime patientID patientSer seriesID seriesSer "\
            "sliceID sliceSer sizeX sizeY sliceZPos modality window level pixelOffset imageType gantryRotation SAD SFD collX1 collX2 collY1 collY2 "\
            "collRotation filedX filedY bladeX1 bladeX2 bladeY1 bladeY2 IDUPosLng IDUPosLat IDUPosVrt IDUPosRtn patientSupportAngle tableTopEccentricAngle "\
            "couchVrt couchLng couchLat IDUResolutionX IDUResolutionY imageResolutionX imageResolutionY energy doseRate xRayKV xrayMA metersetExposure " \
            "acqAdjustment ctProjectionAngle ctNormChamber gatingTimeTag gating4DInfoX gating4DInfoY gating4DInfoZ gating4DInfoTime")
    with open(filename,"rb") as f:
        data = f.read(totbytes)
        formatstring = "=32sI4sI8s8s16sI16sI16sIIId16sIII4sdddddddddddddddddddddddddddddddddddddddd"
        header = HndHeader._make(struct.unpack(formatstring,data))
    image = hndImageReader.readHndImage(filename,[header.sizeX,header.sizeY])

    return image,header    

def readCps(filename):
    results = hndImageReader.readCpsImage(filename)

    return results    
