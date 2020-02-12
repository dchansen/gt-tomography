
import struct
from collections import namedtuple
import numpy as np
cimport numpy as np

from libc.stdio cimport *
cimport cython  

cdef extern from "stdio.h":
    #FILE * fopen ( const char * filename, const char * mode )
    FILE *fopen(const char *, const char *)
    #int fclose ( FILE * stream )
    int fclose(FILE *)
    #ssize_t getline(char **lineptr, size_t *n, FILE *stream);
    size_t fread(void * ptr, size_t, size_t, FILE*)
    int fseek(FILE*, size_t, int)

@cython.boundscheck(False)
def readHndImage(filename,imgSize):
    cdef unsigned int i = 0;
    cdef unsigned int sizeX = imgSize[0]
    cdef unsigned int sizeY = imgSize[1]
    
    cdef np.ndarray[np.uint32_t, ndim=1] buf = np.zeros([sizeX*sizeY], dtype=np.uint32)

    filename_byte = filename.encode("UTF-8")
    cdef char* fname = filename_byte
    cdef FILE* cfile;
    cfile = fopen(fname,"rb")
    fseek(cfile,1024,0)

    cdef np.ndarray[np.uint8_t,ndim=1] pt_lut = np.zeros([sizeX*sizeY], dtype=np.uint8)
    cdef size_t nbytes = (sizeX*(sizeY-1))/4
    fread(&pt_lut[0],1,nbytes,cfile)

    #Read first row
    fread(&buf[0],4,sizeX,cfile)

    #Read first pixel of second row
    fread(&buf[sizeX],4,1,cfile)
    i = sizeX+1
    cdef unsigned int lut_idx=0;
    cdef unsigned int lut_off=0;
    cdef np.uint32_t r11;
    cdef np.uint32_t r12;
    cdef np.uint32_t r21;
    cdef np.uint8_t v;
    cdef np.int8_t dc;
    cdef np.int16_t ds;
    cdef np.uint32_t dl;
    cdef np.int32_t diff=0;

    cdef int m = 0;
    while (i < sizeX*sizeY):
        r11 = buf[i-sizeX-1]
        r12 = buf[i-sizeX]
        r21 = buf[i-1]
        v = pt_lut[lut_idx]
        if (lut_off == 0):
            v = v & 0x03;
            lut_off += 1;
        elif (lut_off == 1):
            v = (v & 0x0C) >> 2
            lut_off += 1
        elif (lut_off == 2):
            v = (v & 0x30) >> 4
            lut_off += 1
        elif (lut_off == 3):
            v = (v & 0xC0) >> 6;
            lut_off = 0;
            lut_idx += 1;
        if (v == 0):
            fread(&dc,1,1,cfile)
            diff = dc;
        elif(v == 1):
            fread(&ds,2,1,cfile)
            diff = ds;
        elif(v==2):
            fread(&diff,4,1,cfile)
        buf[i] = r21+r12+diff-r11
        i += 1;

    fclose(cfile);
    return np.reshape(buf,[sizeY,sizeX])

@cython.boundscheck(False)
def readCpsImage(filename):
    cdef unsigned int i = 0
    cdef unsigned int sizeX 
    cdef unsigned int sizeY
    
    cdef int totbytes = 120+10*4+41*8

#    cdef np.ndarray[char, ndim=1] tmp = np.zeros([1024], dtype=np.char)
    cdef char tmp[1024] 
    cdef np.ndarray[np.uint8_t,ndim=1] pt_lut
    cdef np.ndarray[np.uint32_t, ndim=1] buf 

    cdef size_t nbytes 
    cdef unsigned int lut_idx=0;
    cdef unsigned int lut_off=0;
    cdef np.uint32_t r11;
    cdef np.uint32_t r12;
    cdef np.uint32_t r21;
    cdef np.uint8_t v;
    cdef np.int8_t dc;
    cdef np.int16_t ds;
    cdef np.uint32_t dl;
    cdef np.int32_t diff=0;

    cdef int m = 0;

    filename_byte = filename.encode("UTF-8")
    cdef char* fname = filename_byte
    cdef FILE* cfile;
    cfile = fopen(fname,"rb")
    result = []
    headers = []
    HndHeader = namedtuple("HndHeader","filetype filelength checksumSpec checksum creationDate creationTime patientID patientSer seriesID seriesSer "\
            "sliceID sliceSer sizeX sizeY sliceZPos modality window level pixelOffset imageType gantryRotation SAD SFD collX1 collX2 collY1 collY2 "\
            "collRotation filedX filedY bladeX1 bladeX2 bladeY1 bladeY2 IDUPosLng IDUPosLat IDUPosVrt IDUPosRtn patientSupportAngle tableTopEccentricAngle "\
            "couchVrt couchLng couchLat IDUResolutionX IDUResolutionY imageResolutionX imageResolutionY energy doseRate xRayKV xrayMA metersetExposure " \
            "acqAdjustment ctProjectionAngle ctNormChamber gatingTimeTag gating4DInfoX gating4DInfoY gating4DInfoZ gating4DInfoTime")
    formatstring = "=32sI4sI8s8s16sI16sI16sIIId16sIII4sdddddddddddddddddddddddddddddddddddddddd"
 
    while(fread(&tmp[0],1024,1,cfile) > 0):
        header = HndHeader._make(struct.unpack(formatstring,tmp[:totbytes]))
        sizeX = header.sizeX
        sizeY = header.sizeY
        buf = np.zeros([sizeX*sizeY], dtype=np.uint32)
        pt_lut = np.zeros([sizeX*sizeY], dtype=np.uint8)
        nbytes = (sizeX*(sizeY-1))/4
        lut_idx = 0
        lut_off =0
        diff = 0
        fread(&pt_lut[0],1,nbytes,cfile)

        #Read first row
        fread(&buf[0],4,sizeX,cfile)

        #Read first pixel of second row
        fread(&buf[sizeX],4,1,cfile)
        i = sizeX+1
        m = 0
        while (i < sizeX*sizeY):
            r11 = buf[i-sizeX-1]
            r12 = buf[i-sizeX]
            r21 = buf[i-1]
            v = pt_lut[lut_idx]
            if (lut_off == 0):
                v = v & 0x03;
                lut_off += 1;
            elif (lut_off == 1):
                v = (v & 0x0C) >> 2
                lut_off += 1
            elif (lut_off == 2):
                v = (v & 0x30) >> 4
                lut_off += 1
            elif (lut_off == 3):
                v = (v & 0xC0) >> 6;
                lut_off = 0;
                lut_idx += 1;
            if (v == 0):
                fread(&dc,1,1,cfile)
                diff = dc;
            elif(v == 1):
                fread(&ds,2,1,cfile)
                diff = ds;
            elif(v==2):
                fread(&diff,4,1,cfile)
            buf[i] = r21+r12+diff-r11
            i += 1;
        result.append(np.array(np.reshape(buf,[sizeY,sizeX])))
        headers.append(header)

    fclose(cfile)
    return result,headers
    


    


