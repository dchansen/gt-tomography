cmake_minimum_required(VERSION 2.6)

PROJECT(gt-tomography)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

find_package(Gadgetron REQUIRED)



#find_path(GADGETRON_CMAKE_MODULES FindGadgetron.cmake HINTS
#$ENV{GADGETRON_HOME}/cmake
#/usr/local/gadgetron/cmake)

#if (NOT GADGETRON_CMAKE_MODULES)
#  MESSAGE(FATAL_ERROR "GADGETRON_CMAKE_MODULES cannot be found.
#   Try to set GADGETRON_HOME environment variable.")
#endif(NOT GADGETRON_CMAKE_MODULES)


include_directories(  "${GADGETRON_HOME}/include/gadgetron")
link_directories("${GADGETRON_LIB_DIR}/")
# build options for OpenMP support
find_package(OpenMP)
if (OPENMP_FOUND)
  message("OpenMP multithreading enabled")
  set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  OPTION(USE_OPENMP "Use OpenMP" On)
else (OPENMP_FOUND)
  message("OpenMP multithreading not supported")
endif (OPENMP_FOUND)
if(USE_OPENMP)
  ADD_DEFINITIONS(-DUSE_OMP)
endif(USE_OPENMP)

find_package(DCMTK REQUIRED)

FIND_PACKAGE(GDCM REQUIRED)
    IF(GDCM_FOUND)
        INCLUDE(${GDCM_USE_FILE})
        SET(GDCM_LIBRARIES 
            gdcmCommon
            gdcmDICT
            gdcmDSED
            gdcmIOD
            gdcmjpeg12
            gdcmjpeg16
            gdcmjpeg8
            gdcmMEXD
            gdcmMSFF
            socketxx
            )
    ELSE(GDCM_FOUND)
        MESSAGE(FATAL_ERROR "Cannot find GDCM, did you set GDCM_DIR?")
    ENDIF(GDCM_FOUND)
find_package(Boost COMPONENTS system thread program_options filesystem REQUIRED)
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --std=c++11")
find_package(CUDA 4.0 REQUIRED)

#ADD_DEFINITIONS(-D__STRICT_ANSI__)

#ADD_DEFINITIONS(-D_MATRIX_INTRIN_H_INCLUDED)
if (CUDA_FOUND)
  set(CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-gencode arch=compute_52,code=sm_52" )
  #set(CUDA_VERBOSE_BUILD ON)

  # Compile kernels for compute models 1.0 and 2.0 as default for Cuda 4.1
  # Support compute model 3.0 from Cuda 4.2 and up
  # Support compute model 3.5 from Cuda 5 and up

  MESSAGE("Compiling with ${CUDA_NVCC_FLAGS}")
endif (CUDA_FOUND)

find_package(GTest)
#Add support for the default ubuntu package of gtest (which is not compiled
if (NOT GTEST_FOUND)
find_path(GTEST_SRC_DIR src/gtest.cc HINTS /usr/src/gtest)
find_path(GTEST_INCLUDE_DIRS gtest.h HINTS /usr/include/gtest)
if (GTEST_SRC_DIR AND GTEST_INCLUDE_DIRS)
MESSAGE("GTest src package found. Compiling as part of Gadgetron.")
add_subdirectory(${GTEST_SRC_DIR} ${CMAKE_BINARY_DIR}/gtest )
include_directories(${GTEST_INCLUDE_DIRS})
set(GTEST_FOUND 1)
set(GTEST_LIBRARIES gtest gtest_main)
endif (GTEST_SRC_DIR AND GTEST_INCLUDE_DIRS)
endif (NOT GTEST_FOUND)

INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR})
add_subdirectory(solvers)
add_subdirectory(operators)
add_subdirectory(proton)
add_subdirectory(denoise)
add_subdirectory(xray)
add_subdirectory(test)
#add_subdirectory(mri)
