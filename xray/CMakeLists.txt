find_package(HDF5 REQUIRED HL)

include_directories(
  ${CUDA_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIR}
  ${GADGETRON_INCLUDE_DIR}
  ${GDCM_INCLUDE_DIRS}
  ${CMAKE_SOURCE_DIR}/solvers
  ${CMAKE_SOURCE_DIR}/operators
  ${CMAKE_SOURCE_DIR}/proton
        ${CMAKE_SOURCE_DIR}/denoise
)

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

MESSAGE("Include dir ${GDCM_INCLUDE_DIRS}")

# The Armadillo sparse matrix suppport seems to be too slow be be a real alternative to the eigen version
#

#find_path(EIGEN_INCLUDE_DIR Eigen/Sparse 
#  HINTS /usr/local/include/eigen3 /usr/include/eigen3)

#if(EIGEN_INCLUDE_DIR)
#  set(EIGEN_RESAMPLER hoLinearResampleOperator_eigen.cpp)
#  include_directories(${EIGEN_INCLUDE_DIR})  
#elseif(EIGEN_INCLUDE_DIR)
#  MESSAGE(STATUS "Eigen not found. Not compiling the Eigen based resampler and the CB_OF_reconstruct app") 
#endif(EIGEN_INCLUDE_DIR)

cuda_add_library(Xray_kernels SHARED
  conebeam_projection.cu hoCuConebeamProjectionOperator.cpp cuConebeamProjectionOperator.cpp CBSubsetOperator.cpp dicomWriter.cpp  hoLinearResampleOperator_eigen.cpp hoCuOFPartialDerivativeOperator.cpp  projection_utils.cpp mssim.cpp ${EIGEN_RESAMPLER}
  )

target_link_libraries(Xray_kernels tomography_operators gadgetron_toolbox_gpucore gadgetron_toolbox_gpunfft gadgetron_toolbox_gpucore gadgetron_toolbox_gpufft gadgetron_toolbox_gpureg ${CUDA_nppc_LIBRARY} ${CUDA_nppi_LIBRARY} ${CUDA_npps_LIBRARY} ${CUDA_LIBRARIES} ${DCMTK_dcmdata_LIBRARY} ${DCMTK_oflog_LIBRARY} ${DCMTK_ofstd_LIBRARY} ${GDCM_LIBRARIES})

add_executable(FDK_reconstruct_3d FDK_reconstruct_3d.cpp)
target_link_libraries(FDK_reconstruct_3d Xray_kernels tomography_operators gadgetron_toolbox_log gadgetron_toolbox_cpucore gadgetron_toolbox_cpucore_math gadgetron_toolbox_cpureg gadgetron_toolbox_gpucore gadgetron_toolbox_hostutils ${HDF5_LIBRARIES} ${CUDA_CUFFT_LIBRARIES} ${CULA_LIBRARIES} ${CUDA_LIBRARIES} ${Boost_LIBRARIES})

add_executable(FDK_reconstruct_4d FDK_reconstruct_4d.cpp)
target_link_libraries(FDK_reconstruct_4d Xray_kernels gadgetron_toolbox_log gadgetron_toolbox_cpucore gadgetron_toolbox_cpucore_math gadgetron_toolbox_cpureg gadgetron_toolbox_gpucore  gadgetron_toolbox_hostutils ${HDF5_LIBRARIES} ${CUDA_CUFFT_LIBRARIES} ${CULA_LIBRARIES} ${CUDA_LIBRARIES}  ${Boost_LIBRARIES})

#add_executable(FFTrebin FFTrebin.cpp)
#target_link_libraries(FFTrebin Xray_kernels gadgetron_toolbox_log gadgetron_toolbox_cpucore gadgetron_toolbox_cpucore_math gadgetron_toolbox_cpureg gadgetron_toolbox_gpucore  gadgetron_toolbox_hostutils ${HDF5_LIBRARIES} ${CUDA_CUFFT_LIBRARIES} ${CULA_LIBRARIES} ${CUDA_LIBRARIES}  ${Boost_LIBRARIES})


add_executable(CB_reconstruct CB_reconstruct.cpp)
target_link_libraries(CB_reconstruct Xray_kernels tomography_solvers gadgetron_toolbox_log gadgetron_toolbox_gpudwt  gadgetron_toolbox_cpucore gadgetron_toolbox_cpucore_math gadgetron_toolbox_gpucore gadgetron_toolbox_cpureg gadgetron_toolbox_gpuoperators gadgetron_toolbox_hostutils  ${HDF5_LIBRARIES} ${CUDA_CUFFT_LIBRARIES} ${CULA_LIBRARIES} ${CUDA_LIBRARIES}  ${Boost_LIBRARIES})
add_executable(cuCB_reconstruct cuCB_reconstruct.cpp)

target_link_libraries(cuCB_reconstruct Xray_kernels tomography_operators gadgetron_toolbox_gpudwt  tomography_solvers gadgetron_toolbox_log gadgetron_toolbox_gpusolvers gadgetron_toolbox_cpucore gadgetron_toolbox_cpucore_math gadgetron_toolbox_gpucore gadgetron_toolbox_cpureg gadgetron_toolbox_gpuoperators gadgetron_toolbox_hostutils  ${HDF5_LIBRARIES} ${CUDA_CUFFT_LIBRARIES} ${CULA_LIBRARIES} ${CUDA_LIBRARIES}  ${Boost_LIBRARIES})

add_executable(CBOS_reconstruct CBOS_reconstruct.cpp)
target_link_libraries(CBOS_reconstruct Xray_kernels tomography_solvers gadgetron_toolbox_log gadgetron_toolbox_gpusolvers gadgetron_toolbox_cpucore gadgetron_toolbox_cpucore_math gadgetron_toolbox_gpucore gadgetron_toolbox_cpureg gadgetron_toolbox_gpuoperators gadgetron_toolbox_hostutils  ${HDF5_LIBRARIES} ${CUDA_CUFFT_LIBRARIES} ${CULA_LIBRARIES} ${CUDA_LIBRARIES}  ${Boost_LIBRARIES})

add_executable(CBOS_OF_reconstruct CBOS_OF_reconstruct.cpp)
target_link_libraries(CBOS_OF_reconstruct Xray_kernels tomography_solvers gadgetron_toolbox_cpureg gadgetron_toolbox_gpureg gadgetron_toolbox_log gadgetron_toolbox_gpusolvers gadgetron_toolbox_cpucore gadgetron_toolbox_cpucore_math gadgetron_toolbox_gpucore gadgetron_toolbox_cpureg gadgetron_toolbox_gpuoperators gadgetron_toolbox_hostutils gadgetron_toolbox_gpureg ${HDF5_LIBRARIES} ${CUDA_CUFFT_LIBRARIES} ${CULA_LIBRARIES} ${CUDA_LIBRARIES}  ${Boost_LIBRARIES})

add_executable(cuCBOS_reconstruct cuCBOS_reconstruct.cpp)
target_link_libraries(cuCBOS_reconstruct denoise_kernels Xray_kernels tomography_operators tomography_solvers gadgetron_toolbox_log gadgetron_toolbox_gpusolvers gadgetron_toolbox_cpucore gadgetron_toolbox_cpucore_math gadgetron_toolbox_gpucore gadgetron_toolbox_cpureg gadgetron_toolbox_gpuoperators gadgetron_toolbox_hostutils  ${HDF5_LIBRARIES} ${CUDA_CUFFT_LIBRARIES} ${CULA_LIBRARIES} ${CUDA_LIBRARIES}  ${Boost_LIBRARIES})

add_executable(hoCuCBOS_reconstruct hoCuCBOS_reconstruct.cpp)
target_link_libraries(hoCuCBOS_reconstruct Xray_kernels tomography_operators tomography_solvers gadgetron_toolbox_log gadgetron_toolbox_gpusolvers gadgetron_toolbox_cpucore gadgetron_toolbox_cpucore_math gadgetron_toolbox_gpucore gadgetron_toolbox_cpureg gadgetron_toolbox_gpuoperators gadgetron_toolbox_hostutils  ${HDF5_LIBRARIES} ${CUDA_CUFFT_LIBRARIES} ${CULA_LIBRARIES} ${CUDA_LIBRARIES}  ${Boost_LIBRARIES})


add_executable(cuCBOS2_reconstruct cuCBOS2_reconstruct.cpp)
target_link_libraries(cuCBOS2_reconstruct Xray_kernels tomography_operators tomography_solvers gadgetron_toolbox_log gadgetron_toolbox_gpusolvers gadgetron_toolbox_cpucore gadgetron_toolbox_cpucore_math gadgetron_toolbox_gpucore gadgetron_toolbox_cpureg gadgetron_toolbox_gpuoperators gadgetron_toolbox_hostutils  ${HDF5_LIBRARIES} ${CUDA_CUFFT_LIBRARIES} ${CULA_LIBRARIES} ${CUDA_LIBRARIES}  ${Boost_LIBRARIES})

add_executable(cuCBOSMOM cuCBOSMOM.cpp)
target_link_libraries(cuCBOSMOM Xray_kernels gadgetron_toolbox_log gadgetron_toolbox_gpusolvers gadgetron_toolbox_cpucore gadgetron_toolbox_cpucore_math gadgetron_toolbox_gpucore gadgetron_toolbox_cpureg gadgetron_toolbox_gpuoperators gadgetron_toolbox_hostutils  ${HDF5_LIBRARIES} ${CUDA_CUFFT_LIBRARIES} ${CULA_LIBRARIES} ${CUDA_LIBRARIES}  ${Boost_LIBRARIES})



add_executable(CB_SBreconstruct CB_SBreconstruct.cpp)
target_link_libraries(CB_SBreconstruct Xray_kernels gadgetron_toolbox_log gadgetron_toolbox_cpucore gadgetron_toolbox_cpucore_math gadgetron_toolbox_gpucore gadgetron_toolbox_cpureg gadgetron_toolbox_gpuoperators gadgetron_toolbox_hostutils  ${HDF5_LIBRARIES} ${CULA_LIBRARIES} ${CUDA_CUFFT_LIBRARIES} ${CUDA_LIBRARIES}  ${Boost_LIBRARIES})

add_executable(denoise_TV denoise_TV.cpp)
target_link_libraries(denoise_TV gadgetron_toolbox_log gadgetron_toolbox_cpucore gadgetron_toolbox_gpusolvers gadgetron_toolbox_cpucore_math gadgetron_toolbox_gpucore gadgetron_toolbox_cpureg gadgetron_toolbox_gpuoperators gadgetron_toolbox_hostutils  ${HDF5_LIBRARIES} ${CULA_LIBRARIES} ${CUDA_CUFFT_LIBRARIES} ${CUDA_LIBRARIES}  ${Boost_LIBRARIES})

#add_executable(CB_CGreconstruct CB_CGreconstruct.cpp)
#target_link_libraries(CB_CGreconstruct Xray_kernels cpucore cpucore_math gpucore cpureg gpuoperators gadgetron_toolbox_hostutils  ${HDF5_LIBRARIES} ${CULA_LIBRARIES} ${CUDA_CUFFT_LIBRARIES} ${CUDA_LIBRARIES}  ${Boost_LIBRARIES})

add_executable(CBCT_forwards_projection CBCT_forwards_projection.cpp)
target_link_libraries(CBCT_forwards_projection Xray_kernels gadgetron_toolbox_log gadgetron_toolbox_cpucore gadgetron_toolbox_cpucore_math gadgetron_toolbox_gpucore gadgetron_toolbox_hostutils ${HDF5_LIBRARIES} ${CUDA_LIBRARIES}  ${CUDA_CUFFT_LIBRARIES} ${CULA_LIBRARIES} ${Boost_LIBRARIES})

#add_executable(convolveTest convolveTest.cpp)
#target_link_libraries(convolveTest Xray_kernels gadgetron_toolbox_log gadgetron_toolbox_cpucore gadgetron_toolbox_cpucore_math gadgetron_toolbox_cpureg gadgetron_toolbox_gpucore  gadgetron_toolbox_hostutils ${HDF4_LIBRARIES} ${CUDA_CUFFT_LIBRARIES} ${CULA_LIBRARIES} ${CUDA_LIBRARIES}  ${Boost_LIBRARIES})
#if(EIGEN_INCLUDE_DIR)
add_executable(CB_OF_reconstruct CB_OF_reconstruct.cpp)
target_link_libraries(CB_OF_reconstruct Xray_kernels gadgetron_toolbox_log gadgetron_toolbox_cpucore gadgetron_toolbox_cpucore_math gadgetron_toolbox_gpucore gadgetron_toolbox_cpureg gadgetron_toolbox_gpureg gadgetron_toolbox_gpuoperators  gadgetron_toolbox_hostutils ${HDF5_LIBRARIES} ${CUDA_LIBRARIES}  ${Boost_LIBRARIES})


add_executable(cuCBOSStat_reconstruct cuCBOSStat_reconstruct.cpp)
target_link_libraries(cuCBOSStat_reconstruct Xray_kernels tomography_operators tomography_solvers gadgetron_toolbox_log gadgetron_toolbox_gpusolvers gadgetron_toolbox_cpucore gadgetron_toolbox_cpucore_math gadgetron_toolbox_gpucore gadgetron_toolbox_cpureg gadgetron_toolbox_gpuoperators gadgetron_toolbox_hostutils  ${HDF5_LIBRARIES} ${CUDA_CUFFT_LIBRARIES} ${CULA_LIBRARIES} ${CUDA_LIBRARIES}  ${Boost_LIBRARIES})


add_executable(compare_images compare_images.cpp)
target_link_libraries(compare_images Xray_kernels tomography_operators tomography_solvers gadgetron_toolbox_log gadgetron_toolbox_gpusolvers gadgetron_toolbox_cpucore gadgetron_toolbox_cpucore_math gadgetron_toolbox_gpucore gadgetron_toolbox_cpureg gadgetron_toolbox_gpuoperators gadgetron_toolbox_hostutils  ${HDF5_LIBRARIES} ${CUDA_CUFFT_LIBRARIES} ${CULA_LIBRARIES} ${CUDA_LIBRARIES}  ${Boost_LIBRARIES})



if (${BUILDCT})
    cuda_add_library(CTXray_kernels ct_projection.cu CT_acquisition.cpp dicomCTWriter.cpp CTSubsetOperator.cpp CTProjectionOperator.cpp)
    target_link_libraries(CTXray_kernels gadgetron_toolbox_gpucore gadgetron_toolbox_gpunfft gadgetron_toolbox_gpucore gadgetron_toolbox_gpufft gadgetron_toolbox_gpureg ${CUDA_LIBRARIES} ${DCMTK_dcmdata_LIBRARY} ${DCMTK_oflog_LIBRARY} ${DCMTK_ofstd_LIBRARY} ${GDCM_LIBRARIES})
add_executable(cuCT_reconstruct cuCT_reconstruct.cpp)
target_link_libraries(cuCT_reconstruct CTXray_kernels Xray_kernels tomography_solvers gadgetron_toolbox_log gadgetron_toolbox_cpucore gadgetron_toolbox_cpucore_math gadgetron_toolbox_gpucore gadgetron_toolbox_cpureg gadgetron_toolbox_gpureg gadgetron_toolbox_gpuoperators  gadgetron_toolbox_hostutils ${HDF5_LIBRARIES} ${CUDA_LIBRARIES}  ${Boost_LIBRARIES})

add_executable(CT_reconstruct CT_reconstruct.cpp)
target_link_libraries(CT_reconstruct CTXray_kernels Xray_kernels tomography_solvers gadgetron_toolbox_log gadgetron_toolbox_cpucore gadgetron_toolbox_cpucore_math gadgetron_toolbox_gpucore gadgetron_toolbox_cpureg gadgetron_toolbox_gpureg gadgetron_toolbox_gpuoperators  gadgetron_toolbox_hostutils ${HDF5_LIBRARIES} ${CUDA_LIBRARIES}  ${Boost_LIBRARIES})
#endif(EIGEN_INCLUDE_DIR)

add_executable(cuCTOS_reconstruct cuCTOS_reconstruct.cpp)
target_link_libraries(cuCTOS_reconstruct CTXray_kernels Xray_kernels tomography_operators tomography_solvers gadgetron_toolbox_log gadgetron_toolbox_gpusolvers gadgetron_toolbox_cpucore gadgetron_toolbox_cpucore_math gadgetron_toolbox_gpucore gadgetron_toolbox_cpureg gadgetron_toolbox_gpuoperators gadgetron_toolbox_hostutils  ${HDF5_LIBRARIES} ${CUDA_CUFFT_LIBRARIES} ${CULA_LIBRARIES} ${CUDA_LIBRARIES}  ${Boost_LIBRARIES})

add_executable(CTOS_reconstruct CTOS_reconstruct.cpp)
target_link_libraries(CTOS_reconstruct CTXray_kernels denoise_kernels Xray_kernels tomography_operators tomography_solvers gadgetron_toolbox_log gadgetron_toolbox_gpusolvers gadgetron_toolbox_cpucore gadgetron_toolbox_cpucore_math gadgetron_toolbox_gpucore gadgetron_toolbox_cpureg gadgetron_toolbox_gpuoperators gadgetron_toolbox_hostutils  ${HDF5_LIBRARIES} ${CUDA_CUFFT_LIBRARIES} ${CULA_LIBRARIES} ${CUDA_LIBRARIES}  ${Boost_LIBRARIES})

add_executable(CTOS2_reconstruct CTOS2_reconstruct.cpp)
target_link_libraries(CTOS2_reconstruct CTXray_kernels denoise_kernels Xray_kernels tomography_operators tomography_solvers gadgetron_toolbox_log gadgetron_toolbox_gpusolvers gadgetron_toolbox_cpucore gadgetron_toolbox_cpucore_math gadgetron_toolbox_gpucore gadgetron_toolbox_cpureg gadgetron_toolbox_gpuoperators gadgetron_toolbox_hostutils  ${HDF5_LIBRARIES} ${CUDA_CUFFT_LIBRARIES} ${CULA_LIBRARIES} ${CUDA_LIBRARIES}  ${Boost_LIBRARIES})

add_executable(CTOS3_reconstruct CTOS3_reconstruct.cpp)
target_link_libraries(CTOS3_reconstruct CTXray_kernels denoise_kernels Xray_kernels tomography_operators tomography_solvers gadgetron_toolbox_log gadgetron_toolbox_gpusolvers gadgetron_toolbox_cpucore gadgetron_toolbox_cpucore_math gadgetron_toolbox_gpucore gadgetron_toolbox_cpureg gadgetron_toolbox_gpuoperators gadgetron_toolbox_hostutils  ${HDF5_LIBRARIES} ${CUDA_CUFFT_LIBRARIES} ${CULA_LIBRARIES} ${CUDA_LIBRARIES}  ${Boost_LIBRARIES})
add_executable(CTStat_reconstruct CTStat_reconstruct.cpp)
target_link_libraries(CTStat_reconstruct CTXray_kernels denoise_kernels Xray_kernels tomography_operators tomography_solvers gadgetron_toolbox_log gadgetron_toolbox_gpusolvers gadgetron_toolbox_cpucore gadgetron_toolbox_cpucore_math gadgetron_toolbox_gpucore gadgetron_toolbox_cpureg gadgetron_toolbox_gpuoperators gadgetron_toolbox_hostutils  ${HDF5_LIBRARIES} ${CUDA_CUFFT_LIBRARIES} ${CULA_LIBRARIES} ${CUDA_LIBRARIES}  ${Boost_LIBRARIES})
#install(TARGETS Reconstruct DESTINATION bin)
endif(${BUILDCT})