find_package(HDF5 REQUIRED HL)


include_directories(
  ${CUDA_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIR}
  ${GADGETRON_INCLUDE_DIR}
  ${CMAKE_SOURCE_DIR}/solvers
  ${CMAKE_SOURCE_DIR}/operators
  ${CMAKE_SOURCE_DIR}/xray
)

cuda_add_library( proton_kernels SHARED
  splineBackprojectionOperator.cu proton_kernels.cu histogram.cu hoCuParallelProjection.cu protonPreconditioner.cu proton_utils.cu proton_utils.h protonDataset.cu
)

target_link_libraries(proton_kernels gadgetron_toolbox_gpucore  gadgetron_toolbox_gpufft gadgetron_toolbox_gpunfft tomography_solvers ${CUDA_LIBRARIES} tomography_operators)

add_executable(protonReconstruct protonReconstruct.cpp )

add_executable(hostProtonReconstruct hostProtonReconstruct.cpp)
#add_executable(CGhostProtonReconstruct CGhostProtonReconstruct.cpp)

#add_executable(hostSBProtonReconstruct hostSBProtonReconstruct.cpp)
add_executable(hostFilteredProtonReconstruct hostFilteredProtonReconstruct.cpp)
add_executable(FilteredProtonReconstruct FilteredProtonReconstruct.cpp)

add_executable(hostSubsetProtonReconstruct hostSubsetProtonReconstruct.cpp)
add_executable(subsetProtonReconstruct subsetProtonReconstruct.cpp)
add_executable(DROPProtonReconstruct DROPProtonReconstruct.cpp)
add_executable(hostDROPProtonReconstruct hostDROPProtonReconstruct.cpp)
#add_executable(hostBufferedSubsetProtonReconstruct hostBufferedSubsetProtonReconstruct.cpp)

#add_executable(protonForward protonForwardProject.cpp)

#add_executable(LSQRTest LSQRTest.cpp)

target_link_libraries(protonReconstruct proton_kernels gadgetron_toolbox_gpucore gadgetron_toolbox_cpucore_math gadgetron_toolbox_gpuoperators gadgetron_toolbox_gpusolvers   gadgetron_toolbox_log ${HDF5_LIBRARIES} ${CUDA_LIBRARIES}   ${Boost_LIBRARIES})
target_link_libraries(hostProtonReconstruct proton_kernels gadgetron_toolbox_cpucore gadgetron_toolbox_cpucore_math gadgetron_toolbox_gpuoperators gadgetron_toolbox_gpucore  gadgetron_toolbox_log  ${HDF5_LIBRARIES}   ${CUDA_LIBRARIES}   ${Boost_LIBRARIES})
#target_link_libraries(CGhostProtonReconstruct proton_kernels cpucore cpucore_math gpuoperators gpucore   ${HDF5_LIBRARIES}   ${CUDA_LIBRARIES}   ${Boost_LIBRARIES})
#target_link_libraries(hostSBProtonReconstruct proton_kernels cpucore cpucore_math gpuoperators gpucore   ${HDF5_LIBRARIES}  ${CUDA_LIBRARIES}   ${Boost_LIBRARIES})
target_link_libraries(hostSubsetProtonReconstruct proton_kernels gadgetron_toolbox_cpucore gadgetron_toolbox_cpucore_math gadgetron_toolbox_gpucore  gadgetron_toolbox_gpusolvers   gadgetron_toolbox_gpuoperators gadgetron_toolbox_log ${HDF5_LIBRARIES}  ${CUDA_LIBRARIES}   ${Boost_LIBRARIES})
target_link_libraries(subsetProtonReconstruct proton_kernels gadgetron_toolbox_cpucore gadgetron_toolbox_cpucore_math gadgetron_toolbox_gpucore gadgetron_toolbox_gpusolvers  gadgetron_toolbox_log gadgetron_toolbox_gpuoperators ${HDF5_LIBRARIES}  ${CUDA_LIBRARIES}   ${Boost_LIBRARIES})
target_link_libraries(DROPProtonReconstruct proton_kernels gadgetron_toolbox_cpucore gadgetron_toolbox_cpucore_math gadgetron_toolbox_gpucore gadgetron_toolbox_gpusolvers  gadgetron_toolbox_log gadgetron_toolbox_gpuoperators ${HDF5_LIBRARIES}  ${CUDA_LIBRARIES}   ${Boost_LIBRARIES})
target_link_libraries(hostDROPProtonReconstruct proton_kernels gadgetron_toolbox_cpucore gadgetron_toolbox_cpucore_math gadgetron_toolbox_gpucore gadgetron_toolbox_gpusolvers  gadgetron_toolbox_log gadgetron_toolbox_gpuoperators ${HDF5_LIBRARIES}  ${CUDA_LIBRARIES}   ${Boost_LIBRARIES})
target_link_libraries(hostFilteredProtonReconstruct proton_kernels gadgetron_toolbox_cpucore gadgetron_toolbox_cpucore_math gadgetron_toolbox_gpucore  gadgetron_toolbox_log  gadgetron_toolbox_gpuoperators gadgetron_toolbox_cpufft ${HDF5_LIBRARIES}  ${CUDA_LIBRARIES}   ${Boost_LIBRARIES})
target_link_libraries(FilteredProtonReconstruct proton_kernels gadgetron_toolbox_cpucore gadgetron_toolbox_cpucore_math gadgetron_toolbox_gpucore  gadgetron_toolbox_log  gadgetron_toolbox_gpuoperators gadgetron_toolbox_cpufft ${HDF5_LIBRARIES}  ${CUDA_LIBRARIES}   ${Boost_LIBRARIES})
#target_link_libraries(hostBufferedSubsetProtonReconstruct proton_kernels gadgetron_toolbox_cpucore gadgetron_toolbox_cpucore_math gadgetron_toolbox_gpucore gadgetron_toolbox_log  gadgetron_toolbox_gpuoperators ${HDF5_LIBRARIES} ${CUDA_LIBRARIES}   ${Boost_LIBRARIES})
#target_link_libraries(LSQRTest proton_kernels gpucore  solvers operators ${CUDA_LIBRARIES}  )

#target_link_libraries(protonForward proton_kernels gpucore  solvers operators ${CUDA_LIBRARIES}  )

install(TARGETS protonReconstruct DESTINATION bin)
#install(TARGETS hostProtonReconstruct DESTINATION bin)

#install(TARGETS protonForward DESTINATION bin)

