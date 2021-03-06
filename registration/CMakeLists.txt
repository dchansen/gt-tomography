
include_directories(
  ${CUDA_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIR}
  ${GADGETRON_INCLUDE_DIR}
  ${GDCM_INCLUDE_DIRS}
  ${CMAKE_SOURCE_DIR}/solvers
  ${CMAKE_SOURCE_DIR}/operators
  ${CMAKE_SOURCE_DIR}/proton
)


cuda_add_library(gt_registration SHARED
        cuDemonsSolver.cu
        quadratureKernels.cpp
        morphon.cu
        )



add_executable(register_Demons_3dV2 register_Demons_3dV2.cpp)
target_link_libraries(register_Demons_3dV2 gt_registration tomography_solvers gadgetron_toolbox_cpureg gadgetron_toolbox_gpureg gadgetron_toolbox_log tomography_operators gadgetron_toolbox_gpusolvers gadgetron_toolbox_cpucore gadgetron_toolbox_cpucore_math gadgetron_toolbox_gpucore gadgetron_toolbox_cpureg gadgetron_toolbox_gpuoperators gadgetron_toolbox_hostutils gadgetron_toolbox_gpureg ${HDF5_LIBRARIES} ${CUDA_CUFFT_LIBRARIES} ${CULA_LIBRARIES} ${CUDA_LIBRARIES}  ${Boost_LIBRARIES})

add_executable(register_Demons_3d register_Demons_3d.cpp)
target_link_libraries(register_Demons_3d gt_registration tomography_solvers gadgetron_toolbox_cpureg gadgetron_toolbox_gpureg gadgetron_toolbox_log tomography_operators gadgetron_toolbox_gpusolvers gadgetron_toolbox_cpucore gadgetron_toolbox_cpucore_math gadgetron_toolbox_gpucore gadgetron_toolbox_cpureg gadgetron_toolbox_gpuoperators gadgetron_toolbox_hostutils gadgetron_toolbox_gpureg ${HDF5_LIBRARIES} ${CUDA_CUFFT_LIBRARIES} ${CULA_LIBRARIES} ${CUDA_LIBRARIES}  ${Boost_LIBRARIES})

add_executable(register_Demons_4d register_Demons_4d.cpp)
target_link_libraries(register_Demons_4d gt_registration tomography_solvers gadgetron_toolbox_cpureg gadgetron_toolbox_gpureg gadgetron_toolbox_log tomography_operators gadgetron_toolbox_gpusolvers gadgetron_toolbox_cpucore gadgetron_toolbox_cpucore_math gadgetron_toolbox_gpucore gadgetron_toolbox_cpureg gadgetron_toolbox_gpuoperators gadgetron_toolbox_hostutils gadgetron_toolbox_gpureg ${HDF5_LIBRARIES} ${CUDA_CUFFT_LIBRARIES} ${CULA_LIBRARIES} ${CUDA_LIBRARIES}  ${Boost_LIBRARIES})

install(TARGETS register_Demons_3dV2 DESTINATION bin COMPONENT main)
install(TARGETS register_Demons_3d DESTINATION bin COMPONENT main)
install(TARGETS register_Demons_4d DESTINATION bin COMPONENT main)
install(TARGETS gt_registration
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin
        COMPONENT main
        )
