
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
        )


