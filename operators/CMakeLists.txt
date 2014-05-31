if (WIN32)
ADD_DEFINITIONS(-D__BUILD_GADGETRON_SOLVERS__)
endif (WIN32)


cuda_add_library( tomography_operators SHARED
  cuHaarWaveletOperator.cu
  cuGaussianFilterOperator.cu
)

install(FILES 	
	hoCuPartialDerivativeOperator.h
	cuGaussianFilterOperator.h
	subsetOperator.h
DESTINATION include)
