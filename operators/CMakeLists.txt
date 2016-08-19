if (WIN32)
ADD_DEFINITIONS(-D__BUILD_GADGETRON_SOLVERS__)
endif (WIN32)


cuda_add_library( tomography_operators SHARED
  cuHaarWaveletOperator.cu
  cuGaussianFilterOperator.cu
  cuATrousWavelet.cu
  cuEdgeWavelet.cu
  cuDCT.cu
		cuPartialDifferenceOperator.cu
		cuSmallConvOperator.cu
		cuScaleOperator.cu
		cuTVPrimalDualOperator.cu
)

install(FILES
		cuPartialDifferenceOperator.h
	hoCuPartialDerivativeOperator.h
	cuGaussianFilterOperator.h
	subsetOperator.h
	cuATrousWavelet.h
	invertibleOperator.h
	cuDCT.h
DESTINATION include)
