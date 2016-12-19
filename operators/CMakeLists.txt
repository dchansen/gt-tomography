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
		cuWTVPrimalDualOperator.cu
		cuBilateralFilter.cu
		cuTFFT.cpp
)
target_link_libraries(tomography_operators gadgetron_toolbox_gpusolvers gadgetron_toolbox_gpucore gadgetron_toolbox_gpuoperators gadgetron_toolbox_gpunfft gadgetron_toolbox_gpucore gadgetron_toolbox_gpufft gadgetron_toolbox_gpureg ${CUDA_LIBRARIES} ${DCMTK_dcmdata_LIBRARY} ${DCMTK_oflog_LIBRARY} ${DCMTK_ofstd_LIBRARY} ${GDCM_LIBRARIES})
install(FILES
		cuPartialDifferenceOperator.h
	hoCuPartialDerivativeOperator.h
	cuGaussianFilterOperator.h
	subsetOperator.h
	cuATrousWavelet.h
	invertibleOperator.h
	cuDCT.h
DESTINATION include)
