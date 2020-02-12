if (WIN32)
ADD_DEFINITIONS(-D__BUILD_GADGETRON_SOLVERS__)
endif (WIN32)
include_directories(
		${CUDA_INCLUDE_DIRS}
		${Boost_INCLUDE_DIR}
		${GADGETRON_INCLUDE_DIR}
		${GDCM_INCLUDE_DIRS}
		${CMAKE_SOURCE_DIR}/operators
		${CMAKE_SOURCE_DIR}/proton
		${CMAKE_SOURCE_DIR}/denoise
)


cuda_add_library( tomography_solvers SHARED
   solver_utils.cu solver_utils.h  cuOSMOMSolverD.cu
  
)

target_link_libraries(tomography_solvers gadgetron_toolbox_gpusolvers)
install(FILES
	cuNCGSolver.h 	
	mlSolver.h
	cuMLSolver.h
	cuSARTSolver.h
	sartSolver.h
	hoGPBBSolver.h
DESTINATION include)
install(TARGETS tomography_solvers
		LIBRARY DESTINATION lib
		ARCHIVE DESTINATION lib
		RUNTIME DESTINATION bin
		COMPONENT main
		)
