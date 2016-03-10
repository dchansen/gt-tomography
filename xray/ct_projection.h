#pragma once

#include "hoCuNDArray.h"
#include "cuNDArray.h"
#include "vector_td.h"
#include "gpuxray_export.h"

namespace Gadgetron {


/**
 *
 * @param projections
 * @param image
 * @param angles
 * @param offsets
 * @param indices
 * @param projections_per_batch
 * @param samples_per_pixel
 * @param is_dims_in_mm
 * @param ps_dims_in_mm
 * @param SDD
 * @param SAD
 * @param accumulate
 */
  // Forwards projection of a 3D volume onto a set of projections.
  // - dependening on the provided binnning indices, just a subset of the projections can be targeted.
  //

  void ct_forwards_projection
    ( cuNDArray<float> *projections,
				cuNDArray<float> *image,
				std::vector<float> angles,
				std::vector<floatd2> offsets,
				float samples_per_pixel,
				floatd3 is_dims_in_mm,
				floatd2 ps_dims_in_mm,
				float SDD,
				float SAD,
				bool accumulate
  );

  // Backprojection of a set of projections onto a 3D volume.
  // - depending on the provided binnning indices, just a subset of the projections can be included
  //

void ct_backwards_projection( cuNDArray<float> *projections,
		cuNDArray<float> *image,
       	std::vector<floatd3> & detector_focal_cyls, //phi,rho,z of the detector focal spot in units of rad, mm and mm
		std::vector<floatd3> & focal_offset_cyls, // phi,rho,z offset of the source focal spot compared to detector focal spot
        std::vector<floatd2> & centralElements, // Central element on the detector
        std::vector<intd2> & proj_indices, // Array of size nz containing first to last projection for slice
		floatd3 is_dims_in_mm, // Image size in mm
		floatd2 ps_spacing,  //Size of each projection element in mm
		float ADD, //aparture - detector disance
		bool accumulate
);

void ct_forwards_projection( cuNDArray<float> *projections,
		cuNDArray<float> *image,
       	std::vector<floatd3> & detector_focal_cyls, //phi,rho,z of the detector focal spot in units of rad, mm and mm
		std::vector<floatd3> & focal_offset_cyls, // phi,rho,z offset of the source focal spot compared to detector focal spot
        std::vector<floatd2> & centralElements, // Central element on the detector
		floatd3 is_dims_in_mm, // Image size in mm
		floatd2 ps_spacing,  //Size of each projection element in mm
		float ADD, //aparture - detector disance
                             float samples_per_ray,
		bool accumulate
);

/*
  // Forwards projection of a 3D volume onto a set of projections.
  // - dependening on the provided binnning indices, just a subset of the projections can be targeted.
  //
  
  EXPORTGPUXRAY void conebeam_forwards_projection
    ( hoCuNDArray<float> *projections,
				hoCuNDArray<float> *image,
				std::vector<float> angles, 
				std::vector<floatd2> offsets, 
				std::vector<unsigned int> indices,
				int projections_per_batch, 
				float samples_per_pixel,
				floatd3 is_dims_in_mm, 
				floatd2 ps_dims_in_mm,
				float SDD, 
				float SAD
  );
  
  // Backprojection of a set of projections onto a 3D volume.
  // - depending on the provided binnning indices, just a subset of the projections can be included
  //

  template <bool FBP> EXPORTGPUXRAY void conebeam_backwards_projection( 
        hoCuNDArray<float> *projections,
        hoCuNDArray<float> *image,
        std::vector<float> angles, 
        std::vector<floatd2> offsets, 
        std::vector<unsigned int> indices,
        int projections_per_batch,
        intd3 is_dims_in_pixels, 
        floatd3 is_dims_in_mm, 
        floatd2 ps_dims_in_mm,
        float SDD, 
        float SAD,
        bool short_scan,
        bool use_offset_correction,
        bool accumulate, 
        cuNDArray<float> *cosine_weights = 0x0,
        cuNDArray<float> *frequency_filter = 0x0
  );

void apply_mask(cuNDArray<float>* image, cuNDArray<bool>* mask);
 */
}
