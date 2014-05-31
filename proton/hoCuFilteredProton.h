#pragma once
#include "subsetOperator.h"
#include "hoCuNDArray.h"
#include "hoCuOperatorPathBackprojection.h"
#include "hdf5.h"
#include "hdf5_hl.h"
#include <numeric>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include "hoCuParallelProjection.h"
#include "hoNDFFT.h"

namespace Gadgetron{
class hoCuFilteredProton {

public:
	hoCuFilteredProton() {

	}
	//Terrible terrible name. Penguin_sauce would be as good...or better, 'cos penguin
	boost::shared_ptr<hoCuNDArray<float> >  calculate(std::vector<size_t> dims,vector_td<float,3> physical_dims,vector_td<float,3> origin){

		cuNDArray<float> image(dims);
		clear(&image);

		std::vector<size_t> dims_proj;

		vector_td<float,3> physical_dims_proj = physical_dims;
		for (int i  = 0; i < dims.size(); i++)
		{
			if (dims[i] == 1) dims_proj.push_back(1);
			else{
				dims_proj.push_back(dims[i]*2);
				physical_dims_proj[i] *= std::sqrt(2.0f);
			}
		}

		hoCuNDArray<float> ramp = *calc_filter(dims_proj[0]);
		unsigned int ngroups = spline_arrays.size();

		for (unsigned int group = 0; group < ngroups; group++){

			hoCuOperatorPathBackprojection<float> E;
			E.setup(spline_arrays[group],physical_dims, projection_arrays[group],origin);
			hoCuNDArray<float> projection(dims);
			E.mult_M(projection_arrays[group].get(),&projection);

			boost::shared_ptr<hoNDArray<float_complext> > proj_complex = real_to_complex<float_complext>(&projection);
			hoNDFFT<float>::instance()->fft(proj_complex.get(),0);
			*proj_complex *= ramp;
			hoNDFFT<float>::instance()->ifft(proj_complex.get(),0);
			projection = *real(proj_complex.get());
			parallel_backprojection(&projection,&image,angles[group],physical_dims,physical_dims_proj);
		}

		boost::shared_ptr<hoCuNDArray<float> > out(new hoCuNDArray<float>);
		image.to_host(out.get());
		return out;

	}

	void load_data(std::string filename){
		hid_t file_id = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
		std::vector<std::string> groupnames = group_paths("/",file_id);

		size_t num_element = get_num_elements(file_id,groupnames);

		std::vector<size_t> dims;
		dims.push_back(num_element);



		std::vector<size_t> spline_dims;
		spline_dims.push_back(4);
		spline_dims.push_back(num_element);

		spline_data = boost::shared_ptr<hoCuNDArray<vector_td<float,3> > >(new hoCuNDArray<vector_td<float,3> >(spline_dims));
		projection_data =boost::shared_ptr<hoCuNDArray<float > >(new hoCuNDArray<float>(dims));

		spline_arrays = std::vector< boost::shared_ptr<hoCuNDArray<vector_td<float,3> > > >();
		projection_arrays=std::vector< boost::shared_ptr<hoCuNDArray<float > > >();

		angles = std::vector<float>();
		load_fromtable(file_id,groupnames);

	}





protected:

	boost::shared_ptr<hoCuNDArray<float> > calc_filter(size_t width){
		std::vector<size_t > filter_size;
		filter_size.push_back(width);
		boost::shared_ptr< hoCuNDArray<float> >  filter(new hoCuNDArray<float>(filter_size));

		float* filter_ptr = filter->get_data_ptr();

		for (size_t i = 0; i < width/2; i++)
			filter_ptr[i] = (width/2-float(i))/float(width/2);
		for (size_t i = 0; i < width/2; i++)
			filter_ptr[i+width/2] = (float(i))/float(width/2);

		return filter;


	}

	struct Spline{
		float x,y,z,x2,y2,z2;
		float dirx,diry,dirz,dirx2,diry2,dirz2;
	};



	size_t get_num_elements(hid_t file_id, std::vector<std::string>& groupnames){
		std::string projections_name = "projections";
		std::string splines_name = "splines";
		size_t total_elements = 0;

		for (int i = 0; i < groupnames.size(); i++){
			hsize_t nfields,nrecords,nrecords2;
			herr_t err = H5TBget_table_info (file_id, (groupnames[i]+projections_name).c_str(), &nfields, &nrecords );
			err = H5TBget_table_info (file_id, (groupnames[i]+splines_name).c_str(), &nfields, &nrecords2 );

			if (nrecords != nrecords2) throw std::runtime_error("Illegal data file: number of splines and projections do not match");
			total_elements += nrecords;
		}
		return total_elements;
	}

	void load_fromtable(hid_t file_id, std::vector<std::string>& groupnames){

		const size_t dst_sizes[12] = { sizeof(float) ,sizeof(float), sizeof(float),
				sizeof(float) ,sizeof(float), sizeof(float),
				sizeof(float) ,sizeof(float), sizeof(float),
				sizeof(float) ,sizeof(float), sizeof(float)};
		const size_t dst_offset[12] = { HOFFSET( Spline, x ),HOFFSET( Spline, y),HOFFSET( Spline, z ),
				HOFFSET( Spline, x2 ),HOFFSET( Spline, y2),HOFFSET( Spline, z2 ),
				HOFFSET( Spline, dirx ),HOFFSET( Spline, diry ),HOFFSET( Spline, dirz ),
				HOFFSET( Spline, dirx2 ),HOFFSET( Spline, diry2 ),HOFFSET( Spline, dirz2 )};

		std::string splines_name = "splines";



		const size_t float_size[1] = {sizeof(float) };
		const size_t float_offset[1] = {0};
		std::string projections_name = "projections";

		hid_t strtype;                     /* Datatype ID */
		herr_t status;


		size_t offset = 0;

		for (int i = 0; i < groupnames.size(); i++){
			hsize_t nfields,nrecords;
			herr_t err = H5TBget_table_info (file_id, (groupnames[i]+splines_name).c_str(), &nfields, &nrecords );
			if (err < 0) throw std::runtime_error("Illegal hdf5 dataset provided");
			std::vector<size_t> spline_dims;
			spline_dims.push_back(4);
			spline_dims.push_back(nrecords);
			boost::shared_ptr<hoCuNDArray<vector_td<float,3> > > splines(new hoCuNDArray<vector_td<float,3> >(spline_dims,spline_data->get_data_ptr()+offset*4));
			err = H5TBread_table (file_id, (groupnames[i]+splines_name).c_str(), sizeof(Spline),  dst_offset, dst_sizes,  splines->get_data_ptr());
			if (err < 0) throw std::runtime_error("Unable to read splines from hdf5 file");
			spline_arrays.push_back(splines);


			std::vector<size_t> proj_dims;
			proj_dims.push_back(nrecords);

			boost::shared_ptr<hoCuNDArray<float > > projections(new hoCuNDArray<float >(proj_dims,projection_data->get_data_ptr()+offset));
			err = H5TBread_table (file_id, (groupnames[i]+projections_name).c_str(), sizeof(float),  float_offset, float_size,  projections->get_data_ptr());
			if (err < 0) throw std::runtime_error("Unable to read projections from hdf5 file");
			projection_arrays.push_back(projections);
			offset += nrecords;

			float angle;
			err = H5LTget_attribute_float(file_id,groupnames[i].c_str(),"angle",&angle);
			if (err < 0 ) throw std::runtime_error("No angle provided in the group. Aborting");
			angles.push_back(angle);

		}
	}

	/***
	 * Returns a vector of strings for the paths.
	 * @param path
	 * @param file_id
	 * @return
	 */
	std::vector<std::string> group_paths(std::string path,hid_t file_id){

		char node[2048];
		hsize_t nobj,len;
		herr_t err;
		hid_t group_id = H5Gopen1(file_id,path.c_str());

		err = H5Gget_num_objs(group_id, &nobj);

		std::vector<std::string> result;
		for(hsize_t i =0; i < nobj; i++){
			len = H5Gget_objname_by_idx(group_id, i,
					node, sizeof(node) );
			std::string nodestr = std::string(path).append(node).append("/");
			int otype =  H5Gget_objtype_by_idx(group_id, i );
			switch(otype){
			case H5G_GROUP:
				//cout << nodestr << " is a GROUP" << endl;
				result.push_back(nodestr);
				break;
			}

		}
		H5Gclose(group_id);
		return result;

	}

	boost::shared_ptr<hoCuNDArray<vector_td<float,3> > > spline_data;
	boost::shared_ptr<hoCuNDArray<float > > projection_data;
	std::vector<float> angles;

	//These two vectors contain views of the original data. This way we can pass around individual projections and the complete dataset. Just for fun.
	std::vector< boost::shared_ptr<hoCuNDArray<float > > > projection_arrays;
	std::vector< boost::shared_ptr<hoCuNDArray<vector_td<float,3> > > > spline_arrays;
};
}
