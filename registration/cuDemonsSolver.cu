#include "Matrix.h"
#include "cuDemonsSolver.h"
#include "gpureg_export.h"
#include "morphon.h"
#include "setup_grid.h"
#include "vector_td_utilities.h"
#include <boost/make_shared.hpp>
#include <cuNDArray_fileio.h>
#include <cuPartialDerivativeOperator.h>
#include <numeric>
#include <thrust/extrema.h>
using namespace Gadgetron;


namespace {
    struct TextureObject {
        TextureObject(const cuNDArray<float>& image){


            if (image.dimensions().size() == 2){
                this->array = copy_to_2d_array(image);
            } else if (image.dimensions().size() == 3){
                this->array = copy_to_3d_array(image);
            }


            struct cudaResourceDesc resDesc;
            memset(&resDesc, 0, sizeof(resDesc));
            resDesc.resType         = cudaResourceTypeArray;
            resDesc.res.array.array = array;

            struct cudaTextureDesc texDesc;
            memset(&texDesc, 0, sizeof(texDesc));
            texDesc.addressMode[0]     = cudaAddressModeClamp;
            texDesc.addressMode[1]     = cudaAddressModeClamp;
            texDesc.addressMode[2]     = cudaAddressModeClamp;
            texDesc.filterMode         = cudaFilterModeLinear;
            texDesc.readMode           = cudaReadModeElementType;
            texDesc.normalizedCoords   = 0;
            cudaCreateTextureObject(&texture, &resDesc, &texDesc, NULL);
        }

        ~TextureObject(){
            cudaDestroyTextureObject(texture);
            cudaFreeArray(array);
        }





        cudaTextureObject_t texture;
        cudaArray* array;

    private:

        static cudaArray* copy_to_2d_array(const cuNDArray<float>& image){

            cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
            cudaExtent extent;
            extent.width  = image.get_size(0);
            extent.height = image.get_size(1);


            cudaArray* image_array;
            cudaMallocArray(&image_array,&channelDesc,extent.width,extent.height);
            size_t width = extent.width*sizeof(float);
            cudaMemcpy2DToArray(image_array,0,0,image.data(),width,width,image.get_size(1),cudaMemcpyDeviceToDevice);
            return image_array;
        }

        static cudaArray* copy_to_3d_array(const cuNDArray<float>& image){

            cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
            cudaExtent extent;
            extent.width  = image.get_size(0);
            extent.height = image.get_size(1);
            extent.depth  = image.get_size(2);

            cudaMemcpy3DParms cpy_params = { 0 };
            cpy_params.kind              = cudaMemcpyDeviceToDevice;
            cpy_params.extent            = extent;
            cudaArray* image_array;
            cudaMalloc3DArray(&image_array, &channelDesc, extent);
            cpy_params.dstArray = image_array;
            cpy_params.srcPtr
                = make_cudaPitchedPtr((void*)image.data(), extent.width * sizeof(float), extent.width, extent.height);
            cudaMemcpy3D(&cpy_params);

            return image_array;
        }



    };


    template <int D> struct vsquarefunctor {};

    template <> struct vsquarefunctor<3> : public thrust::unary_function<thrust::tuple<float, float, float>, float> {
        __host__ __device__ float operator()(thrust::tuple<float, float, float> tup) {
            float x = thrust::get<0>(tup);
            float y = thrust::get<1>(tup);
            float z = thrust::get<2>(tup);
            return x * x + y * y + z * z;
        }
    };

    template <> struct vsquarefunctor<2> : public thrust::unary_function<thrust::tuple<float, float>, float> {
        __host__ __device__ float operator()(thrust::tuple<float, float> tup) {
            float x = thrust::get<0>(tup);
            float y = thrust::get<1>(tup);
            return x * x + y * y;
        }
    };

    void setup_image_grid(dim3& threads, dim3& grid, const vector_td<int,3>& dims){
        threads = dim3(8, 8, 8);
        grid = ((dims[0] + threads.x - 1) / threads.x, (dims[1] + threads.y - 1) / threads.y,
                  (dims[2] + threads.z - 1) / threads.z);
    }

    void setup_image_grid(dim3& threads, dim3& grid, const vector_td<int,2>& dims){
        threads = dim3(16, 16);
        grid = dim3((dims[0] + threads.x - 1) / threads.x, (dims[1] + threads.y - 1) / threads.y);
    }


}



static void vfield_exponential(cuNDArray<float>& vfield) {

    auto dims3D = vfield.dimensions();
    dims3D.pop_back();

    size_t elements = std::accumulate(dims3D.begin(),dims3D.end(),1,std::multiplies<size_t>());

    cuNDArray<float> xview(dims3D, vfield.data());
    cuNDArray<float> yview(dims3D, vfield.data() + elements);

    int n;
    if (vfield.dimensions().back() == 3) {
        cuNDArray<float> zview(dims3D, vfield.data() + elements * 2);

        auto iter     = thrust::make_zip_iterator(thrust::make_tuple(xview.begin(), yview.begin(), zview.begin()));
        auto iter_end = thrust::make_zip_iterator(thrust::make_tuple(xview.end(), yview.end(), zview.end()));
        auto msquare  = thrust::max_element(thrust::make_transform_iterator(iter, vsquarefunctor<3>()),
            thrust::make_transform_iterator(iter_end, vsquarefunctor<3>()));

        n = ceil(2 + log2(sqrt(*msquare) / 0.5));

        n = std::max(n, 0);
        std::cout << " N " << n << " " << float(std::pow(2, -float(n))) << " " << sqrt(*msquare) << std::endl;
    } else {
        auto iter     = thrust::make_zip_iterator(thrust::make_tuple(xview.begin(), yview.begin()));
        auto iter_end = thrust::make_zip_iterator(thrust::make_tuple(xview.end(), yview.end()));
        auto msquare  = thrust::max_element(thrust::make_transform_iterator(iter, vsquarefunctor<2>()),
            thrust::make_transform_iterator(iter_end, vsquarefunctor<2>()));

        n = ceil(2 + log2(sqrt(*msquare) / 0.5));

        n = std::max(n, 0);
        std::cout << " N " << n << " " << float(std::pow(2, -float(n))) << " " << sqrt(*msquare) << std::endl;
    }
    vfield *= float(std::pow(2, -float(n)));

    for (int i = 0; i < n; i++) {
        cuNDArray<float> vfield_copy(vfield);
        deform_vfield(vfield, vfield_copy);
        vfield += vfield_copy;
    }
}

template <class T, unsigned int D>
static inline __device__ vector_td<T, D> partialDerivs(
    const T* in, const vector_td<int, D>& dims, const vector_td<int, D>& co) {

    vector_td<int, D> co2 = co;
    vector_td<T, D> out;
    // T xi = in[co_to_idx<D>(co,dims)];
    for (int i = 0; i < D; i++) {
        co2[i] = min(co[i] + 1, dims[i] - 1);
        T dt   = in[co_to_idx<D>(co2, dims)];
        co2[i] = max(co[i] - 1, 0);
        T xi   = in[co_to_idx<D>(co2, dims)];
        out[i] = (dt - xi) * 0.5f;
        co2[i] = co[i];
    }
    return out;
}

/***
 *
 * @param fixed The fixed image
 * @param moving The Moving image
 * @param tot_elemens Total number of elements in fixed (and moving)
 * @param dims Dimensions of the subspace into which the convolution needs to be done
 * @param out Output vector field. Must have same dimensions as fixed and moving + an additional D dimension
 * @param alpha Regularization weight
 * @param beta Small constant added to prevent division by 0.
 */

template <class T, unsigned int D>
static __global__ void demons_kernel(
    T* fixed, T* moving, size_t tot_elements, const vector_td<int, D> dims, T* out, T alpha, T beta) {

    const int idx = blockIdx.y * gridDim.x * blockDim.x + blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < tot_elements) {

        size_t elements = prod(dims);

        int batch       = idx / elements;
        T* fixed_batch  = fixed + elements * batch;
        T* moving_batch = moving + elements * batch;

        vector_td<int, D> co = idx_to_co<D>(idx, dims);

        auto dfix = partialDerivs(fixed_batch, dims, co);
        auto dmov = partialDerivs(moving_batch, dims, co);
        T It      = fixed_batch[idx] - moving_batch[idx];

        dmov += dfix;
        dmov *= 0.5f;
        T gradNorm = norm_squared(dmov);

        vector_td<T, D> res;
        for (int i = 0; i < D; i++) {
            res[i] = It * (dmov[i]) / (gradNorm + alpha * alpha * It * It + beta);
        }
        for (int i = 0; i < D; i++) {
            out[idx + i * tot_elements] = res[i];
        }
    }
}

template <class T, unsigned int D>
cuNDArray<T> cuDemonsSolver<T, D>::demonicStep(cuNDArray<T>& fixed, cuNDArray<T>& moving) {

    std::vector<size_t> dims = fixed.dimensions();
    dims.push_back(D);

    vector_td<int, D> idims = vector_td<int, D>(from_std_vector<size_t, D>(dims));
    cuNDArray<T> out(&dims);
    clear(&out);

    dim3 gridDim;
    dim3 blockDim;
    setup_grid(fixed.size(), &blockDim, &gridDim);

    demons_kernel<T, D><<<gridDim, blockDim>>>(
        fixed.data(), moving.data(), fixed.get_number_of_elements(), idims, out.data(), 1.0 / alpha, beta);

    return out;
}

namespace {
    __device__ bool check_dimensions(const vector_td<int, 1>& dims, int x, int y, int z) {
        return dims[0] > x;
    }
    __device__ bool check_dimensions(const vector_td<int, 2>& dims, int x, int y, int z) {
        return dims[0] > x && dims[1] > y;
    }
    __device__ bool check_dimensions(const vector_td<int, 3>& dims, int x, int y, int z) {
        return dims[0] > x && dims[1] > y && dims[2] > z;
    }

    template <int D> __device__ vector_td<int, D> make_vector_td(int x, int y, int z);
    template <> __device__ vector_td<int, 3> make_vector_td<3>(int x, int y, int z) {
        return vector_td<int, 3>(x, y, z);
    }
    template <> __device__ vector_td<int, 2> make_vector_td<2>(int x, int y, int z) {
        return vector_td<int, 2>(x, y);
    }

    template <int... Ds>
    __device__ auto load_helper(const float* data, int elements, std::integer_sequence<int, Ds...>) {
        return vector_td<float, sizeof...(Ds)>(data[Ds * elements]...);
    }

    template <unsigned int D> __device__ vector_td<float, D> load_vector_from_SOA(const float* data, int elements) {
        return load_helper(data, elements, std::make_integer_sequence<int, D>());
    }

}

template <class T, unsigned int D>
static __global__ void NGF_kernel(T* image, size_t tot_elements, const vector_td<int, D> dims, T* out, T eps) {

    const int idx = blockIdx.y * gridDim.x * blockDim.x + blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < tot_elements) {
        vector_td<int, D> co = idx_to_co<D>(idx, dims);

        auto diff = partialDerivs(image, dims, co);

        T dnorm = sqrt(norm_squared(diff) + eps);
        for (int i = 0; i < D; i++) {
            out[idx + i * tot_elements] = abs(diff[i]) / dnorm;
        }
    }
}

template <class T, unsigned int D> static cuNDArray<T> normalized_gradient_field(cuNDArray<T>& image, T eps) {

    std::vector<size_t> dims = image.dimensions();
    dims.push_back(D);

    vector_td<int, D> idims = vector_td<int, D>(from_std_vector<size_t, D>(dims));
    cuNDArray<T> out(&dims);
    clear(&out);

    dim3 gridDim;
    dim3 blockDim;
    setup_grid(image.size(), &blockDim, &gridDim);

    NGF_kernel<T, D><<<gridDim, blockDim>>>(image.data(), image.get_number_of_elements(), idims, out.data(), eps);

    return out;
}





template<class T, int D>
__device__ Matrix<T,D,D> hessian(const T* derivatives, const vector_td<int,D>& dims, const vector_td<int,D>& co){
    Matrix<T, D,D> H;

    auto elements = prod(dims);
    for (int i = 0; i < D; i++){
        H.data[i] = partialDerivs(derivatives+i*elements,dims,co );
    }
    return H;
}



template<class T, unsigned int D>
static __global__ void NGF_step_kernel(T* out, const T* L_moving, const T* L_fixed, vector_td<int,D> dims, T eps){


    const int ixo = blockDim.x * blockIdx.x + threadIdx.x;
    const int iyo = blockDim.y * blockIdx.y + threadIdx.y;
    const int izo = blockDim.z * blockIdx.z + threadIdx.z;


    if (check_dimensions(dims,ixo,iyo,izo)){
        auto elements = prod(dims);
        auto co = make_vector_td<D>(ixo,iyo,izo);
        auto idx = co_to_idx(co,dims);

        auto H = hessian<T,D>(L_moving,dims,co);



        vector_td<T,D> grad_fixed = load_vector_from_SOA<D>(L_fixed+idx,elements);
        vector_td<T,D> grad_moving = load_vector_from_SOA<D>(L_moving+idx,elements);

        auto fixed_norm = sqrt(norm_squared(grad_fixed)+eps);
        auto moving_norm = sqrt(norm_squared(grad_moving)+eps);

        grad_fixed /= fixed_norm;
        grad_moving /= moving_norm;



        auto fg = dot(grad_fixed,grad_moving);

        auto reprojected = grad_fixed-fg*grad_moving;
        auto result = -2*fg * dot(H,reprojected)/moving_norm;

        for (int i = 0; i < D; i++) {
            out[idx + i * elements] = result[i];

        }

    }

}



template<class T, unsigned int D>
static __global__ void laplacian_kernel(T* out, const T* image, vector_td<int, D> dims){

    const int ixo = blockDim.x * blockIdx.x + threadIdx.x;
    const int iyo = blockDim.y * blockIdx.y + threadIdx.y;
    const int izo = blockDim.z * blockIdx.z + threadIdx.z;


    if (check_dimensions(dims,ixo,iyo,izo)) {
        auto co = make_vector_td<D>(ixo,iyo,izo);
        auto elements = prod(dims);
        auto idx = co_to_idx(co,dims);
        auto L = partialDerivs(image,dims,co);

        for (int i = 0; i < D; i++) {
            out[idx+i*elements] = L[i];
        }


    }

}


template<class T, unsigned int D>
static cuNDArray<T> laplacian(const cuNDArray<T>& image){

    auto dims = image.dimensions();
    dims.push_back(D);
    auto result = cuNDArray<T>(dims);
    dim3 threads;
    dim3 grid;

    auto vdims = vector_td<int,D>(from_std_vector<size_t,D>(image.dimensions()));
    setup_image_grid(threads,grid,vdims);
    laplacian_kernel<<<grid,threads>>>(result.data(),image.data(),vdims);
    return result;
}

template<class T, unsigned int D>
static cuNDArray<T> NGF_step(const cuNDArray<T>& fixed, const cuNDArray<T>& moving,T eps){

    auto L_fixed = laplacian<T,D>(fixed);
    auto L_moving = laplacian<T,D>(moving);
    nrm2(&L_fixed);
    nrm2(&L_moving);

    write_nd_array(L_fixed.to_host().get(),"L_fixed.real");

    cudaDeviceSynchronize();
    dim3 threads,grid;
    auto vdims = vector_td<int,D>(from_std_vector<size_t,D>(fixed.dimensions()));

    setup_image_grid(threads,grid,vdims);

    auto dims = fixed.dimensions();
    dims.push_back(D);
    auto result = cuNDArray<T>(dims);

    NGF_step_kernel<<<grid,threads>>>(result.data(),L_moving.data(),L_fixed.data(),vdims,eps);
    cudaDeviceSynchronize();

    std::cout << " Result Norm " << nrm2(&result) << std::endl;
    return result;

}


template <class T, unsigned int D>
cuNDArray<T> cuDemonsSolver<T, D>::registration(cuNDArray<T>& fixed_image, cuNDArray<T>& moving_image) {

    auto vdims = fixed_image.dimensions();
    vdims.push_back(D);
    auto result = cuNDArray<T>(vdims);
    clear(&result);
    single_level_reg(fixed_image, moving_image, result);

    return result;
}

template <class T, unsigned int D>
cuNDArray<T> cuDemonsSolver<T, D>::multi_level_reg(
    cuNDArray<T>& fixed_image, cuNDArray<T>& moving_image, int levels, float scale) {

    std::cout << "Level " << levels << std::endl;
    auto vdims = fixed_image.dimensions();
    vdims.push_back(D);
    auto result = cuNDArray<T>(vdims);
    clear(&result);
    if (levels <= 1) {
        single_level_reg(fixed_image, moving_image, result, scale);
    } else {

        auto d_fixed  = downsample<T, D>(fixed_image);
        auto d_moving = downsample<T, D>(moving_image);
        auto tmp_res  = multi_level_reg(d_fixed, d_moving, levels - 1, scale / 2);

        auto dims = tmp_res.dimensions();
        for (auto d : dims)
            std::cout << d << " ";
        std::cout << std::endl;

        upscale_vfield(tmp_res, result);
        result *= T(2);

        single_level_reg(fixed_image, moving_image, result, scale);
    }

    return result;
}

template<class T, unsigned int D>
void vector_gauss(cuGaussianFilterOperator<T,D>& op ,cuNDArray<T>& vfield){
    for (int i = 0; i < D; i++){
        auto view_dims = vfield.dimensions();
        view_dims.pop_back();
        auto elements = vfield.size()/vfield.dimensions().back();
        cuNDArray<T> view(view_dims,vfield.data()+i*elements);
        auto copy = view;
        op.mult_M(&copy,&view);

    }
}

template <class T, unsigned int D>
void cuDemonsSolver<T, D>::single_level_reg(
    cuNDArray<T>& fixed_image, cuNDArray<T>& moving_image, cuNDArray<T>& result, float scale) {

    auto vdims = fixed_image.dimensions();
    vdims.push_back(D);


    auto texture = TextureObject(moving_image);

    cuGaussianFilterOperator<T, D> gaussDiff;
    gaussDiff.set_sigma(sigmaDiff * T(scale));

    cuGaussianFilterOperator<T, D> gaussFluid;
    gaussFluid.set_sigma(sigmaFluid * scale);

    std::vector<size_t> image_dims = moving_image.dimensions();
    std::vector<size_t> dims       = moving_image.dimensions();

    auto def_moving = deform_image(texture.texture, image_dims, result);

    dims.push_back(D);

    for (int i = 0; i < iterations; i++) {
        // Calculate the gradients
        cuNDArray<T> update;
        if (epsilonNGF > 0) {
            auto diff_fixed  = normalized_gradient_field<T, D>(fixed_image, epsilonNGF);
            auto diff_moving = normalized_gradient_field<T, D>(def_moving, epsilonNGF);
            update           = cuNDArray<T>(diff_fixed.dimensions());
            clear(&update);
            size_t elements = fixed_image.size();
            auto dims       = fixed_image.dimensions();

            for (int d = 0; d < D; d++) {
                cuNDArray<T> f_view(dims, diff_fixed.data() + d * elements);
                cuNDArray<T> m_view(dims, diff_moving.data() + d * elements);
                update += demonicStep(f_view, m_view);
            }
            update /= T(D);
//            update = NGF_step<T,D>(fixed_image,moving_image,epsilonNGF);
//            update *= 0.1f;
//            update *= this->alpha;
            std::cout << "Update NGF norm " << nrm2(&update) << std::endl;

        } else {
            update = demonicStep(fixed_image, def_moving);
            // update = morphon(&def_moving,fixed_image);
            std::cout << "Updated norm " << nrm2(&update) << std::endl;
        }
        if (sigmaFluid > 0) {
            vector_gauss(gaussFluid, update);
            // blurred_update = *update;
        }


        if (exponential)
            vfield_exponential(update);
        if (compositive)
            deform_vfield(result,update );
        result += update;


        if (sigmaDiff > vector_td<T, D>(0)) {
            if (sigmaInt > 0 || sigmaVDiff > 0)
                bilateral_vfield(result, def_moving, sigmaDiff * T(scale), sigmaInt, sigmaVDiff);
            else {

                vector_gauss(gaussDiff,result);
            }
        }
        //*result -= tmp;

        // if (exponential) vfield_exponential(result);
        //*result += tmp;

        def_moving = deform_image(texture.texture, image_dims, result);
        {
            cuNDArray<T> tmp = fixed_image;
            tmp -= def_moving;
            std::cout << "Diff " << nrm2(&tmp) << std::endl;
        }
    }
}

// Simple transformation kernel
__global__ static void deform_vfieldKernel3D(float* output, const float* vector_field, cudaTextureObject_t texX,
    cudaTextureObject_t texY, cudaTextureObject_t texZ, int width, int height, int depth) {

    const int ixo = blockDim.x * blockIdx.x + threadIdx.x;
    const int iyo = blockDim.y * blockIdx.y + threadIdx.y;
    const int izo = blockDim.z * blockIdx.z + threadIdx.z;

    if (ixo < width && iyo < height && izo < depth) {

        const int idx      = ixo + iyo * width + izo * width * height;
        const int elements = width * height * depth;
        float ux           = vector_field[idx] + 0.5f + ixo;
        float uy           = vector_field[idx + elements] + 0.5f + iyo;
        float uz           = vector_field[idx + 2 * elements] + 0.5f + izo;

        output[idx] = tex3D<float>(texX, ux, uy, uz);

        output[idx + elements]     = tex3D<float>(texY, ux, uy, uz);
        output[idx + 2 * elements] = tex3D<float>(texZ, ux, uy, uz);
    }
}

__global__ static void deform_vfieldKernel2D(float* output, const float* vector_field, cudaTextureObject_t texX,
    cudaTextureObject_t texY,  int width, int height) {

    const int ixo = blockDim.x * blockIdx.x + threadIdx.x;
    const int iyo = blockDim.y * blockIdx.y + threadIdx.y;

    if (ixo < width && iyo < height) {

        const int idx      = ixo + iyo * width;
        const int elements = width * height;
        float ux           = vector_field[idx] + 0.5f + ixo;
        float uy           = vector_field[idx + elements] + 0.5f + iyo;

        output[idx]            = tex2D<float>(texX, ux, uy);
        output[idx + elements] = tex2D<float>(texY, ux, uy);
    }
}

// Simple transformation kernel
__global__ static void deform_imageKernel3D(
    float* output, const float* vector_field, cudaTextureObject_t texObj, int width, int height, int depth) {

    const int ixo = blockDim.x * blockIdx.x + threadIdx.x;
    const int iyo = blockDim.y * blockIdx.y + threadIdx.y;
    const int izo = blockDim.z * blockIdx.z + threadIdx.z;

    if (ixo < width && iyo < height && izo < depth) {

        const int idx      = ixo + iyo * width + izo * width * height;
        const int elements = width * height * depth;
        float ux           = vector_field[idx] + 0.5f + ixo;
        float uy           = vector_field[idx + elements] + 0.5f + iyo;
        float uz           = vector_field[idx + 2 * elements] + 0.5f + izo;

        output[idx] = tex3D<float>(texObj, ux, uy, uz);
    }
}

// Simple transformation kernel
__global__ static void deform_imageKernel2D(
    float* output, const float* vector_field, cudaTextureObject_t texObj, int width, int height) {

    const int ixo = blockDim.x * blockIdx.x + threadIdx.x;
    const int iyo = blockDim.y * blockIdx.y + threadIdx.y;

    if (ixo < width && iyo < height) {
        const int idx      = ixo + iyo * width;
        const int elements = width * height;
        float ux           = vector_field[idx] + 0.5f + ixo;
        float uy           = vector_field[idx + elements] + 0.5f + iyo;
        output[idx]        = tex2D<float>(texObj, ux, uy);
    }
}
cuNDArray<float> Gadgetron::deform_image(
    cudaTextureObject_t texObj, const std::vector<size_t>& dimensions, cuNDArray<float>& vector_field) {

    cuNDArray<float> output(dimensions);
    clear(&output);

    if (vector_field.dimensions().back() == 3) {
        dim3 threads(8, 8, 8);
        dim3 grid((dimensions[0] + threads.x - 1) / threads.x, (dimensions[1] + threads.y - 1) / threads.y,
            (dimensions[2] + threads.z - 1) / threads.z);
        deform_imageKernel3D<<<grid, threads>>>(
            output.data(), vector_field.data(), texObj, dimensions[0], dimensions[1], dimensions[2]);
        return output;
    } else if (vector_field.dimensions().back() == 2) {
        dim3 threads(16, 16);
        dim3 grid((dimensions[0] + threads.x - 1) / threads.x, (dimensions[1] + threads.y - 1) / threads.y);
        deform_imageKernel2D<<<grid, threads>>>(
            output.data(), vector_field.data(), texObj, dimensions[0], dimensions[1]);
        return output;
    } else {
        throw std::runtime_error("Illegal number of dimensions");
    }
}

namespace {
    template <unsigned int D> void deform_vfield_internal(cuNDArray<float>& vfield1, const cuNDArray<float>& vector_field);

    template <> void deform_vfield_internal<2>( cuNDArray<float>& vfield1, const cuNDArray<float>& vector_field) {

        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
        cudaExtent extent;
        extent.width  = vfield1.get_size(0);
        extent.height = vfield1.get_size(1);
        extent.depth  = vfield1.get_size(2);

        size_t elements = extent.height * extent.width;


        cudaArray* x_array;
        cudaArray* y_array;
        cudaMallocArray(&x_array, &channelDesc, vfield1.get_size(0),vfield1.get_size(1));
        cudaMallocArray(&y_array, &channelDesc, vfield1.get_size(0),vfield1.get_size(1));


        size_t width = vfield1.get_size(0)*sizeof(float);
        cudaMemcpy2DToArray(x_array,0,0,vfield1.data(),width,width,vfield1.get_size(1),cudaMemcpyDeviceToDevice);
        cudaMemcpy2DToArray(y_array,0,0,vfield1.data()+elements,width,width,vfield1.get_size(1),cudaMemcpyDeviceToDevice);


        struct cudaResourceDesc resDescX;
        memset(&resDescX, 0, sizeof(resDescX));
        resDescX.resType         = cudaResourceTypeArray;
        resDescX.res.array.array = x_array;
        struct cudaTextureDesc texDesc;
        memset(&texDesc, 0, sizeof(texDesc));
        texDesc.addressMode[0]   = cudaAddressModeClamp;
        texDesc.addressMode[1]   = cudaAddressModeClamp;
        texDesc.addressMode[2]   = cudaAddressModeClamp;
        texDesc.filterMode       = cudaFilterModeLinear;
        texDesc.readMode         = cudaReadModeElementType;
        texDesc.normalizedCoords = 0;
        cudaTextureObject_t texX = 0;
        cudaCreateTextureObject(&texX, &resDescX, &texDesc, NULL);

        struct cudaResourceDesc resDescY;
        memset(&resDescY, 0, sizeof(resDescY));
        resDescY.resType         = cudaResourceTypeArray;
        resDescY.res.array.array = y_array;

        cudaTextureObject_t texY = 0;
        cudaCreateTextureObject(&texY, &resDescY, &texDesc, NULL);


        cudaDeviceSynchronize();
        dim3 threads(16, 16);

        dim3 grid((extent.width + threads.x - 1) / threads.x, (extent.height + threads.y - 1) / threads.y);

        deform_vfieldKernel2D<<<grid, threads>>>(
            vfield1.data(), vector_field.data(), texX, texY,  extent.width, extent.height);

        cudaDeviceSynchronize();
        cudaDestroyTextureObject(texX);
        cudaDestroyTextureObject(texY);

        // Free device memory
        cudaFreeArray(x_array);
        cudaFreeArray(y_array);
    }
    template <> void deform_vfield_internal<3>(cuNDArray<float>& vfield1, const cuNDArray<float>& vector_field) {

        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
        cudaExtent extent;
        extent.width  = vfield1.get_size(0);
        extent.height = vfield1.get_size(1);
        extent.depth  = vfield1.get_size(2);

        size_t elements = extent.height * extent.depth * extent.width;

        cudaMemcpy3DParms cpy_params = { 0 };
        cpy_params.kind              = cudaMemcpyDeviceToDevice;
        cpy_params.extent            = extent;

        cudaArray* x_array;
        cudaArray* y_array;
        cudaArray* z_array;
        cudaMalloc3DArray(&x_array, &channelDesc, extent);
        cudaMalloc3DArray(&y_array, &channelDesc, extent);
        cudaMalloc3DArray(&z_array, &channelDesc, extent);

        // Copy x, y and z coordinates into their own textures.
        cpy_params.dstArray = x_array;
        cpy_params.srcPtr
            = make_cudaPitchedPtr((void*)vfield1.data(), extent.width * sizeof(float), extent.width, extent.height);
        cudaMemcpy3D(&cpy_params);
        cpy_params.dstArray = y_array;
        cpy_params.srcPtr   = make_cudaPitchedPtr(
            (void*)(vfield1.data() + elements), extent.width * sizeof(float), extent.width, extent.height);
        cudaMemcpy3D(&cpy_params);
        cpy_params.dstArray = z_array;
        cpy_params.srcPtr   = make_cudaPitchedPtr(
            (void*)(vfield1.data() + 2 * elements), extent.width * sizeof(float), extent.width, extent.height);
        cudaMemcpy3D(&cpy_params);

        struct cudaResourceDesc resDescX;
        memset(&resDescX, 0, sizeof(resDescX));
        resDescX.resType         = cudaResourceTypeArray;
        resDescX.res.array.array = x_array;
        struct cudaTextureDesc texDesc;
        memset(&texDesc, 0, sizeof(texDesc));
        texDesc.addressMode[0]   = cudaAddressModeClamp;
        texDesc.addressMode[1]   = cudaAddressModeClamp;
        texDesc.addressMode[2]   = cudaAddressModeClamp;
        texDesc.filterMode       = cudaFilterModeLinear;
        texDesc.readMode         = cudaReadModeElementType;
        texDesc.normalizedCoords = 0;
        cudaTextureObject_t texX = 0;
        cudaCreateTextureObject(&texX, &resDescX, &texDesc, NULL);

        struct cudaResourceDesc resDescY;
        memset(&resDescY, 0, sizeof(resDescY));
        resDescY.resType         = cudaResourceTypeArray;
        resDescY.res.array.array = y_array;

        cudaTextureObject_t texY = 0;
        cudaCreateTextureObject(&texY, &resDescY, &texDesc, NULL);

        struct cudaResourceDesc resDescZ;
        memset(&resDescZ, 0, sizeof(resDescZ));
        resDescZ.resType         = cudaResourceTypeArray;
        resDescZ.res.array.array = z_array;

        cudaTextureObject_t texZ = 0;
        cudaCreateTextureObject(&texZ, &resDescZ, &texDesc, NULL);

        cudaDeviceSynchronize();
        dim3 threads(8, 8, 8);

        dim3 grid((extent.width + threads.x - 1) / threads.x, (extent.height + threads.y - 1) / threads.y,
            (extent.depth + threads.z - 1) / threads.z);

        deform_vfieldKernel3D<<<grid, threads>>>(
            vfield1.data(), vector_field.data(), texX, texY, texZ, extent.width, extent.height, extent.depth);

        cudaDeviceSynchronize();
        cudaDestroyTextureObject(texX);
        cudaDestroyTextureObject(texY);
        cudaDestroyTextureObject(texZ);

        // Free device memory
        cudaFreeArray(x_array);
        cudaFreeArray(y_array);
        cudaFreeArray(z_array);
    }

}

void Gadgetron::deform_vfield(cuNDArray<float>& vfield1, const cuNDArray<float>& vector_field) {

    if (vfield1.dimensions().size() == 4)
        return deform_vfield_internal<3>(vfield1, vector_field);
    if (vfield1.dimensions().size() == 3)
        return deform_vfield_internal<2>(vfield1, vector_field);

    throw std::runtime_error("Illegal number of dimensions");
}

cuNDArray<float> Gadgetron::deform_image(cuNDArray<float>& image, cuNDArray<float>& vector_field) {

    auto tex = TextureObject(image);

    return deform_image(tex.texture,image.dimensions(),vector_field);
}

// Simple transformation kernel
__global__ static void upscale_vfieldKernel(
    float* output, cudaTextureObject_t texObj, int width, int height, int depth) {

    const int ixo = blockDim.x * blockIdx.x + threadIdx.x;
    const int iyo = blockDim.y * blockIdx.y + threadIdx.y;
    const int izo = blockDim.z * blockIdx.z + threadIdx.z;

    if (ixo < width && iyo < height && izo < depth) {

        const int idx = ixo + iyo * width + izo * width * height;
        float ux      = 0.5f + ixo;
        float uy      = 0.5f + iyo;
        float uz      = 0.5f + izo;

        output[idx] = tex3D<float>(texObj, ux / width, uy / height, uz / depth);
    }
}

template <class T, unsigned int D> void cuDemonsSolver<T, D>::upscale_vfield(cuNDArray<T>& in, cuNDArray<T>& out) {

    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    cudaExtent extent;
    extent.width  = in.get_size(0);
    extent.height = in.get_size(1);
    extent.depth  = in.get_size(2);

    size_t elements              = extent.depth * extent.height * extent.height;
    cudaMemcpy3DParms cpy_params = { 0 };
    cpy_params.kind              = cudaMemcpyDeviceToDevice;
    cpy_params.extent            = extent;

    cudaArray* image_array;
    cudaMalloc3DArray(&image_array, &channelDesc, extent);

    for (int i = 0; i < in.get_size(3); i++) {
        cpy_params.dstArray = image_array;
        cpy_params.srcPtr   = make_cudaPitchedPtr(
            (void*)(in.data() + i * elements), extent.width * sizeof(float), extent.width, extent.height);
        cudaMemcpy3D(&cpy_params);

        struct cudaResourceDesc resDesc;
        memset(&resDesc, 0, sizeof(resDesc));
        resDesc.resType         = cudaResourceTypeArray;
        resDesc.res.array.array = image_array;

        struct cudaTextureDesc texDesc;
        memset(&texDesc, 0, sizeof(texDesc));
        texDesc.addressMode[0]     = cudaAddressModeClamp;
        texDesc.addressMode[1]     = cudaAddressModeClamp;
        texDesc.addressMode[2]     = cudaAddressModeClamp;
        texDesc.filterMode         = cudaFilterModeLinear;
        texDesc.readMode           = cudaReadModeElementType;
        texDesc.normalizedCoords   = 1;
        cudaTextureObject_t texObj = 0;
        cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);

        dim3 threads(8, 8, 8);

        dim3 grid((out.get_size(0) + threads.x - 1) / threads.x, (out.get_size(1) + threads.y - 1) / threads.y,
            (out.get_size(2) + threads.z - 1) / threads.z);

        upscale_vfieldKernel<<<grid, threads>>>(
            out.data() + i * elements, texObj, out.get_size(0), out.get_size(1), out.get_size(2));

        cudaDestroyTextureObject(texObj);
    }
    cudaFreeArray(image_array);
    // Free device memory
}

static __global__ void jacobian_kernel(
    float* __restrict__ image, const vector_td<int, 3> dims, float* __restrict__ out) {

    const int idx = blockIdx.y * gridDim.x * blockDim.x + blockIdx.x * blockDim.x + threadIdx.x;
    int elements  = prod(dims);

    if (idx < elements) {

        vector_td<int, 3> co = idx_to_co<3>(idx, dims);

        auto dX = partialDerivs(image, dims, co);
        auto dY = partialDerivs(image + elements, dims, co);
        auto dZ = partialDerivs(image + elements * 2, dims, co);

        out[idx] = (1 + dX[0]) * (1 + dY[1]) * (1 + dZ[2]) + (1 + dX[0]) * dY[2] * dZ[1] + dX[1] * dY[0] * (1 + dZ[2])
                   + dX[2] * dY[0] * dZ[1] + dX[1] * dY[2] * dZ[0] + dX[2] * (1 + dY[1]) * dZ[0];
    }
}

static __global__ void jacobian_kernel(
    float* __restrict__ image, const vector_td<int, 2> dims, float* __restrict__ out) {

    const int idx = blockIdx.y * gridDim.x * blockDim.x + blockIdx.x * blockDim.x + threadIdx.x;
    int elements  = prod(dims);

    if (idx < elements) {
        vector_td<int, 2> co = idx_to_co<2>(idx, dims);
        auto dX              = partialDerivs(image, dims, co);
        auto dY              = partialDerivs(image + elements, dims, co);
        out[idx]             = (1 + dX[0]) * (1 + dY[1]) + dX[1] * dY[0];
    }
}

cuNDArray<float> Gadgetron::Jacobian(cuNDArray<float>& vfield) {

    std::vector<size_t> dims = *vfield.get_dimensions();
    int num_dims             = dims.back();
    dims.pop_back();
    cuNDArray<float> out(dims);

    dim3 gridDim;
    dim3 blockDim;
    setup_grid(out.get_number_of_elements(), &blockDim, &gridDim);

    if (num_dims == 3) {
        vector_td<int, 3> idims = vector_td<int, 3>(from_std_vector<size_t, 3>(dims));
        jacobian_kernel<<<gridDim, blockDim>>>(vfield.data(), idims, out.data());
    } else if (num_dims == 2) {
        vector_td<int, 2> idims = vector_td<int, 2>(from_std_vector<size_t, 2>(dims));
        jacobian_kernel<<<gridDim, blockDim>>>(vfield.data(), idims, out.data());
    } else {
        throw std::runtime_error("Unsuported number of dimensions");
    }

    return out;
}

static __global__ void bilateral_kernel3D(float* __restrict__ out, const float* __restrict__ vfield,
    const float* __restrict__ image, int width, int height, int depth, floatd3 sigma_spatial, float sigma_int,
    float sigma_diff) {

    const int ixo = blockDim.x * blockIdx.x + threadIdx.x;
    const int iyo = blockDim.y * blockIdx.y + threadIdx.y;
    const int izo = blockDim.z * blockIdx.z + threadIdx.z;

    const int idx = ixo + iyo * width + izo * width * height;
    int elements  = width * height * depth;

    if (ixo < width && iyo < height && izo < depth) {

        int steps = 8;

        vector_td<float, 3> vec(vfield[idx], vfield[idx + elements], vfield[idx + elements * 2]);

        vector_td<float, 3> res(0);
        float image_value = image[idx];
        float norm        = 0;
        for (int dz = -steps; dz <= steps; dz++) {
            int z = (izo + dz + depth) % depth;

            for (int dy = -steps; dy <= steps; dy++) {
                int y = (iyo + dy + height) % height;
                for (int dx = -steps; dx <= steps; dx++) {
                    int x = (ixo + dx + width) % width;

                    const int idx2 = x + y * width + z * width * height;

                    vector_td<float, 3> vec2(vfield[idx2], vfield[idx2 + elements], vfield[idx2 + elements * 2]);

                    float image_diff = image_value - image[idx2];
                    float vdiff      = (vec2[0] - vec[0]) * (vec2[0] - vec[0]) + (vec2[1] - vec[1]) * (vec2[1] - vec[1])
                                  + (vec2[2] - vec[2]) * (vec2[2] - vec[2]);
                    float weight = expf(-float(dx * dx) / (2 * sigma_spatial[0] * sigma_spatial[0])
                                        - float(dy * dy) / (2 * sigma_spatial[1] * sigma_spatial[1])
                                        - float(dz * dz) / (2 * sigma_spatial[2] * sigma_spatial[2])
                                        - image_diff * image_diff / (2 * sigma_int * sigma_int)
                                        - vdiff / (2 * sigma_diff * sigma_diff));
                    norm += weight;
                    res[0] += weight * vec2[0];
                    res[1] += weight * vec2[1];
                    res[2] += weight * vec2[2];
                }
            }
        }

        int idx                 = ixo + iyo * width + izo * width * height;
        int elements            = height * width * depth;
        out[idx]                = res[0] / norm;
        out[idx + elements]     = res[1] / norm;
        out[idx + 2 * elements] = res[2] / norm;
    }
}



template <int D, int N>
static __global__ void bilateral_kernel1D(float* __restrict__ out, const float* __restrict__ vfield,
    const float* __restrict__ image, vector_td<int, D> dims, float sigma_spatial, float sigma_int, float sigma_diff) {

    const int ixo = blockDim.x * blockIdx.x + threadIdx.x;
    const int iyo = blockDim.y * blockIdx.y + threadIdx.y;
    const int izo = blockDim.z * blockIdx.z + threadIdx.z;

    if (check_dimensions(dims, ixo, iyo, izo)) {
        auto coord   = make_vector_td<D>(ixo, iyo, izo);
        auto coord2  = coord;
        int elements = prod(dims);
        int steps    = max(ceil(sigma_spatial * 4), 1.0f);

        int idx = co_to_idx<D>(coord, dims);
        vector_td<float, D> res(0);
        float image_value = image[idx];
        auto vec          = load_vector_from_SOA<D>(&vfield[idx], elements);

        float norm_factor = 0;
        if (ixo == 0 && iyo == 0) printf("Steps: %d\n",steps);
        for (int i = -steps; i < steps; i++) {
            coord2[N] = coord[N] + i;

            int idx2  = co_to_idx<D>((coord2 + dims) % dims, dims);
            auto vec2 = load_vector_from_SOA<D>(&vfield[idx2], elements);

            float image_diff = image_value - image[idx2];
            float vdiff      = norm_squared(vec - vec2);
            float weight
                = expf(- float(i * i) / (2 * sigma_spatial * sigma_spatial)
                       - image_diff * image_diff / (2 * sigma_int * sigma_int) - vdiff / (2 * sigma_diff * sigma_diff));
            norm_factor += weight;

            res += weight * vec2;
        }

        // atomicAdd(&out[co_to_idx<3>(coord,dims)],res/norm);

        out[idx] = res[0] / norm_factor;
        if (D > 1) {
            out[idx + elements] = res[1] / norm_factor;
            if (D > 2) {
                out[idx + 2 * elements] = res[2] / norm_factor;
            }
        }
    }
}

void Gadgetron::bilateral_vfield(
    cuNDArray<float>& vfield1, cuNDArray<float>& image, floatd3 sigma_spatial, float sigma_int, float sigma_diff) {
    cudaExtent extent;
    extent.width  = vfield1.get_size(0);
    extent.height = vfield1.get_size(1);
    extent.depth  = vfield1.get_size(2);

    dim3 threads(8, 8, 8);
    dim3 grid((extent.width + threads.x - 1) / threads.x, (extent.height + threads.y - 1) / threads.y,
        (extent.depth + threads.z - 1) / threads.z);
    auto vfield_copy = vfield1;

    vector_td<int, 3> image_dims(image.get_size(0), image.get_size(1), image.get_size(2));

    bilateral_kernel1D<3, 0><<<grid, threads>>>(
        vfield1.data(), vfield_copy.data(), image.data(), image_dims, sigma_spatial[0], sigma_int, sigma_diff);
    vfield_copy = vfield1;
    bilateral_kernel1D<3, 1><<<grid, threads>>>(
        vfield1.data(), vfield_copy.data(), image.data(), image_dims, sigma_spatial[1], sigma_int, sigma_diff);

    if (vfield1.dimensions()[2] > 1) {
        vfield_copy = vfield1;
        bilateral_kernel1D<3, 2><<<grid, threads>>>(
            vfield1.data(), vfield_copy.data(), image.data(), image_dims, sigma_spatial[2], sigma_int, sigma_diff);
    }
}

void Gadgetron::bilateral_vfield(
    cuNDArray<float>& vfield1, cuNDArray<float>& image, floatd2 sigma_spatial, float sigma_int, float sigma_diff) {
    cudaExtent extent;
    extent.width  = vfield1.get_size(0);
    extent.height = vfield1.get_size(1);

    dim3 threads(16, 16);
    dim3 grid((extent.width + threads.x - 1) / threads.x, (extent.height + threads.y - 1) / threads.y);
    auto vfield_copy = vfield1;

    vector_td<int, 2> image_dims(image.get_size(0), image.get_size(1));

    bilateral_kernel1D<2, 0><<<grid, threads>>>(
        vfield1.data(), vfield_copy.data(), image.data(), image_dims, sigma_spatial[0], sigma_int, sigma_diff);
    vfield_copy = vfield1;
    bilateral_kernel1D<2, 1><<<grid, threads>>>(
        vfield1.data(), vfield_copy.data(), image.data(), image_dims, sigma_spatial[1], sigma_int, sigma_diff);
}

template class cuDemonsSolver<float, 3>;
template class cuDemonsSolver<float, 2>;
