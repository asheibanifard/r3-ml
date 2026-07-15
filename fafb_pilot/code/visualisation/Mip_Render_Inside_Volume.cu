
// gaussian_mip_inside_camera.cu
//
// Perspective MIP renderer with the camera fixed INSIDE the volume block.
//
// The camera position remains fixed, normally at the block centre. Yaw, pitch,
// and roll rotate the camera around its own position. Each pixel ray starts at
// the camera and travels outward until it exits the axis-aligned block.
//
// Rendering equation:
//
//   M(pixel) = max_t sum_i intensity_i *
//              exp(-0.5 * (ray(t)-mu_i)^T Q_i (ray(t)-mu_i))
//
// Build:
//   nvcc -O3 -std=c++17 --use_fast_math -lineinfo \
//        -gencode arch=compute_89,code=sm_89 \
//        gaussian_mip_inside_camera.cu -o gaussian_mip_inside_camera
//
// Run:
//   ./gaussian_mip_inside_camera gaussians.bin output.pfm \
//       128 128 64 200 \
//       0 0 0 \
//       0 0 0 \
//       90 \
//       -1 -1 -1 1 1 1
//
// Arguments:
//   1  input Gaussian binary
//   2  output PFM
//   3  width
//   4  height
//   5  depth/ray samples
//   6  benchmark frames
//   7-9   camera position x y z
//   10-12 yaw pitch roll in degrees
//   13 vertical field of view in degrees
//   14-19 block min xyz and max xyz
//
// Coordinate conventions:
//   * Right-handed world.
//   * At yaw=pitch=roll=0, camera forward is +Z, right is +X, up is +Y.
//   * Positive yaw rotates toward +X.
//   * Positive pitch rotates toward +Y.
//   * Roll rotates the image plane around forward.
//   * Quaternion checkpoint order is w,x,y,z.
//   * The camera sees only its perspective frustum, not all 360 degrees.
//
// Binary format:
//   uint32 magic = 0x47534D50
//   uint32 version = 1
//   uint64 count
//   P records of 11 float32 values:
//       mean xyz, scale xyz, quaternion wxyz, intensity
//
// This version conservatively bins Gaussians into image tiles. If a Gaussian
// support sphere intersects the near plane, it is assigned to all tiles to
// avoid false-negative culling.

#include <cuda_runtime.h>
#include <cub/cub.cuh>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cfloat>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t e__ = (call);                                               \
        if (e__ != cudaSuccess) {                                               \
            std::fprintf(stderr, "CUDA error %s:%d: %s\n",                      \
                         __FILE__, __LINE__, cudaGetErrorString(e__));           \
            std::exit(EXIT_FAILURE);                                            \
        }                                                                       \
    } while (0)

constexpr uint32_t GAUSSIAN_FILE_MAGIC = 0x47534D50u;
constexpr uint32_t GAUSSIAN_FILE_VERSION = 1u;

constexpr int TILE_W = 16;
constexpr int TILE_H = 16;
constexpr int TILE_THREADS = TILE_W * TILE_H;
constexpr int GAUSSIAN_BATCH = 128;

constexpr float MAHAL_CUTOFF = 20.0f;
constexpr float EPS_SCALE = 1e-6f;
constexpr float CAMERA_NEAR = 1e-4f;

struct GaussianDisk {
    float mean[3];
    float scale[3];
    float quat[4];
    float intensity;
};

struct GaussianGPU {
    float3 mean;
    float q00, q01, q02, q11, q12, q22;
    float intensity;
    int2 tile_min;
    int2 tile_max;
    int visible;
};

struct Range {
    uint32_t begin;
    uint32_t end;
};

struct GaussianFileHeader {
    uint32_t magic;
    uint32_t version;
    uint64_t count;
};

struct CameraGPU {
    float3 position;
    float3 right;
    float3 up;
    float3 forward;
    float tan_half_fov_y;
    float aspect;
};

struct BoxGPU {
    float3 minimum;
    float3 maximum;
};

__host__ __device__ inline int div_up(int a, int b) {
    return (a + b - 1) / b;
}

__host__ __device__ inline float3 add3(float3 a, float3 b) {
    return make_float3(a.x+b.x, a.y+b.y, a.z+b.z);
}

__host__ __device__ inline float3 sub3(float3 a, float3 b) {
    return make_float3(a.x-b.x, a.y-b.y, a.z-b.z);
}

__host__ __device__ inline float3 mul3(float3 a, float s) {
    return make_float3(a.x*s, a.y*s, a.z*s);
}

__host__ __device__ inline float dot3(float3 a, float3 b) {
    return a.x*b.x + a.y*b.y + a.z*b.z;
}

__host__ __device__ inline float3 cross3(float3 a, float3 b) {
    return make_float3(
        a.y*b.z-a.z*b.y,
        a.z*b.x-a.x*b.z,
        a.x*b.y-a.y*b.x
    );
}

__host__ __device__ inline float3 normalize3(float3 a) {
    const float n2 = dot3(a,a);
    const float inv = rsqrtf(fmaxf(n2, 1e-20f));
    return mul3(a, inv);
}

__device__ inline void quaternion_to_rotation(
    float w, float x, float y, float z,
    float R[9])
{
    const float inv = rsqrtf(fmaxf(w*w+x*x+y*y+z*z, 1e-20f));
    w*=inv; x*=inv; y*=inv; z*=inv;

    R[0]=1-2*(y*y+z*z); R[1]=2*(x*y-z*w); R[2]=2*(x*z+y*w);
    R[3]=2*(x*y+z*w); R[4]=1-2*(x*x+z*z); R[5]=2*(y*z-x*w);
    R[6]=2*(x*z-y*w); R[7]=2*(y*z+x*w); R[8]=1-2*(x*x+y*y);
}

__device__ inline bool inverse_symmetric_3x3(
    float a00, float a01, float a02,
    float a11, float a12, float a22,
    float& q00, float& q01, float& q02,
    float& q11, float& q12, float& q22)
{
    const float c00 = a11*a22-a12*a12;
    const float c01 = a02*a12-a01*a22;
    const float c02 = a01*a12-a02*a11;
    const float c11 = a00*a22-a02*a02;
    const float c12 = a01*a02-a00*a12;
    const float c22 = a00*a11-a01*a01;
    const float det = a00*c00+a01*c01+a02*c02;

    if (!(det > 1e-20f) || !isfinite(det)) return false;

    const float inv = 1.0f/det;
    q00=c00*inv; q01=c01*inv; q02=c02*inv;
    q11=c11*inv; q12=c12*inv; q22=c22*inv;
    return true;
}

__device__ inline float largest_eigenvalue_upper_bound(
    float s00, float s01, float s02,
    float s11, float s12, float s22)
{
    // Gershgorin upper bound. Conservative and inexpensive.
    const float r0 = fabsf(s01)+fabsf(s02);
    const float r1 = fabsf(s01)+fabsf(s12);
    const float r2 = fabsf(s02)+fabsf(s12);
    return fmaxf(s00+r0, fmaxf(s11+r1, s22+r2));
}

__global__ void preprocess_kernel(
    const GaussianDisk* __restrict__ input,
    GaussianGPU* __restrict__ output,
    uint32_t* __restrict__ tile_counts,
    int P,
    int width,
    int height,
    int tiles_x,
    int tiles_y,
    CameraGPU camera,
    float mahal_cutoff)
{
    const int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i >= P) return;

    const GaussianDisk in = input[i];
    GaussianGPU out{};
    out.mean = make_float3(in.mean[0], in.mean[1], in.mean[2]);
    out.intensity = fmaxf(in.intensity, 0.0f);

    const float sx=fmaxf(fabsf(in.scale[0]),EPS_SCALE);
    const float sy=fmaxf(fabsf(in.scale[1]),EPS_SCALE);
    const float sz=fmaxf(fabsf(in.scale[2]),EPS_SCALE);

    float R[9];
    quaternion_to_rotation(
        in.quat[0],in.quat[1],in.quat[2],in.quat[3],R
    );

    const float vx=sx*sx, vy=sy*sy, vz=sz*sz;

    const float s00=R[0]*R[0]*vx+R[1]*R[1]*vy+R[2]*R[2]*vz;
    const float s01=R[0]*R[3]*vx+R[1]*R[4]*vy+R[2]*R[5]*vz;
    const float s02=R[0]*R[6]*vx+R[1]*R[7]*vy+R[2]*R[8]*vz;
    const float s11=R[3]*R[3]*vx+R[4]*R[4]*vy+R[5]*R[5]*vz;
    const float s12=R[3]*R[6]*vx+R[4]*R[7]*vy+R[5]*R[8]*vz;
    const float s22=R[6]*R[6]*vx+R[7]*R[7]*vy+R[8]*R[8]*vz;

    const bool ok = inverse_symmetric_3x3(
        s00,s01,s02,s11,s12,s22,
        out.q00,out.q01,out.q02,out.q11,out.q12,out.q22
    );

    if (!ok || out.intensity <= 0.0f || !isfinite(out.intensity)) {
        tile_counts[i]=0;
        output[i]=out;
        return;
    }

    const float3 relative = sub3(out.mean,camera.position);
    const float cx=dot3(relative,camera.right);
    const float cy=dot3(relative,camera.up);
    const float cz=dot3(relative,camera.forward);

    const float lambda_bound = fmaxf(
        largest_eigenvalue_upper_bound(s00,s01,s02,s11,s12,s22),
        0.0f
    );
    const float support_radius = sqrtf(mahal_cutoff*lambda_bound);

    int min_tx=0,max_tx=tiles_x,min_ty=0,max_ty=tiles_y;

    if (cz > support_radius + CAMERA_NEAR) {
        const float ndc_x = cx/(cz*camera.tan_half_fov_y*camera.aspect);
        const float ndc_y = cy/(cz*camera.tan_half_fov_y);

        // Conservative projected radius from support sphere.
        const float denom = fmaxf(cz-support_radius,CAMERA_NEAR);
        const float radius_ndc_y =
            support_radius/(denom*camera.tan_half_fov_y);
        const float radius_ndc_x = radius_ndc_y/camera.aspect;

        const float px=(ndc_x*0.5f+0.5f)*float(width);
        const float py=(0.5f-ndc_y*0.5f)*float(height);
        const float radius_px=fmaxf(
            radius_ndc_x*0.5f*float(width),
            radius_ndc_y*0.5f*float(height)
        )+1.0f;

        const int min_px_x=int(floorf(px-radius_px));
        const int max_px_x=int(ceilf(px+radius_px));
        const int min_px_y=int(floorf(py-radius_px));
        const int max_px_y=int(ceilf(py+radius_px));

        min_tx=max(0,min(min_px_x/TILE_W,tiles_x));
        max_tx=max(0,min(max_px_x/TILE_W+1,tiles_x));
        min_ty=max(0,min(min_px_y/TILE_H,tiles_y));
        max_ty=max(0,min(max_px_y/TILE_H+1,tiles_y));

        // Explicit frustum rejection.
        if (max_px_x < 0 || min_px_x >= width ||
            max_px_y < 0 || min_px_y >= height) {
            min_tx=max_tx=min_ty=max_ty=0;
        }
    } else if (cz + support_radius <= CAMERA_NEAR) {
        // Entire support is behind the camera.
        min_tx=max_tx=min_ty=max_ty=0;
    }
    // Otherwise support intersects the near plane: assign all tiles.

    const int count_x=max_tx-min_tx;
    const int count_y=max_ty-min_ty;
    const uint32_t count=(count_x>0 && count_y>0)
        ? uint32_t(count_x*count_y) : 0u;

    out.tile_min=make_int2(min_tx,min_ty);
    out.tile_max=make_int2(max_tx,max_ty);
    out.visible=(count>0u);

    tile_counts[i]=count;
    output[i]=out;
}

__global__ void duplicate_kernel(
    const GaussianGPU* __restrict__ gaussians,
    const uint32_t* __restrict__ offsets,
    uint32_t* __restrict__ keys,
    uint32_t* __restrict__ values,
    int P,
    int tiles_x)
{
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if (i>=P) return;

    const GaussianGPU g=gaussians[i];
    if (!g.visible) return;

    uint32_t write=(i==0)?0u:offsets[i-1];

    for (int ty=g.tile_min.y;ty<g.tile_max.y;++ty) {
        for (int tx=g.tile_min.x;tx<g.tile_max.x;++tx) {
            keys[write]=uint32_t(ty*tiles_x+tx);
            values[write]=uint32_t(i);
            ++write;
        }
    }
}

__global__ void identify_ranges_kernel(
    const uint32_t* __restrict__ keys,
    Range* __restrict__ ranges,
    uint32_t N)
{
    const uint32_t i=blockIdx.x*blockDim.x+threadIdx.x;
    if (i>=N) return;

    const uint32_t current=keys[i];

    if (i==0) ranges[current].begin=0;
    else {
        const uint32_t previous=keys[i-1];
        if (current!=previous) {
            ranges[previous].end=i;
            ranges[current].begin=i;
        }
    }
    if (i==N-1) ranges[current].end=N;
}

__device__ inline bool ray_box_exit(
    float3 origin,
    float3 direction,
    BoxGPU box,
    float& t_enter,
    float& t_exit)
{
    float near_t=-FLT_MAX;
    float far_t=FLT_MAX;

    const float o[3]={origin.x,origin.y,origin.z};
    const float d[3]={direction.x,direction.y,direction.z};
    const float lo[3]={box.minimum.x,box.minimum.y,box.minimum.z};
    const float hi[3]={box.maximum.x,box.maximum.y,box.maximum.z};

    #pragma unroll
    for (int axis=0;axis<3;++axis) {
        if (fabsf(d[axis])<1e-12f) {
            if (o[axis]<lo[axis] || o[axis]>hi[axis]) return false;
        } else {
            const float inv=1.0f/d[axis];
            float t0=(lo[axis]-o[axis])*inv;
            float t1=(hi[axis]-o[axis])*inv;
            if (t0>t1) { const float tmp=t0;t0=t1;t1=tmp; }
            near_t=fmaxf(near_t,t0);
            far_t=fminf(far_t,t1);
            if (far_t<near_t) return false;
        }
    }

    t_enter=fmaxf(near_t,0.0f);
    t_exit=far_t;
    return t_exit>t_enter;
}

__device__ inline float mahalanobis(
    const GaussianGPU& g,
    float3 p)
{
    const float dx=p.x-g.mean.x;
    const float dy=p.y-g.mean.y;
    const float dz=p.z-g.mean.z;

    return
        g.q00*dx*dx+
        2.0f*g.q01*dx*dy+
        2.0f*g.q02*dx*dz+
        g.q11*dy*dy+
        2.0f*g.q12*dy*dz+
        g.q22*dz*dz;
}

__global__ void render_kernel(
    const GaussianGPU* __restrict__ gaussians,
    const uint32_t* __restrict__ sorted_values,
    const Range* __restrict__ ranges,
    float* __restrict__ output,
    int width,
    int height,
    int tiles_x,
    int depth_samples,
    CameraGPU camera,
    BoxGPU box,
    float mahal_cutoff)
{
    const int lx=threadIdx.x;
    const int ly=threadIdx.y;
    const int linear_tid=ly*TILE_W+lx;

    const int px=blockIdx.x*TILE_W+lx;
    const int py=blockIdx.y*TILE_H+ly;
    const bool inside_image=(px<width && py<height);

    const int tile_id=blockIdx.y*tiles_x+blockIdx.x;
    const Range range=ranges[tile_id];

    float3 direction=make_float3(0,0,1);
    float t0=0.0f,t1=0.0f;
    bool valid_ray=false;

    if (inside_image) {
        const float ndc_x =
            (2.0f*(float(px)+0.5f)/float(width)-1.0f);
        const float ndc_y =
            (1.0f-2.0f*(float(py)+0.5f)/float(height));

        const float camera_x =
            ndc_x*camera.aspect*camera.tan_half_fov_y;
        const float camera_y =
            ndc_y*camera.tan_half_fov_y;

        direction=normalize3(add3(
            camera.forward,
            add3(mul3(camera.right,camera_x),
                 mul3(camera.up,camera_y))
        ));

        valid_ray=ray_box_exit(
            camera.position,direction,box,t0,t1
        );
        t0=fmaxf(t0,CAMERA_NEAR);
    }

    float best=0.0f;
    __shared__ GaussianGPU shared_g[GAUSSIAN_BATCH];

    for (int sample=0;sample<depth_samples;++sample) {
        float density=0.0f;
        float3 point=make_float3(0,0,0);

        if (inside_image && valid_ray) {
            const float u=(depth_samples>1)
                ? float(sample)/float(depth_samples-1)
                : 0.5f;
            const float t=t0+(t1-t0)*u;
            point=add3(camera.position,mul3(direction,t));
        }

        for (uint32_t begin=range.begin;
             begin<range.end;
             begin+=GAUSSIAN_BATCH)
        {
            const uint32_t count=min(
                uint32_t(GAUSSIAN_BATCH),
                range.end-begin
            );

            if (linear_tid<int(count)) {
                shared_g[linear_tid]=
                    gaussians[sorted_values[begin+linear_tid]];
            }
            __syncthreads();

            if (inside_image && valid_ray) {
                #pragma unroll 4
                for (uint32_t j=0;j<count;++j) {
                    const GaussianGPU g=shared_g[j];
                    const float m=mahalanobis(g,point);
                    if (m>=0.0f && m<=mahal_cutoff) {
                        density+=g.intensity*__expf(-0.5f*m);
                    }
                }
            }
            __syncthreads();
        }

        if (inside_image && valid_ray) best=fmaxf(best,density);
    }

    if (inside_image) output[py*width+px]=best;
}

static std::vector<GaussianDisk> read_gaussians(const std::string& path) {
    std::ifstream stream(path,std::ios::binary);
    if (!stream) throw std::runtime_error("Cannot open "+path);

    GaussianFileHeader header{};
    stream.read(reinterpret_cast<char*>(&header),sizeof(header));
    if (!stream || header.magic!=GAUSSIAN_FILE_MAGIC || header.version!=GAUSSIAN_FILE_VERSION)
        throw std::runtime_error("Invalid Gaussian binary header.");

    if (header.count==0 ||
        header.count>uint64_t(std::numeric_limits<int>::max()))
        throw std::runtime_error("Invalid Gaussian count.");

    std::vector<GaussianDisk> data(size_t(header.count));
    stream.read(
        reinterpret_cast<char*>(data.data()),
        std::streamsize(data.size()*sizeof(GaussianDisk))
    );
    if (!stream) throw std::runtime_error("Truncated Gaussian binary.");
    return data;
}

static void write_pfm(
    const std::string& path,
    const std::vector<float>& image,
    int width,
    int height)
{
    std::ofstream stream(path,std::ios::binary);
    if (!stream) throw std::runtime_error("Cannot create "+path);

    stream<<"Pf\n"<<width<<" "<<height<<"\n-1.0\n";
    for (int y=height-1;y>=0;--y) {
        stream.write(
            reinterpret_cast<const char*>(
                image.data()+size_t(y)*width
            ),
            std::streamsize(width*sizeof(float))
        );
    }
}

static float deg_to_rad(float degrees) {
    return degrees*3.14159265358979323846f/180.0f;
}

static CameraGPU make_camera(
    float3 position,
    float yaw_deg,
    float pitch_deg,
    float roll_deg,
    float fov_y_deg,
    int width,
    int height)
{
    const float yaw=deg_to_rad(yaw_deg);
    const float pitch=deg_to_rad(pitch_deg);
    const float roll=deg_to_rad(roll_deg);

    // Forward at zero rotation is +Z.
    float3 forward=make_float3(
        sinf(yaw)*cosf(pitch),
        sinf(pitch),
        cosf(yaw)*cosf(pitch)
    );
    forward=normalize3(forward);

    const float3 world_up=make_float3(0,1,0);
    float3 right=normalize3(cross3(world_up,forward));
    float3 up=normalize3(cross3(forward,right));

    // Roll around forward.
    const float cr=cosf(roll), sr=sinf(roll);
    const float3 rolled_right=add3(mul3(right,cr),mul3(up,sr));
    const float3 rolled_up=add3(mul3(up,cr),mul3(right,-sr));

    CameraGPU camera{};
    camera.position=position;
    camera.right=normalize3(rolled_right);
    camera.up=normalize3(rolled_up);
    camera.forward=forward;
    camera.tan_half_fov_y=tanf(0.5f*deg_to_rad(fov_y_deg));
    camera.aspect=float(width)/float(height);
    return camera;
}

class GaussianRenderer {
public:
    GaussianRenderer(
        const std::vector<GaussianDisk>& host,
        int width,
        int height,
        int depth_samples,
        CameraGPU camera,
        BoxGPU box)
        : P_(int(host.size())),
          width_(width),
          height_(height),
          depth_samples_(depth_samples),
          tiles_x_(div_up(width,TILE_W)),
          tiles_y_(div_up(height,TILE_H)),
          tile_count_(tiles_x_*tiles_y_),
          camera_(camera),
          box_(box)
    {
        CUDA_CHECK(cudaStreamCreateWithFlags(
            &stream_,cudaStreamNonBlocking
        ));

        CUDA_CHECK(cudaMalloc(&d_input_,size_t(P_)*sizeof(GaussianDisk)));
        CUDA_CHECK(cudaMalloc(&d_gaussians_,size_t(P_)*sizeof(GaussianGPU)));
        CUDA_CHECK(cudaMalloc(&d_counts_,size_t(P_)*sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&d_offsets_,size_t(P_)*sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&d_ranges_,size_t(tile_count_)*sizeof(Range)));
        CUDA_CHECK(cudaMalloc(
            &d_output_,size_t(width_)*height_*sizeof(float)
        ));

        CUDA_CHECK(cudaMemcpyAsync(
            d_input_,host.data(),size_t(P_)*sizeof(GaussianDisk),
            cudaMemcpyHostToDevice,stream_
        ));

        rebuild_bins();
    }

    ~GaussianRenderer() {
        cudaFree(d_input_); cudaFree(d_gaussians_);
        cudaFree(d_counts_); cudaFree(d_offsets_);
        cudaFree(d_keys_in_); cudaFree(d_keys_out_);
        cudaFree(d_values_in_); cudaFree(d_values_out_);
        cudaFree(d_ranges_); cudaFree(d_output_);
        cudaFree(d_scan_temp_); cudaFree(d_sort_temp_);
        cudaStreamDestroy(stream_);
    }

    void render() {
        dim3 block(TILE_W,TILE_H);
        dim3 grid(tiles_x_,tiles_y_);
        render_kernel<<<grid,block,0,stream_>>>(
            d_gaussians_,d_values_out_,d_ranges_,d_output_,
            width_,height_,tiles_x_,depth_samples_,
            camera_,box_,MAHAL_CUTOFF
        );
        CUDA_CHECK(cudaGetLastError());
    }

    void synchronize() {
        CUDA_CHECK(cudaStreamSynchronize(stream_));
    }

    cudaStream_t stream() const { return stream_; }
    uint32_t pair_count() const { return pair_count_; }

    std::vector<float> download() {
        std::vector<float> out(size_t(width_)*height_);
        CUDA_CHECK(cudaMemcpyAsync(
            out.data(),d_output_,out.size()*sizeof(float),
            cudaMemcpyDeviceToHost,stream_
        ));
        synchronize();
        return out;
    }

private:
    void rebuild_bins() {
        const int threads=256;
        const int blocks=div_up(P_,threads);

        preprocess_kernel<<<blocks,threads,0,stream_>>>(
            d_input_,d_gaussians_,d_counts_,
            P_,width_,height_,tiles_x_,tiles_y_,
            camera_,MAHAL_CUTOFF
        );
        CUDA_CHECK(cudaGetLastError());

        size_t scan_bytes=0;
        CUDA_CHECK(cub::DeviceScan::InclusiveSum(
            nullptr,scan_bytes,d_counts_,d_offsets_,P_,stream_
        ));
        CUDA_CHECK(cudaMalloc(&d_scan_temp_,scan_bytes));
        CUDA_CHECK(cub::DeviceScan::InclusiveSum(
            d_scan_temp_,scan_bytes,d_counts_,d_offsets_,P_,stream_
        ));

        CUDA_CHECK(cudaMemcpyAsync(
            &pair_count_,d_offsets_+(P_-1),sizeof(uint32_t),
            cudaMemcpyDeviceToHost,stream_
        ));
        synchronize();

        if (pair_count_==0)
            throw std::runtime_error("No Gaussian overlaps the camera frustum.");

        CUDA_CHECK(cudaMalloc(&d_keys_in_,size_t(pair_count_)*sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&d_keys_out_,size_t(pair_count_)*sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&d_values_in_,size_t(pair_count_)*sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&d_values_out_,size_t(pair_count_)*sizeof(uint32_t)));

        duplicate_kernel<<<blocks,threads,0,stream_>>>(
            d_gaussians_,d_offsets_,d_keys_in_,d_values_in_,
            P_,tiles_x_
        );
        CUDA_CHECK(cudaGetLastError());

        size_t sort_bytes=0;
        CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
            nullptr,sort_bytes,
            d_keys_in_,d_keys_out_,
            d_values_in_,d_values_out_,
            pair_count_,0,32,stream_
        ));
        CUDA_CHECK(cudaMalloc(&d_sort_temp_,sort_bytes));
        CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
            d_sort_temp_,sort_bytes,
            d_keys_in_,d_keys_out_,
            d_values_in_,d_values_out_,
            pair_count_,0,32,stream_
        ));

        CUDA_CHECK(cudaMemsetAsync(
            d_ranges_,0,size_t(tile_count_)*sizeof(Range),stream_
        ));

        identify_ranges_kernel<<<
            div_up(int(pair_count_),threads),threads,0,stream_
        >>>(d_keys_out_,d_ranges_,pair_count_);
        CUDA_CHECK(cudaGetLastError());
        synchronize();
    }

    int P_,width_,height_,depth_samples_;
    int tiles_x_,tiles_y_,tile_count_;
    uint32_t pair_count_=0;

    CameraGPU camera_;
    BoxGPU box_;
    cudaStream_t stream_{};

    GaussianDisk* d_input_=nullptr;
    GaussianGPU* d_gaussians_=nullptr;
    uint32_t* d_counts_=nullptr;
    uint32_t* d_offsets_=nullptr;
    uint32_t* d_keys_in_=nullptr;
    uint32_t* d_keys_out_=nullptr;
    uint32_t* d_values_in_=nullptr;
    uint32_t* d_values_out_=nullptr;
    Range* d_ranges_=nullptr;
    float* d_output_=nullptr;
    void* d_scan_temp_=nullptr;
    void* d_sort_temp_=nullptr;
};


// -----------------------------------------------------------------------------
// Dense voxel input path
// -----------------------------------------------------------------------------

constexpr uint32_t DENSE_FILE_MAGIC = 0x564F584Cu; // 'VOXL'
constexpr uint32_t DENSE_FILE_VERSION = 1u;

struct DenseFileHeader {
    uint32_t magic;
    uint32_t version;
    uint32_t depth;
    uint32_t height;
    uint32_t width;
};

static_assert(
    sizeof(DenseFileHeader) == 20,
    "Unexpected DenseFileHeader size."
);

__device__ inline float3 dense_world_to_texture(
    float3 point,
    BoxGPU box)
{
    const float3 extent = sub3(box.maximum, box.minimum);

    return make_float3(
        (point.x - box.minimum.x) / extent.x,
        (point.y - box.minimum.y) / extent.y,
        (point.z - box.minimum.z) / extent.z
    );
}

__global__ void dense_mip_kernel(
    cudaTextureObject_t texture,
    float* __restrict__ output,
    int width,
    int height,
    int depth_samples,
    CameraGPU camera,
    BoxGPU box)
{
    const int px = blockIdx.x * blockDim.x + threadIdx.x;
    const int py = blockIdx.y * blockDim.y + threadIdx.y;

    if (px >= width || py >= height) {
        return;
    }

    const float ndc_x =
        2.0f * (float(px) + 0.5f) / float(width) - 1.0f;
    const float ndc_y =
        1.0f - 2.0f * (float(py) + 0.5f) / float(height);

    const float camera_x =
        ndc_x * camera.aspect * camera.tan_half_fov_y;
    const float camera_y =
        ndc_y * camera.tan_half_fov_y;

    const float3 direction = normalize3(
        add3(
            camera.forward,
            add3(
                mul3(camera.right, camera_x),
                mul3(camera.up, camera_y)
            )
        )
    );

    float t_enter = 0.0f;
    float t_exit = 0.0f;

    if (!ray_box_exit(
            camera.position,
            direction,
            box,
            t_enter,
            t_exit)) {
        output[py * width + px] = 0.0f;
        return;
    }

    t_enter = fmaxf(t_enter, CAMERA_NEAR);

    float maximum_value = -FLT_MAX;

    for (int sample = 0; sample < depth_samples; ++sample) {
        const float u =
            depth_samples > 1
                ? float(sample) / float(depth_samples - 1)
                : 0.5f;

        const float t =
            t_enter + (t_exit - t_enter) * u;

        const float3 point = add3(
            camera.position,
            mul3(direction, t)
        );

        const float3 texture_position =
            dense_world_to_texture(point, box);

        const float value = tex3D<float>(
            texture,
            texture_position.x,
            texture_position.y,
            texture_position.z
        );

        maximum_value = fmaxf(maximum_value, value);
    }

    output[py * width + px] =
        isfinite(maximum_value) ? maximum_value : 0.0f;
}

static std::vector<float> read_dense_volume(
    const std::string& path,
    uint32_t& depth,
    uint32_t& height,
    uint32_t& width)
{
    std::ifstream stream(path, std::ios::binary);

    if (!stream) {
        throw std::runtime_error(
            "Cannot open dense-volume binary: " + path
        );
    }

    DenseFileHeader header{};

    stream.read(
        reinterpret_cast<char*>(&header),
        sizeof(header)
    );

    if (!stream ||
        header.magic != DENSE_FILE_MAGIC ||
        header.version != DENSE_FILE_VERSION) {
        throw std::runtime_error(
            "Invalid dense-volume binary header."
        );
    }

    if (header.depth == 0 ||
        header.height == 0 ||
        header.width == 0) {
        throw std::runtime_error(
            "Dense-volume dimensions must be positive."
        );
    }

    const uint64_t voxel_count =
        uint64_t(header.depth) *
        uint64_t(header.height) *
        uint64_t(header.width);

    if (voxel_count >
        uint64_t(
            std::numeric_limits<size_t>::max() /
            sizeof(float)
        )) {
        throw std::runtime_error(
            "Dense volume is too large."
        );
    }

    std::vector<float> volume(
        static_cast<size_t>(voxel_count)
    );

    stream.read(
        reinterpret_cast<char*>(volume.data()),
        static_cast<std::streamsize>(
            volume.size() * sizeof(float)
        )
    );

    if (!stream) {
        throw std::runtime_error(
            "Truncated dense-volume binary."
        );
    }

    depth = header.depth;
    height = header.height;
    width = header.width;

    return volume;
}

class DenseVoxelRenderer {
public:
    DenseVoxelRenderer(
        const std::vector<float>& host_volume,
        uint32_t volume_depth,
        uint32_t volume_height,
        uint32_t volume_width,
        int output_width,
        int output_height,
        int depth_samples,
        CameraGPU camera,
        BoxGPU box)
        : output_width_(output_width),
          output_height_(output_height),
          depth_samples_(depth_samples),
          camera_(camera),
          box_(box)
    {
        CUDA_CHECK(cudaStreamCreate(&stream_));

        const cudaChannelFormatDesc channel =
            cudaCreateChannelDesc<float>();

        const cudaExtent extent = make_cudaExtent(
            volume_width,
            volume_height,
            volume_depth
        );

        CUDA_CHECK(cudaMalloc3DArray(
            &volume_array_,
            &channel,
            extent
        ));

        cudaMemcpy3DParms copy{};
        copy.srcPtr = make_cudaPitchedPtr(
            const_cast<float*>(host_volume.data()),
            size_t(volume_width) * sizeof(float),
            volume_width,
            volume_height
        );
        copy.dstArray = volume_array_;
        copy.extent = extent;
        copy.kind = cudaMemcpyHostToDevice;

        CUDA_CHECK(cudaMemcpy3DAsync(
            &copy,
            stream_
        ));

        cudaResourceDesc resource{};
        resource.resType = cudaResourceTypeArray;
        resource.res.array.array = volume_array_;

        cudaTextureDesc texture{};
        texture.addressMode[0] = cudaAddressModeClamp;
        texture.addressMode[1] = cudaAddressModeClamp;
        texture.addressMode[2] = cudaAddressModeClamp;
        texture.filterMode = cudaFilterModeLinear;
        texture.readMode = cudaReadModeElementType;
        texture.normalizedCoords = 1;

        CUDA_CHECK(cudaCreateTextureObject(
            &texture_,
            &resource,
            &texture,
            nullptr
        ));

        CUDA_CHECK(cudaMalloc(
            &d_output_,
            size_t(output_width_) *
            size_t(output_height_) *
            sizeof(float)
        ));

        synchronize();
    }

    ~DenseVoxelRenderer() {
        if (d_output_) {
            cudaFree(d_output_);
        }
        if (texture_) {
            cudaDestroyTextureObject(texture_);
        }
        if (volume_array_) {
            cudaFreeArray(volume_array_);
        }
        if (stream_) {
            cudaStreamDestroy(stream_);
        }
    }

    DenseVoxelRenderer(const DenseVoxelRenderer&) = delete;
    DenseVoxelRenderer& operator=(
        const DenseVoxelRenderer&) = delete;

    void render() {
        const dim3 block(16, 16);
        const dim3 grid(
            div_up(output_width_, int(block.x)),
            div_up(output_height_, int(block.y))
        );

        dense_mip_kernel<<<grid, block, 0, stream_>>>(
            texture_,
            d_output_,
            output_width_,
            output_height_,
            depth_samples_,
            camera_,
            box_
        );

        CUDA_CHECK(cudaGetLastError());
    }

    void synchronize() {
        CUDA_CHECK(cudaStreamSynchronize(stream_));
    }

    cudaStream_t stream() const {
        return stream_;
    }

    std::vector<float> download() {
        std::vector<float> output(
            size_t(output_width_) *
            size_t(output_height_)
        );

        CUDA_CHECK(cudaMemcpyAsync(
            output.data(),
            d_output_,
            output.size() * sizeof(float),
            cudaMemcpyDeviceToHost,
            stream_
        ));

        synchronize();

        return output;
    }

private:
    int output_width_{};
    int output_height_{};
    int depth_samples_{};

    CameraGPU camera_{};
    BoxGPU box_{};

    cudaStream_t stream_{};
    cudaArray_t volume_array_{};
    cudaTextureObject_t texture_{};
    float* d_output_{};
};

// -----------------------------------------------------------------------------
// Shared command-line and benchmark logic
// -----------------------------------------------------------------------------

enum class RepresentationType {
    DenseVoxel,
    PretrainedGaussian
};

static RepresentationType parse_representation_type(
    const std::string& value)
{
    if (value == "dense_voxel") {
        return RepresentationType::DenseVoxel;
    }

    if (value == "pretrained_gaussian") {
        return RepresentationType::PretrainedGaussian;
    }

    throw std::runtime_error(
        "Invalid representation type '" + value +
        "'. Expected dense_voxel or pretrained_gaussian."
    );
}

static float3 box_center(const BoxGPU& box) {
    return mul3(
        add3(box.minimum, box.maximum),
        0.5f
    );
}

template <typename RendererType>
static std::vector<float> benchmark_and_download(
    RendererType& renderer,
    int warmup_frames,
    int measured_frames,
    float& mean_render_ms,
    float& frames_per_second)
{
    for (int frame = 0;
         frame < warmup_frames;
         ++frame) {
        renderer.render();
    }

    renderer.synchronize();

    cudaEvent_t start{};
    cudaEvent_t stop{};

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(
        start,
        renderer.stream()
    ));

    for (int frame = 0;
         frame < measured_frames;
         ++frame) {
        renderer.render();
    }

    CUDA_CHECK(cudaEventRecord(
        stop,
        renderer.stream()
    ));

    CUDA_CHECK(cudaEventSynchronize(stop));

    float total_ms = 0.0f;

    CUDA_CHECK(cudaEventElapsedTime(
        &total_ms,
        start,
        stop
    ));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    mean_render_ms =
        total_ms / float(measured_frames);

    frames_per_second =
        1000.0f / mean_render_ms;

    renderer.render();

    return renderer.download();
}

int main(int argc, char** argv) {
    try {
        if (argc != 18) {
            std::cerr
                << "Usage:\n  "
                << argv[0]
                << " <dense_voxel|pretrained_gaussian>"
                << " input.bin output.pfm"
                << " width height depth_samples frames"
                << " yaw pitch roll fov_y"
                << " min_x min_y min_z"
                << " max_x max_y max_z\n\n"
                << "Examples:\n"
                << "  " << argv[0]
                << " dense_voxel volume.bin voxel.pfm"
                << " 128 128 256 200"
                << " 0 0 0 90"
                << " -1 -1 -1 1 1 1\n\n"
                << "  " << argv[0]
                << " pretrained_gaussian gaussians.bin gaussian.pfm"
                << " 128 128 64 200"
                << " 0 0 0 90"
                << " -1 -1 -1 1 1 1\n";

            return EXIT_FAILURE;
        }

        const RepresentationType representation =
            parse_representation_type(argv[1]);

        const std::string input_path = argv[2];
        const std::string output_path = argv[3];

        const int width = std::stoi(argv[4]);
        const int height = std::stoi(argv[5]);
        const int depth_samples = std::stoi(argv[6]);
        const int measured_frames = std::stoi(argv[7]);

        const float yaw = std::stof(argv[8]);
        const float pitch = std::stof(argv[9]);
        const float roll = std::stof(argv[10]);
        const float fov_y = std::stof(argv[11]);

        BoxGPU box{};
        box.minimum = make_float3(
            std::stof(argv[12]),
            std::stof(argv[13]),
            std::stof(argv[14])
        );
        box.maximum = make_float3(
            std::stof(argv[15]),
            std::stof(argv[16]),
            std::stof(argv[17])
        );

        if (width <= 0 ||
            height <= 0 ||
            depth_samples <= 0 ||
            measured_frames <= 0) {
            throw std::runtime_error(
                "Dimensions, samples, and frames must be positive."
            );
        }

        if (!(fov_y > 1.0f && fov_y < 179.0f)) {
            throw std::runtime_error(
                "FOV must be between 1 and 179 degrees."
            );
        }

        if (!(box.maximum.x > box.minimum.x &&
              box.maximum.y > box.minimum.y &&
              box.maximum.z > box.minimum.z)) {
            throw std::runtime_error(
                "Invalid block bounds."
            );
        }

        const float3 camera_position =
            box_center(box);

        const CameraGPU camera = make_camera(
            camera_position,
            yaw,
            pitch,
            roll,
            fov_y,
            width,
            height
        );

        constexpr int warmup_frames = 20;

        std::cout
            << "Representation: "
            << (
                representation ==
                RepresentationType::DenseVoxel
                    ? "dense_voxel"
                    : "pretrained_gaussian"
            )
            << "\n"
            << "Camera centre: "
            << camera_position.x << " "
            << camera_position.y << " "
            << camera_position.z << "\n"
            << "Yaw/pitch/roll: "
            << yaw << " "
            << pitch << " "
            << roll << "\n"
            << "Vertical FOV: "
            << fov_y << "\n"
            << "Output resolution: "
            << width << " x " << height << "\n"
            << "Ray samples: "
            << depth_samples << "\n";

        float mean_render_ms = 0.0f;
        float frames_per_second = 0.0f;
        std::vector<float> output;

        if (representation ==
            RepresentationType::DenseVoxel) {
            uint32_t volume_depth = 0;
            uint32_t volume_height = 0;
            uint32_t volume_width = 0;

            const std::vector<float> volume =
                read_dense_volume(
                    input_path,
                    volume_depth,
                    volume_height,
                    volume_width
                );

            std::cout
                << "Dense volume: "
                << volume_width << " x "
                << volume_height << " x "
                << volume_depth << "\n";

            DenseVoxelRenderer renderer(
                volume,
                volume_depth,
                volume_height,
                volume_width,
                width,
                height,
                depth_samples,
                camera,
                box
            );

            output = benchmark_and_download(
                renderer,
                warmup_frames,
                measured_frames,
                mean_render_ms,
                frames_per_second
            );
        } else {
            const std::vector<GaussianDisk> gaussians =
                read_gaussians(input_path);

            std::cout
                << "Gaussians: "
                << gaussians.size() << "\n";

            GaussianRenderer renderer(
                gaussians,
                width,
                height,
                depth_samples,
                camera,
                box
            );

            std::cout
                << "Gaussian-tile pairs: "
                << renderer.pair_count() << "\n";

            output = benchmark_and_download(
                renderer,
                warmup_frames,
                measured_frames,
                mean_render_ms,
                frames_per_second
            );
        }

        write_pfm(
            output_path,
            output,
            width,
            height
        );

        const auto minmax = std::minmax_element(
            output.begin(),
            output.end()
        );

        std::cout
            << "Mean render time: "
            << mean_render_ms << " ms\n"
            << "FPS: "
            << frames_per_second << "\n"
            << "Output range: ["
            << *minmax.first << ", "
            << *minmax.second << "]\n"
            << "Saved: "
            << output_path << "\n";

        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr
            << "Error: "
            << error.what()
            << "\n";

        return EXIT_FAILURE;
    }
}