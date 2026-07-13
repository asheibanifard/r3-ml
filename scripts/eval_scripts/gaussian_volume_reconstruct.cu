
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#define CUDA_CHECK(x) do { cudaError_t e=(x); if(e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); } while(0)

#pragma pack(push,1)
struct Header { uint32_t magic, version; uint64_t count; };
#pragma pack(pop)

struct GIn {
    float mx,my,mz;
    float sx,sy,sz;
    float qw,qx,qy,qz;
    float intensity;
};

struct GDev {
    float mx,my,mz;
    float q00,q01,q02,q11,q12,q22;
    float intensity;
    float rx,ry,rz;
};

struct Grid {
    int nx,ny,nz;
    float minx,miny,minz,maxx,maxy,maxz;
    float dx,dy,dz,cutoff;
};

static constexpr uint32_t MAGIC=0x47534D50u;
static constexpr uint32_t VERSION=1u;

static std::vector<GIn> read_file(const std::string& p){
    std::ifstream f(p,std::ios::binary);
    if(!f) throw std::runtime_error("cannot open input");
    Header h{};
    f.read((char*)&h,sizeof(h));
    if(!f || h.magic!=MAGIC || h.version!=VERSION) throw std::runtime_error("bad gaussian binary");
    std::vector<GIn> g((size_t)h.count);
    f.read((char*)g.data(),(std::streamsize)(g.size()*sizeof(GIn)));
    if(!f) throw std::runtime_error("truncated gaussian binary");
    return g;
}

static void normq(float& w,float& x,float& y,float& z){
    float n2=w*w+x*x+y*y+z*z;
    if(!(n2>0.f)){w=1.f;x=y=z=0.f;return;}
    float inv=1.f/std::sqrt(n2);
    w*=inv;x*=inv;y*=inv;z*=inv;
}

static std::vector<GDev> prepare(const std::vector<GIn>& in,float cutoff){
    std::vector<GDev> out;
    out.reserve(in.size());
    float k=std::sqrt(cutoff);
    for(const auto& a:in){
        if(!std::isfinite(a.intensity) || a.intensity==0.f) continue;
        float sx=std::max(std::fabs(a.sx),1e-8f);
        float sy=std::max(std::fabs(a.sy),1e-8f);
        float sz=std::max(std::fabs(a.sz),1e-8f);
        float w=a.qw,x=a.qx,y=a.qy,z=a.qz;
        normq(w,x,y,z);
        float r00=1-2*(y*y+z*z), r01=2*(x*y-z*w), r02=2*(x*z+y*w);
        float r10=2*(x*y+z*w), r11=1-2*(x*x+z*z), r12=2*(y*z-x*w);
        float r20=2*(x*z-y*w), r21=2*(y*z+x*w), r22=1-2*(x*x+y*y);
        float ix=1/(sx*sx), iy=1/(sy*sy), iz=1/(sz*sz);
        GDev g{};
        g.mx=a.mx; g.my=a.my; g.mz=a.mz;
        g.q00=r00*r00*ix+r01*r01*iy+r02*r02*iz;
        g.q01=r00*r10*ix+r01*r11*iy+r02*r12*iz;
        g.q02=r00*r20*ix+r01*r21*iy+r02*r22*iz;
        g.q11=r10*r10*ix+r11*r11*iy+r12*r12*iz;
        g.q12=r10*r20*ix+r11*r21*iy+r12*r22*iz;
        g.q22=r20*r20*ix+r21*r21*iy+r22*r22*iz;
        g.intensity=a.intensity;
        g.rx=k*std::sqrt(r00*r00*sx*sx+r01*r01*sy*sy+r02*r02*sz*sz);
        g.ry=k*std::sqrt(r10*r10*sx*sx+r11*r11*sy*sy+r12*r12*sz*sz);
        g.rz=k*std::sqrt(r20*r20*sx*sx+r21*r21*sy*sy+r22*r22*sz*sz);
        out.push_back(g);
    }
    return out;
}

__global__ void reconstruct(const GDev* gs,size_t n,Grid grid,float* vol){
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=n) return;
    GDev g=gs[i];
    int x0=max(0,(int)floorf((g.mx-g.rx-grid.minx)/grid.dx));
    int x1=min(grid.nx-1,(int)ceilf((g.mx+g.rx-grid.minx)/grid.dx));
    int y0=max(0,(int)floorf((g.my-g.ry-grid.miny)/grid.dy));
    int y1=min(grid.ny-1,(int)ceilf((g.my+g.ry-grid.miny)/grid.dy));
    int z0=max(0,(int)floorf((g.mz-g.rz-grid.minz)/grid.dz));
    int z1=min(grid.nz-1,(int)ceilf((g.mz+g.rz-grid.minz)/grid.dz));
    for(int z=z0;z<=z1;++z){
        float pz=grid.minz+(z+0.5f)*grid.dz, dz=pz-g.mz;
        for(int y=y0;y<=y1;++y){
            float py=grid.miny+(y+0.5f)*grid.dy, dy=py-g.my;
            for(int x=x0;x<=x1;++x){
                float px=grid.minx+(x+0.5f)*grid.dx, dx=px-g.mx;
                float m=g.q00*dx*dx+2*g.q01*dx*dy+2*g.q02*dx*dz+g.q11*dy*dy+2*g.q12*dy*dz+g.q22*dz*dz;
                if(m<=grid.cutoff){
                    size_t idx=((size_t)z*grid.ny+y)*grid.nx+x;
                    atomicAdd(&vol[idx],g.intensity*__expf(-0.5f*m));
                }
            }
        }
    }
}

int main(int argc,char** argv){
    try{
        if(argc!=13){
            std::cerr<<"usage: "<<argv[0]<<" input.bin output.raw nx ny nz minx miny minz maxx maxy maxz cutoff\n";
            return 1;
        }
        std::string inpath=argv[1], outpath=argv[2];
        Grid grid{};
        grid.nx=std::stoi(argv[3]); grid.ny=std::stoi(argv[4]); grid.nz=std::stoi(argv[5]);
        grid.minx=std::stof(argv[6]); grid.miny=std::stof(argv[7]); grid.minz=std::stof(argv[8]);
        grid.maxx=std::stof(argv[9]); grid.maxy=std::stof(argv[10]); grid.maxz=std::stof(argv[11]);
        grid.cutoff=std::stof(argv[12]);
        if(grid.nx<=0||grid.ny<=0||grid.nz<=0) throw std::runtime_error("invalid dimensions");
        if(!(grid.maxx>grid.minx&&grid.maxy>grid.miny&&grid.maxz>grid.minz)) throw std::runtime_error("invalid bounds");
        grid.dx=(grid.maxx-grid.minx)/grid.nx;
        grid.dy=(grid.maxy-grid.miny)/grid.ny;
        grid.dz=(grid.maxz-grid.minz)/grid.nz;

        auto raw=read_file(inpath);
        auto gs=prepare(raw,grid.cutoff);
        size_t vox=(size_t)grid.nx*grid.ny*grid.nz;
        size_t vb=vox*sizeof(float), gb=gs.size()*sizeof(GDev);

        GDev* dgs=nullptr; float* dvol=nullptr;
        CUDA_CHECK(cudaMalloc(&dgs,gb));
        CUDA_CHECK(cudaMalloc(&dvol,vb));
        CUDA_CHECK(cudaMemset(dvol,0,vb));
        CUDA_CHECK(cudaMemcpy(dgs,gs.data(),gb,cudaMemcpyHostToDevice));

        cudaEvent_t a,b; CUDA_CHECK(cudaEventCreate(&a)); CUDA_CHECK(cudaEventCreate(&b));
        int threads=128, blocks=(int)((gs.size()+threads-1)/threads);
        CUDA_CHECK(cudaEventRecord(a));
        reconstruct<<<blocks,threads>>>(dgs,gs.size(),grid,dvol);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(b));
        CUDA_CHECK(cudaEventSynchronize(b));
        float ms=0; CUDA_CHECK(cudaEventElapsedTime(&ms,a,b));

        std::vector<float> vol(vox);
        CUDA_CHECK(cudaMemcpy(vol.data(),dvol,vb,cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(dgs)); CUDA_CHECK(cudaFree(dvol));
        CUDA_CHECK(cudaEventDestroy(a)); CUDA_CHECK(cudaEventDestroy(b));

        std::ofstream out(outpath,std::ios::binary);
        out.write((char*)vol.data(),(std::streamsize)vb);
        if(!out) throw std::runtime_error("failed writing output");

        auto mm=std::minmax_element(vol.begin(),vol.end());
        std::ofstream meta(outpath+".json");
        meta<<"{\n"
            <<"  \"dtype\": \"float32\",\n"
            <<"  \"axis_order\": \"zyx\",\n"
            <<"  \"shape\": ["<<grid.nz<<", "<<grid.ny<<", "<<grid.nx<<"],\n"
            <<"  \"bounds_min\": ["<<grid.minx<<", "<<grid.miny<<", "<<grid.minz<<"],\n"
            <<"  \"bounds_max\": ["<<grid.maxx<<", "<<grid.maxy<<", "<<grid.maxz<<"],\n"
            <<"  \"voxel_size\": ["<<grid.dx<<", "<<grid.dy<<", "<<grid.dz<<"],\n"
            <<"  \"cutoff\": "<<grid.cutoff<<",\n"
            <<"  \"gaussian_count\": "<<gs.size()<<",\n"
            <<"  \"value_min\": "<<*mm.first<<",\n"
            <<"  \"value_max\": "<<*mm.second<<"\n"
            <<"}\n";

        std::cout<<"Input Gaussians: "<<raw.size()<<"\n";
        std::cout<<"Active Gaussians: "<<gs.size()<<"\n";
        std::cout<<"Volume shape (Z Y X): "<<grid.nz<<" "<<grid.ny<<" "<<grid.nx<<"\n";
        std::cout<<"CUDA reconstruction time: "<<ms<<" ms\n";
        std::cout<<"Output range: ["<<*mm.first<<", "<<*mm.second<<"]\n";
        std::cout<<"Saved: "<<outpath<<"\n";
        std::cout<<"Metadata: "<<outpath<<".json\n";
        return 0;
    }catch(const std::exception& e){
        std::cerr<<"Error: "<<e.what()<<"\n";
        return 1;
    }
}