// day05: FP32 sum, reduce0 -> reduce7, plus an atomic comparison.
// Each first-stage block writes its own partial. A common second kernel
// combines partials so that changing the first-stage algorithm is easy to study.
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#define CUDA(call) do { cudaError_t e=(call); if(e!=cudaSuccess) \
 throw std::runtime_error(std::string(#call)+": "+cudaGetErrorString(e)); } while(0)

// Only warp 0 calls this helper, with all 32 lanes participating.
// At each step writers and readers address disjoint halves. __syncwarp
// orders shared-memory accesses before the next step. No implicit lockstep.
__device__ __forceinline__ void shared_warp_tail(float* s, unsigned t) {
    s[t] += s[t+32]; __syncwarp();
    if(t<16) s[t] += s[t+16]; __syncwarp();
    if(t< 8) s[t] += s[t+ 8]; __syncwarp();
    if(t< 4) s[t] += s[t+ 4]; __syncwarp();
    if(t< 2) s[t] += s[t+ 2]; __syncwarp();
    if(t< 1) s[t] += s[t+ 1]; __syncwarp();
}

// reduce0: interleaved addressing, modulo test; one input per thread.
__global__ void reduce0(const float* x,float* p,size_t n) {
    extern __shared__ float s[];
    unsigned t=threadIdx.x; size_t i=size_t(blockIdx.x)*blockDim.x+t;
    s[t]=i<n?x[i]:0.0f; __syncthreads();
    #pragma unroll 1
    for(unsigned step=1;step<blockDim.x;step*=2) {
        if(t%(2*step)==0) s[t]+=s[t+step];
        __syncthreads();
    }
    if(t==0) p[blockIdx.x]=s[0];
}
// reduce1: contiguous active thread IDs, strided shared-memory addresses.
__global__ void reduce1(const float* x,float* p,size_t n) {
    extern __shared__ float s[];
    unsigned t=threadIdx.x; size_t i=size_t(blockIdx.x)*blockDim.x+t;
    s[t]=i<n?x[i]:0.0f; __syncthreads();
    #pragma unroll 1
    for(unsigned step=1;step<blockDim.x;step*=2) {
        unsigned index=2*step*t;
        if(index<blockDim.x) s[index]+=s[index+step];
        __syncthreads();
    }
    if(t==0) p[blockIdx.x]=s[0];
}
// reduce2: descending stride; consecutive shared-memory accesses per warp.
__global__ void reduce2(const float* x,float* p,size_t n) {
    extern __shared__ float s[];
    unsigned t=threadIdx.x; size_t i=size_t(blockIdx.x)*blockDim.x+t;
    s[t]=i<n?x[i]:0.0f; __syncthreads();
    #pragma unroll 1
    for(unsigned step=blockDim.x/2;step>0;step/=2) {
        if(t<step) s[t]+=s[t+step];
        __syncthreads();
    }
    if(t==0) p[blockIdx.x]=s[0];
}
// reduce3: first add performed during loading; two inputs per thread.
__global__ void reduce3(const float* x,float* p,size_t n) {
    extern __shared__ float s[];
    unsigned t=threadIdx.x; size_t i=size_t(blockIdx.x)*(2*blockDim.x)+t;
    s[t]=(i<n?x[i]:0.0f)+(i+blockDim.x<n?x[i+blockDim.x]:0.0f);
    __syncthreads();
    #pragma unroll 1
    for(unsigned step=blockDim.x/2;step>0;step/=2) {
        if(t<step) s[t]+=s[t+step];
        __syncthreads();
    }
    if(t==0) p[blockIdx.x]=s[0];
}
// reduce4: finish within warp 0, replacing block barriers with warp barriers.
__global__ void reduce4(const float* x,float* p,size_t n) {
    extern __shared__ float s[];
    unsigned t=threadIdx.x; size_t i=size_t(blockIdx.x)*(2*blockDim.x)+t;
    s[t]=(i<n?x[i]:0.0f)+(i+blockDim.x<n?x[i+blockDim.x]:0.0f);
    __syncthreads();
    #pragma unroll 1
    for(unsigned step=blockDim.x/2;step>32;step/=2) {
        if(t<step) s[t]+=s[t+step];
        __syncthreads();
    }
    if(t<32) shared_warp_tail(s,t);
    if(t==0) p[blockIdx.x]=s[0];
}

// reduce5: compile-time block size, explicitly unrolled shared-memory tree.
// reduce6: same tree plus grid-stride register accumulation and capped grid.
template<unsigned B> __device__ __forceinline__ void unrolled_tree(float* s,unsigned t) {
    if constexpr(B>=1024) { if(t<512) s[t]+=s[t+512]; __syncthreads(); }
    if constexpr(B>= 512) { if(t<256) s[t]+=s[t+256]; __syncthreads(); }
    if constexpr(B>= 256) { if(t<128) s[t]+=s[t+128]; __syncthreads(); }
    if constexpr(B>= 128) { if(t< 64) s[t]+=s[t+ 64]; __syncthreads(); }
    if(t<32) shared_warp_tail(s,t);
}
template<unsigned B> __global__ void reduce5(const float* x,float* p,size_t n) {
    __shared__ float s[B]; unsigned t=threadIdx.x;
    size_t i=size_t(blockIdx.x)*(2*B)+t;
    s[t]=(i<n?x[i]:0.0f)+(i+B<n?x[i+B]:0.0f); __syncthreads();
    unrolled_tree<B>(s,t);
    if(t==0) p[blockIdx.x]=s[0];
}
template<unsigned B> __global__ void reduce6(const float* x,float* p,size_t n) {
    __shared__ float s[B]; unsigned t=threadIdx.x; float v=0;
    size_t stride=size_t(gridDim.x)*(2*B);
    for(size_t i=size_t(blockIdx.x)*(2*B)+t;i<n;i+=stride) {
        v+=x[i]; if(i+B<n) v+=x[i+B];
    }
    s[t]=v; __syncthreads(); unrolled_tree<B>(s,t);
    if(t==0) p[blockIdx.x]=s[0];
}
__device__ __forceinline__ float warp_sum(float v) {
    #pragma unroll
    for(int d=16;d>0;d/=2) v+=__shfl_down_sync(0xffffffffu,v,d);
    return v;
}
__device__ __forceinline__ float block_shuffle_sum(float v,float* warp_sums) {
    unsigned lane=threadIdx.x%32,warp=threadIdx.x/32;
    v=warp_sum(v);
    if(lane==0) warp_sums[warp]=v;
    __syncthreads();
    if(warp==0) {
        v=lane<blockDim.x/32?warp_sums[lane]:0.0f;
        v=warp_sum(v); // all 32 lanes participate; unused lanes hold zero.
    }
    return v;
}
// reduce7: same input distribution/grid as reduce6; warp shuffles replace tree.
// ATOMIC=false writes a partial; ATOMIC=true accumulates into a scalar.
template<unsigned B,bool ATOMIC=false> __global__ void reduce7(const float* x,float* p,size_t n) {
    __shared__ float warp_sums[B/32]; unsigned t=threadIdx.x; float v=0;
    size_t stride=size_t(gridDim.x)*(2*B);
    for(size_t i=size_t(blockIdx.x)*(2*B)+t;i<n;i+=stride) {
        v+=x[i]; if(i+B<n) v+=x[i+B];
    }
    v=block_shuffle_sum(v,warp_sums);
    if(t==0) {
        if constexpr(ATOMIC) atomicAdd(p,v);
        else p[blockIdx.x]=v;
    }
}
// Identical second stage for all reduce0..7. One block can consume any number
// of partials via grid-stride accumulation; a block barrier is not a grid barrier.
__global__ void finish_sum(const float* p,float* out,size_t count) {
    __shared__ float warp_sums[8]; float v=0;
    for(size_t i=threadIdx.x;i<count;i+=blockDim.x) v+=p[i];
    v=block_shuffle_sum(v,warp_sums);
    if(threadIdx.x==0) *out=v;
}

const char* names[]={"reduce0_modulo","reduce1_index","reduce2_sequential","reduce3_two_loads",
 "reduce4_warp_tail","reduce5_unrolled","reduce6_grid_stride","reduce7_shuffle","atomic_shuffle"};
struct Options {
    size_t n=1u<<24; int block=256,grid=0,version=-1,warmup=10,repeats=50,trials=5;
    std::string mode="both",csv; bool check_only=false;
};
int sm_count=0;
size_t blocks_for(int ver,size_t n,const Options& o) {
    size_t chunk=size_t(o.block)*(ver<3?1:2);
    size_t natural=std::max<size_t>(1,(n+chunk-1)/chunk);
    if(ver>=6) return std::min(natural,size_t(o.grid?o.grid:sm_count*16));
    return natural;
}
template<unsigned B> void launch_special(int ver,const float* x,float* p,size_t n,unsigned g) {
    if(ver==5) reduce5<B><<<g,B>>>(x,p,n);
    if(ver==6) reduce6<B><<<g,B>>>(x,p,n);
    if(ver==7) reduce7<B><<<g,B>>>(x,p,n);
    if(ver==8) reduce7<B,true><<<g,B>>>(x,p,n);
}
void launch(int ver,const float* x,float* p,size_t n,unsigned g,int b) {
    size_t bytes=b*sizeof(float);
    switch(ver) {
        case 0: reduce0<<<g,b,bytes>>>(x,p,n);break;
        case 1: reduce1<<<g,b,bytes>>>(x,p,n);break;
        case 2: reduce2<<<g,b,bytes>>>(x,p,n);break;
        case 3: reduce3<<<g,b,bytes>>>(x,p,n);break;
        case 4: reduce4<<<g,b,bytes>>>(x,p,n);break;
        default:
            switch(b) {
                case 128:launch_special<128>(ver,x,p,n,g);break;
                case 256:launch_special<256>(ver,x,p,n,g);break;
                case 512:launch_special<512>(ver,x,p,n,g);break;
                case 1024:launch_special<1024>(ver,x,p,n,g);break;
            }
    }
    CUDA(cudaGetLastError());
}
struct Buffers {
    float *x=nullptr,*p=nullptr,*out=nullptr;
    Buffers(size_t n,size_t count) {
        CUDA(cudaMalloc(&x,std::max<size_t>(1,n)*sizeof(float)));
        CUDA(cudaMalloc(&p,count*sizeof(float))); CUDA(cudaMalloc(&out,sizeof(float)));
    }
    ~Buffers(){cudaFree(x);cudaFree(p);cudaFree(out);}
};
void execute(int ver,const Options& o,Buffers& b,unsigned g,bool full) {
    if(ver==8) {
        // Included in atomic timing: every independent sum needs a zero scalar.
        CUDA(cudaMemsetAsync(b.out,0,sizeof(float)));
        launch(ver,b.x,b.out,o.n,g,o.block);
    } else {
        launch(ver,b.x,b.p,o.n,g,o.block);
        if(full) { finish_sum<<<1,256>>>(b.p,b.out,g); CUDA(cudaGetLastError()); }
    }
}
double reference(const std::vector<float>& x) {
    double total=0; for(float v:x) total+=double(v); return total;
}
void verify(int ver,const Options& o,Buffers& b,const std::vector<float>& x,bool exact=false) {
    unsigned g=unsigned(blocks_for(ver,o.n,o));
    if(!x.empty()) CUDA(cudaMemcpy(b.x,x.data(),x.size()*sizeof(float),cudaMemcpyHostToDevice));
    execute(ver,o,b,g,true);
    float actual;CUDA(cudaMemcpy(&actual,b.out,sizeof(float),cudaMemcpyDeviceToHost));
    double ref=reference(x),err=std::abs(double(actual)-ref);
    double tol=exact?0.0:1e-4+5e-5*std::max(std::abs(ref),std::sqrt(double(o.n)));
    if(!std::isfinite(actual)||err>tol) throw std::runtime_error(std::string(names[ver])+" correctness failed: error="+std::to_string(err)+" tolerance="+std::to_string(tol));
}
void correctness_suite(const Options& options) {
    // All cases run for the selected block/grid; --version all checks all nine.
    for(size_t n:{size_t(0),size_t(1),size_t(31),size_t(127),size_t(128),size_t(129),size_t(255),size_t(257),size_t(511),size_t(513),size_t(1025),size_t(65539)}) {
        Options o=options;o.n=n;Buffers b(n,blocks_for(0,n,o));
        std::vector<float> x(n);std::mt19937 rng(42);std::uniform_real_distribution<float> rand(-1,1);
        for(int pattern=0;pattern<3;++pattern) {
            for(size_t i=0;i<n;++i) x[i]=pattern==0?1.0f:(pattern==1?float(int(i%17)-8)*0.25f:rand(rng));
            for(int ver=0;ver<=8;++ver) if(o.version<0||ver==o.version) verify(ver,o,b,x,pattern<2);
        }
    }
    std::cout<<"PASS correctness: empty/tails/ones/signed-pattern/random; block="<<options.block<<" grid-cap="<<options.grid<<"\n";
}
double benchmark(int ver,const Options& o,Buffers& b,unsigned g,bool full) {
    for(int i=0;i<o.warmup;++i) execute(ver,o,b,g,full);
    CUDA(cudaDeviceSynchronize());
    cudaEvent_t start,end;CUDA(cudaEventCreate(&start));CUDA(cudaEventCreate(&end));
    std::vector<double> times;
    for(int t=0;t<o.trials;++t) {
        CUDA(cudaEventRecord(start));
        for(int r=0;r<o.repeats;++r) execute(ver,o,b,g,full);
        CUDA(cudaEventRecord(end));CUDA(cudaEventSynchronize(end));
        float ms;CUDA(cudaEventElapsedTime(&ms,start,end));times.push_back(ms/o.repeats);
    }
    CUDA(cudaEventDestroy(start));CUDA(cudaEventDestroy(end));
    std::sort(times.begin(),times.end());return times[times.size()/2];
}
Options parse(int argc,char** argv) {
    Options o;
    for(int i=1;i<argc;++i) {
        std::string key=argv[i];
        if(key=="--help") {std::cout<<"--n N --version all|0..7|atomic --block 128|256|512|1024 --grid CAP\n--mode stage1|full|both --warmup 10 --repeats 50 --trials 5 --csv PATH --check-only\n--grid affects only reduce6/reduce7/atomic; default cap=SM_count*16.\n";std::exit(0);}
        if(key=="--check-only") {o.check_only=true;continue;}
        if(i+1>=argc) throw std::runtime_error("missing value for "+key);
        std::string v=argv[++i];
        if(key=="--n") {if(v.empty()||v[0]=='-') throw std::runtime_error("n must be nonnegative");o.n=std::stoull(v);}
        else if(key=="--version") o.version=v=="all"?-1:(v=="atomic"?8:std::stoi(v));
        else if(key=="--block") o.block=std::stoi(v);
        else if(key=="--grid") o.grid=std::stoi(v);
        else if(key=="--warmup") o.warmup=std::stoi(v);
        else if(key=="--repeats") o.repeats=std::stoi(v);
        else if(key=="--trials") o.trials=std::stoi(v);
        else if(key=="--mode") o.mode=v;
        else if(key=="--csv") o.csv=v;
        else throw std::runtime_error("unknown option "+key);
    }
    if(o.block!=128&&o.block!=256&&o.block!=512&&o.block!=1024) throw std::runtime_error("block must be 128/256/512/1024");
    if(o.grid<0||o.warmup<0||o.repeats<=0||o.trials<=0||o.version< -1||o.version>8) throw std::runtime_error("invalid numeric option");
    if(o.mode!="stage1"&&o.mode!="full"&&o.mode!="both") throw std::runtime_error("mode must be stage1/full/both");
    if(o.version==8 && o.mode=="stage1") throw std::runtime_error("atomic has no separate stage1; use --mode full or both");
    if(o.n>(size_t(1)<<28)) throw std::runtime_error("learning size limit: N <= 268435456 (1 GiB input)");
    return o;
}
int main(int argc,char** argv) {try {
    Options o=parse(argc,argv);cudaDeviceProp prop{};CUDA(cudaGetDeviceProperties(&prop,0));sm_count=prop.multiProcessorCount;
    if(o.block>prop.maxThreadsPerBlock) throw std::runtime_error("block exceeds GPU limit");
    std::cout<<"GPU="<<prop.name<<" SM="<<prop.major<<'.'<<prop.minor<<" SM_count="<<sm_count<<" N="<<o.n<<" block="<<o.block<<"\n";
    correctness_suite(o);
    std::vector<float> x(o.n);std::mt19937 rng(42);std::uniform_real_distribution<float> rand(-1,1);
    for(float& v:x)v=rand(rng);
    Buffers b(o.n,blocks_for(0,o.n,o));
    if(o.check_only) {
        for(int ver=0;ver<=8;++ver) if(o.version<0||ver==o.version) verify(ver,o,b,x);
        std::cout<<"PASS requested-size checks: N="<<o.n<<" (FP64 CPU reference)\n";
        return 0;
    }
    std::ofstream csv;if(!o.csv.empty()){csv.open(o.csv);if(!csv)throw std::runtime_error("cannot open CSV: "+o.csv);csv<<"version,mode,n,block,blocks,ms,speedup_vs_reduce0,input_GB_s,abs_error,tolerance\n";}
    double baseline[2]={0,0};double ref=reference(x);
    std::cout<<"version,mode,n,block,blocks,ms,speedup_vs_reduce0,input_GB_s,abs_error,tolerance\n";
    // Measure reduce0 even for a single selected version, so speedup is defined.
    for(int mode=0;mode<2;++mode) {
        bool full=mode==1;if((full&&o.mode=="stage1")||(!full&&o.mode=="full"))continue;
        verify(0,o,b,x);baseline[mode]=benchmark(0,o,b,unsigned(blocks_for(0,o.n,o)),full);
        for(int ver=0;ver<=8;++ver) {
            if(o.version>=0&&ver!=o.version)continue;
            if(ver==8&&!full)continue; // atomic has no materialized stage1 output.
            verify(ver,o,b,x);float actual;CUDA(cudaMemcpy(&actual,b.out,sizeof(float),cudaMemcpyDeviceToHost));
            unsigned g=unsigned(blocks_for(ver,o.n,o));
            double ms=ver==0?baseline[mode]:benchmark(ver,o,b,g,full);
            double tol=1e-4+5e-5*std::max(std::abs(ref),std::sqrt(double(o.n)));
            auto row=[&](std::ostream& out){out<<names[ver]<<','<<(full?"full":"stage1")<<','<<o.n<<','<<o.block<<','<<g<<','<<std::setprecision(9)<<ms<<','<<baseline[mode]/ms<<','<<(double(o.n)*4/(ms*1e6))<<','<<std::abs(double(actual)-ref)<<','<<tol<<'\n';};
            row(std::cout);if(csv)row(csv);
        }
    }
    std::cout<<"PASS requested-size checks. Timing excludes allocation/H2D/D2H; warm-cache CUDA events. Atomic full includes output reset.\n";
    return 0;
} catch(const std::exception& e) {std::cerr<<"ERROR: "<<e.what()<<'\n';return 1;}}
