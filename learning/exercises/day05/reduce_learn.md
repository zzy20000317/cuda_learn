博客完整内容：
并行算法设计
从上至下，将数据不断地累加，直到得出最后的结果，即25。但由于GPU没有针对global数据的同步操作，只能针对block的数据进行同步。所以，一般而言将reduce分为两个阶段，其示意图如下：


我们仔细来看看这个事，假设给定一个长度为N的数组，需要计算该数组的所有元素之和。首先需要将数组分为m个小份。而后，在第一阶段中，开启m个block计算出m个小份的reduce值。最后，在第二阶段中，使用一个block将m个小份再次进行reduce，得到最终的结果。由于第二阶段本质上是可以调用第一个阶段的kernel，所以不做单独说明，本文只是探索第一阶段的优化技巧。

所以kernel接口为：

__global__ void reduce(T *input, T* output)
其中，input代表输入的数组，即一个长度为N的数组，output代表输出数组，即第一阶段的结果，即长度为M的数组。随后要开始激动人心的coding阶段，但在CUDA编程中，我们首先需要设置三个参数:

BlockNum：即开启的block数量，即上面所说的M，代表需要将数组切分为几份。
Thread_per_block:每个block中开启的线程数，一般而言，取128，256，512，1024这几个参数会比较多。
Num_per_block:每个block需要进行reduce操作的长度。
其中，BlockNum* Num_per_block=N
三个参数的示意图如下：


reduce baseline算法介绍
Baseline算法比较简单，分为三个步骤。第一个步骤是将数据load至shared memory中，第二个步骤是在shared memory中对数据进行reduce操作，第三个步骤是将最后的结果写回global memory中。代码如下：

__global__ void reduce0(float *d_in,float *d_out){
    __shared__ float sdata[THREAD_PER_BLOCK];

    //each thread loads one element from global memory to shared mem
    unsigned int i=blockIdx.x*blockDim.x+threadIdx.x;
    unsigned int tid=threadIdx.x;
    sdata[tid]=d_in[i];
    __syncthreads();

    // do reduction in shared mem
    for(unsigned int s=1; s<blockDim.x; s*=2){
        if(tid%(2*s) == 0){
            sdata[tid]+=sdata[tid+s];
        }
        __syncthreads();
    }
    
    // write result for this block to global mem
    if(tid==0)d_out[blockIdx.x]=sdata[tid];
}
在进行优化之前，我们需要再来好好地梳理一下这个baseline代码。优化的本质是通过软件榨干硬件资源，所以必须清楚地了解代码在硬件上的执行过程才能更好地进行优化。因此，本节将花较多的篇幅说明代码和硬件的对应关系，为后续的优化打好基础。

在第一个步骤中，我们让Num_per_block与Thread_per_block一致，每个block设定为256个线程，一个block负责256个数据的reduce工作。假设需要处理32M的数据，则有128K个block。tid代表线程号，i代表在原始数组中的索引号。第tid号线程将第i号的数据从global中取出，放到shared memory的第tid元素中。比如在第0号block中，0号线程将0号元素取出，放到shared memory的第0号位置。示意图见：


从硬件角度来分析一下代码。为了执行代码，GPU需要分配两种资源，一个是存储资源，一个是计算资源。存储资源包括在global memory中分配的一块32M
sizeof(float)的空间以及在shared memory中分配的256
sizeof(float)的空间。需要注意的是，shared memory存在bank冲突的问题，因而需要格外小心。 计算资源其实是根据thread数量来确定的，一个block中分配256个thread线程，32个线程为一组，绑定在一个SIMD单元。所以256个线程可以简单地理解为分配了8组SIMD单元。（但实际的硬件资源分配不是这样，因为一个SM的计算资源有限，不可能真的给每一个block都分配这么多的SIMD单元。）总而言之，在第一个阶段，就是tid号线程将i号数据从global memory中取出，再放进shared memory中，严谨一点的话，中间是走一遍寄存器再到shared memory中的。

到了第二个阶段，block中需要计算的256个元素已经全部被存储在了shared memory中，此时需要对其进行reduce操作。这个过程需要进行多轮迭代，在第一轮迭代中，如果tid%2 ==0, 则第tid号线程将shared memory中第tid号位置的值和第tid+1号的值进行相加，而后放在第tid号位置。在第二轮迭代中，如果tid%4==0,则第tid号线程将shared memory中第tid号位置的值和第tid+2号的值进行相加，而后放在第tid号位置。不断迭代，则所有元素都将被累加到第0号位置。其示意图如下。其中，红色的线程代表符合if条件的线程，只有它们有任务，需要干活。


在第三个阶段中，block负责的256个元素之和都放置在shared memory的0号位置，此时，只需要将0号位置的元素写回即可。

实验结果

在本次实验中，采用的GPU是V100。耗时结果是使用nsight测出来的。在V100 中global的带宽是900GB/s。可以看出，其带宽利用率较差，reduce0版本存在着较大的改进空间。

优化技巧1：解决warp divergence
现有问题
目前reduce0存在的最大问题就是warp divergent的问题。对于一个block而言，它所有的thread都是执行同一条指令。如果存在if-else这样的分支情况的话，thread会执行所有的分支。只是不满足条件的分支，所产生的结果不会记录下来。可以在上图中看到，在每一轮迭代中都会产生两个分支，分别是红色和橙色的分支。这严重影响了代码执行的效率。

解决方式
解决的方式也比较明了，就是尽可能地让所有线程走到同一个分支里面。
代码示意如下：

__global__ void reduce1(float *d_in,float *d_out){
    __shared__ float sdata[THREAD_PER_BLOCK];

    //each thread loads one element from global memory to shared mem
    unsigned int i=blockIdx.x*blockDim.x+threadIdx.x;
    unsigned int tid=threadIdx.x;
    sdata[tid]=d_in[i];
    __syncthreads();

    // do reduction in shared mem
    for(unsigned int s=1; s<blockDim.x; s*=2){
        int index = 2*s*tid;
        if(index < blockDim.x){
            sdata[index]+=sdata[index+s];
        }
        __syncthreads();
    }
    
    // write result for this block to global mem
    if(tid==0)d_out[blockIdx.x]=sdata[tid];
}
虽然代码依旧存在着if语句，但是却与reduce0代码有所不同。我们继续假定block中存在256个thread，即拥有256/32=8个warp。当进行第1次迭代时，0-3号warp的index<blockDim.x， 4-7号warp的index>=blockDim.x。对于每个warp而言，都只是进入到一个分支内，所以并不会存在warp divergence的情况。当进行第2次迭代时，0、1号两个warp进入计算分支。当进行第3次迭代时，只有0号warp进入计算分支。当进行第4次迭代时，只有0号warp的前16个线程进入分支。此时开始产生warp divergence。通过这种方式，我们消除了前3次迭代的warp divergence。

实验结果

优化技巧2：解决bank冲突
现有问题
reduce1的最大问题是bank冲突。我们把目光聚焦在这个for循环中。并且只聚焦在0号warp。在第一次迭代中，0号线程需要去load shared memory的0号地址以及1号地址的数，然后写回到0号地址。而此时，这个warp中的16号线程，需要去load shared memory中的32号地址和33号地址。可以发现，0号地址跟32号地址产生了2路的bank冲突。在第2次迭代中，0号线程需要去load shared memory中的0号地址和2号地址。这个warp中的8号线程需要load shared memory中的32号地址以及34号地址，16号线程需要load shared memory中的64号地址和68号地址，24号线程需要load shared memory中的96号地址和100号地址。又因为0、32、64、96号地址对应着同一个bank，所以此时产生了4路的bank冲突。现在，可以继续算下去，8路bank冲突，16路bank冲突。由于bank冲突，所以reduce1性能受限。下图说明了在load第一个数据时所产生的bank冲突。


解决方式
在reduce中，解决bank冲突的方式就是把for循环逆着来。原来stride从0到256，现在stride从128到0。其伪代码如下：

__global__ void reduce2(float *d_in,float *d_out){
    __shared__ float sdata[THREAD_PER_BLOCK];

    //each thread loads one element from global memory to shared mem
    unsigned int i=blockIdx.x*blockDim.x+threadIdx.x;
    unsigned int tid=threadIdx.x;
    sdata[tid]=d_in[i];
    __syncthreads();

    // do reduction in shared mem
    for(unsigned int s=blockDim.x/2; s>0; s>>=1){
        if(tid < s){
            sdata[tid]+=sdata[tid+s];
        }
        __syncthreads();
    }
    
    // write result for this block to global mem
    if(tid==0)d_out[blockIdx.x]=sdata[tid];
}
那为什么通过这么一个小小的改变就能消除bank冲突呢，我们继续进行分析。

把目光继续看到这个for循环中，并且只分析0号warp。0号线程需要load shared memory的0号元素以及128号元素。1号线程需要load shared memory中的1号元素和129号元素。这一轮迭代中，在读取第一个数时，warp中的32个线程刚好load 一行shared memory数据。再分析第2轮迭代，0号线程load 0号元素和64号元素，1号线程load 1号元素和65号元素。咦，也是这样，每次load shared memory的一行。再来分析第3轮迭代，0号线程load 0号元素和32号元素，接下来不写了，总之，一个warp load shared memory的一行。没有bank冲突。到了4轮迭代，0号线程load 0号元素和16号元素。那16号线程呢，16号线程啥也不干，因为s=16，16-31号线程啥也不干，跳过去了。示意图如下：


实验结果

优化技巧3：解决idle线程
现有问题
reduce2最大的问题就是线程的浪费。可以看到我们启动了256个线程，但是在第1轮迭代时只有128个线程在干活，第2轮迭代只有64个线程在干活，每次干活的线程都会减少一半。第一轮迭代示意图如下，只有前128个线程在load数据。后128个线程啥也不干，光看着。


解决方式
对于HPC从业者而言，我们希望变成GPU的资本家，去尽可能地压榨GPU。但是呢，在这里，每一次迭代有一半的线程不干活。而且，128-255号线程最过分，它娘的，没有任何贡献，啥也不干。想来想去，能不能让它们干点活呢。想来想去，那这样吧，让它好歹做一次加法。除了去global memory中取数外，再做一次加法。当然为了实现这个，block数就得改一改了。Block数量减少，Num_per_block增加一倍。也就是说原来一个block只需要管256个数就行，现在得管512个数了。代码如下：

__global__ void reduce3(float *d_in,float *d_out){
    __shared__ float sdata[THREAD_PER_BLOCK];

    //each thread loads one element from global memory to shared mem
    unsigned int i=blockIdx.x*(blockDim.x*2)+threadIdx.x;
    unsigned int tid=threadIdx.x;
    sdata[tid]=d_in[i] + d_in[i+blockDim.x];
    __syncthreads();

    // do reduction in shared mem
    for(unsigned int s=blockDim.x/2; s>0; s>>=1){
        if(tid < s){
            sdata[tid]+=sdata[tid+s];
        }
        __syncthreads();
    }
    
    // write result for this block to global mem
    if(tid==0)d_out[blockIdx.x]=sdata[tid];
}
通过这种方式，将一些idle的线程给利用起来了。

实验结果

这里面多说一句，让idle线程利用起来的这个加速比其实有点出乎意料。

优化技巧4：展开最后一维减少同步
现有问题
对于reduce3来说，性能已经算是比较好了。但是依旧没有达到我们想要的效果。我们再来仔细地看看还有什么可以改进的地方。我们发现，当进行到最后几轮迭代时，此时的block中只有warp0在干活时，线程还在进行同步操作。这一条语句造成了极大的浪费。

解决方式
由于一个warp中的32个线程其实是在一个SIMD单元上，这32个线程每次都是执行同一条指令，这天然地保持了同步状态，因而当s=32时，即只有一个SIMD单元在工作时，完全可以将__syncthreads()这条同步代码去掉。所以我们将最后一维进行展开以减少同步。伪代码如下：

__device__ void warpReduce(volatile float* cache,int tid){
    cache[tid]+=cache[tid+32];
    cache[tid]+=cache[tid+16];
    cache[tid]+=cache[tid+8];
    cache[tid]+=cache[tid+4];
    cache[tid]+=cache[tid+2];
    cache[tid]+=cache[tid+1];
}

__global__ void reduce4(float *d_in,float *d_out){
    __shared__ float sdata[THREAD_PER_BLOCK];

    //each thread loads one element from global memory to shared mem
    unsigned int i=blockIdx.x*(blockDim.x*2)+threadIdx.x;
    unsigned int tid=threadIdx.x;
    sdata[tid]=d_in[i] + d_in[i+blockDim.x];
    __syncthreads();

    // do reduction in shared mem
    for(unsigned int s=blockDim.x/2; s>32; s>>=1){
        if(tid < s){
            sdata[tid]+=sdata[tid+s];
        }
        __syncthreads();
    }
    
    // write result for this block to global mem
    if(tid<32)warpReduce(sdata,tid);
    if(tid==0)d_out[blockIdx.x]=sdata[tid];
}
可以通过下面的示意图更好地了解，（图画的有点丑，比例也不太对，大家将就着看看。）warp0会被绑定在一个SIMD单元上，上面有thread0-thread31。warp1会被绑在另外一个SIMD单元上，上面有thread32-thread63。由于在一个SIMD单元上，然后不管啥时候thread0和thread7肯定是同一状态，不需要同步。而thread0和thread34就不能保证同步，必须用__syncthreads()来保证同步操作。


实验结果

在做到这一步时，带宽已经到了756GB/s。这个时候就已经优化地差不多了。性能也很难做到有效提升了。

优化技巧5：完全展开减少计算
现有问题
其实到了这一步，reduce的效率已经足够高了。再进一步优化其实已经非常困难了。为了探索极致的性能表现，Mharris接下来给出的办法是对for循环进行完全展开。我觉得这里主要是减少for循环的开销。Mharris的实验表明这种方式有着1.41x的加速比。但是用的机器是G80，十几年前的卡。性能数据也比较老了，至于能不能真的有这么好的加速比，我们拭目以待。

解决方法
我们将整个for循环进行展开，非常暴力，代码如下：

template <unsigned int blockSize>
__device__ void warpReduce(volatile float* cache,int tid){
    if(blockSize >= 64)cache[tid]+=cache[tid+32];
    if(blockSize >= 32)cache[tid]+=cache[tid+16];
    if(blockSize >= 16)cache[tid]+=cache[tid+8];
    if(blockSize >= 8)cache[tid]+=cache[tid+4];
    if(blockSize >= 4)cache[tid]+=cache[tid+2];
    if(blockSize >= 2)cache[tid]+=cache[tid+1];
}

template <unsigned int blockSize>
__global__ void reduce5(float *d_in,float *d_out){
    __shared__ float sdata[THREAD_PER_BLOCK];

    //each thread loads one element from global memory to shared mem
    unsigned int i=blockIdx.x*(blockDim.x*2)+threadIdx.x;
    unsigned int tid=threadIdx.x;
    sdata[tid]=d_in[i] + d_in[i+blockDim.x];
    __syncthreads();

    // do reduction in shared mem
    if(blockSize>=512){
        if(tid<256){
            sdata[tid]+=sdata[tid+256];
        }
        __syncthreads();
    }
    if(blockSize>=256){
        if(tid<128){
            sdata[tid]+=sdata[tid+128];
        }
        __syncthreads();
    }
    if(blockSize>=128){
        if(tid<64){
            sdata[tid]+=sdata[tid+64];
        }
        __syncthreads();
    }
    
    // write result for this block to global mem
    if(tid<32)warpReduce<blockSize>(sdata,tid);
    if(tid==0)d_out[blockIdx.x]=sdata[tid];
}
实验结果

可以看到，还是有所收益，但是并没有那么地显著。这主要是因为GPU硬件架构的不断发展，以及NV在编译器上面也做了较多的工作。

优化技巧6：合理设置block数量
现有问题
当走到这一步的时候，能调的东西已经基本上调完了。我们再把眼光放在block和thread的设置上。之前默认了Num_per_block=Thread_per_block。也就是说，一个block开启256个线程时，这个block负责256个元素的reduce操作。那可不可以让一个block多管点数。这样的话，开启的block数量少一些。以此对block设置进行调整，获得最优block取值，这样或许能够带来一些性能收益？

解决方式
这样需要再思考一下block的取值。对于GPU而言，block的取值到底是多更好，还是少更好。如此对CUDA编程熟悉的同学，肯定会毫不犹豫地说：“那肯定是多更好啦。Block数量多，block可以进行快速地切换，去掩盖访存的延时。”这个问题按下不表，我们看看Mharris是怎么说的。

如果一个线程被分配更多的work时，可能会更好地覆盖延时。这一点比较好理解。如果线程有更多的work时，对于编译器而言，就可能有更多的机会对相关指令进行重排，从而去覆盖访存时的巨大延时。虽然这句话并没有很好地说明在某种程度上而言，block少一些会更好。但是，有一点不可否认,block需要进行合理地设置。唠唠叨叨说了很多，现在把代码贴一下：

template <unsigned int blockSize>
__global__ void reduce6(float *d_in,float *d_out){
    __shared__ float sdata[THREAD_PER_BLOCK];

    //each thread loads one element from global memory to shared mem
    unsigned int i=blockIdx.x*(blockDim.x*2)+threadIdx.x;
    unsigned int tid=threadIdx.x;
    unsigned int gridSize = blockSize * 2 * gridDim.x;
    sdata[tid] = 0;

    while(i<n){
        sdata[tid] +=d_in[i]+d_in[i+blockSize];
        i+=gridSize;
    }
    __syncthreads();

    // do reduction in shared mem
    if(blockSize>=512){
        if(tid<256){
            sdata[tid]+=sdata[tid+256];
        }
        __syncthreads();
    }
    if(blockSize>=256){
        if(tid<128){
            sdata[tid]+=sdata[tid+128];
        }
        __syncthreads();
    }
    if(blockSize>=128){
        if(tid<64){
            sdata[tid]+=sdata[tid+64];
        }
        __syncthreads();
    }
    
    // write result for this block to global mem
    if(tid<32)warpReduce<blockSize>(sdata,tid);
    if(tid==0)d_out[blockIdx.x]=sdata[tid];
}
实验结果

对于block的取值，我进行了微调，大概取2048会比512，1024，4096的效果要好。理论上来说，这个值取SM数量的倍数会比较合理。但是V100的SM是80，取一个完美的倍数还是比较困难。目前达到了768GB/s。

优化技巧7：使用shuffle指令
现有问题
其实，对于Mharris的讲义。reduce优化就到此结束了。但是NV后来出了Shuffle指令，对于reduce优化有着非常好的效果。目前绝大多数访存类算子，像是softmax，batch_norm，reduce等，都是用Shuffle实现。所以，在这里谈一下这么把shuffle指令用在reduce优化上。

Shuffle指令是一组针对warp的指令。Shuffle指令最重要的特性就是warp内的寄存器可以相互访问。在没有shuffle指令的时候，各个线程在进行通信时只能通过shared memory来访问彼此的寄存器。而采用了shuffle指令之后，warp内的线程可以直接对其他线程的寄存器进行访存。通过这种方式可以减少访存的延时。除此之外，带来的最大好处就是可编程性提高了，在某些场景下，就不用shared memory了。毕竟，开发者要自己去控制 shared memory还是挺麻烦的一个事。

伪代码如下：

template <unsigned int blockSize>
__device__ __forceinline__ float warpReduceSum(float sum){
    if(blockSize >= 32)sum += __shfl_down_sync(0xffffffff,sum,16);
    if(blockSize >= 16)sum += __shfl_down_sync(0xffffffff,sum,8);
    if(blockSize >= 8)sum += __shfl_down_sync(0xffffffff,sum,4);
    if(blockSize >= 4)sum += __shfl_down_sync(0xffffffff,sum,2);
    if(blockSize >= 2)sum += __shfl_down_sync(0xffffffff,sum,1);
    return sum;
}

template <unsigned int blockSize>
__global__ void reduce7(float *d_in,float *d_out, unsigned int n){
    float sum = 0;

    //each thread loads one element from global memory to shared mem
    unsigned int i=blockIdx.x*(blockDim.x*2)+threadIdx.x;
    unsigned int tid=threadIdx.x;
    unsigned int gridSize = blockSize * 2 * gridDim.x;

    while(i<n){
        sum +=d_in[i]+d_in[i+blockSize];
        i+=gridSize;
    }

    // shared mem for partial sums(one per warp in the block
    static __shared__ float warpLevelSums[WARP_SIZE];
    const int laneId = threadIdx.x % WARP_SIZE;
    const int warpId = threadIdx.x / WARP_SIZE;

    sum = warpReduceSum<blockSize>(sum);

    if(laneId == 0)warpLevelSums[warpId]=sum;
    __syncthreads();

    sum = (threadIdx.x < blockDim.x / WARP_SIZE)? warpLevelSums[laneId]:0;
    // Final reduce using first warp
    if(warpId == 0)sum = warpReduceSum<blockSize/WARP_SIZE>(sum);
    // write result for this block to global mem
    if(tid==0)d_out[blockIdx.x]=sum;
}

总结与思考
通过上次一系列的优化技巧，我们对reduce进行了不断地优化。最后效果是770.3GB/s。带宽利用达到85%。PS：具体的reduce性能数据还需要大量测试，但由于我比较懒。所以测试工作就到此为此，大家有兴趣可以自己再跑跑。代码在这里，还有点乱，没怎么整理，大家有疑问可以直接评论或者私信我。

可以看到，其实在reduce4的时候就已经很难再提升了。而且这个数据跟NV在博客上的数据还是有比较大的出入，主要是对最后一维展开后，在V100上的带宽利用率就很难再提升了，而G80还有显著的提升。我觉得主要原因是硬件已经更新了好几代。

AI总结：
归约优化全景：从 baseline 到 shuffle，以及两种实现的异同
你这个总结做得很到位——把博客里的 7 步优化和之前 reduce_sum.cu 放在一起看，确实能看出一套完整的设计哲学。下面我分三块讲：每一步优化解决什么硬件问题、两种实现的异同、跨 block 通信的真相。

第一部分：7 步优化，每一步解决什么硬件问题
先建立一个心智模型：归约的瓶颈不外乎四类——指令发散、访存冲突、线程浪费、同步开销。每一步优化都在打掉其中一个。

Baseline（reduce0）：起点，三大问题全暴露
cpp
for (unsigned int s = 1; s < blockDim.x; s *= 2) {
    if (tid % (2*s) == 0) {
        sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
}
问题 1：warp divergence

tid % (2*s) == 0 这个条件，warp 里一半线程满足、一半不满足。GPU 是 SIMT 执行：一个 warp 的所有 lane 必须执行同一条指令。遇到分支时，硬件会让两部分都执行一遍，只是用谓词屏蔽掉不需要的结果。

举例：warp 里 tid=0..31，条件 tid % 2 == 0 时，0,2,4,...,30 满足，1,3,5,...,31 不满足。硬件先让"满足组"执行 sdata[tid] += sdata[tid+1]，再让"不满足组"空转（或反过来），执行时间翻倍。

问题 2：bank conflict（shared memory 的隐藏陷阱）

shared memory 被划分为 32 个 bank，每个 bank 4 字节。地址 addr 属于 bank = (addr / 4) % 32。

如果 warp 里 32 个线程访问 32 个不同 bank → 1 个周期完成；

如果 两个线程访问同一个 bank 的不同地址 → 串行化，2 个周期；

这就是 bank conflict。

reduce0 第 1 轮：线程 0 读 sdata[0] 和 sdata[1]，线程 16 读 sdata[32] 和 sdata[33]。

sdata[0] 和 sdata[32] 都在 bank 0（因为 0/4%32 = 0，32/4%32 = 0）。冲突！

第 2 轮：线程 0 读 sdata[0]、线程 8 读 sdata[32]、线程 16 读 sdata[64]、线程 24 读 sdata[96]——全部 bank 0，4 路冲突。越往后越严重。

问题 3：idle 线程

后面几轮只有 128、64、32 个线程干活，其他线程都在 __syncthreads() 处干等。

reduce1：只解决 warp divergence
cpp
int index = 2*s*tid;
if (index < blockDim.x) {
    sdata[index] += sdata[index+s];
}
关键改动：把"每个线程处理 tid 对应的元素"改成"每个线程处理 index = 2*s*tid 对应的元素"。

第 1 轮 s=1：index = 2*tid，只有 tid < 128 时 index < 256，所以 warp 0-3（tid 0-127）全干，warp 4-7（tid 128-255）全不干。每个 warp 内部完全一致，没有 divergence。

第 2 轮 s=2：index = 4*tid，tid < 64 才干活。warp 0-1 干，warp 2-7 不干。也没有 divergence。

第 4 轮 s=8：index = 16*tid，tid < 16 才干活。warp 0 里只有 lane 0-15 干，lane 16-31 不干。开始 divergence。

所以 reduce1 消除了前几轮的 divergence，后几轮仍存在。

但 bank conflict 更严重了：index = 2*s*tid 让相邻线程访问间隔很大，更容易撞同一 bank。

reduce2：解决 bank conflict
cpp
for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
        sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
}
关键改动：把循环方向反过来——stride 从 blockDim.x/2 递减到 1。

分析 warp 0（tid 0..31）第 1 轮（s=128）：

线程 0 读 sdata[0] 和 sdata[128] → bank 0 和 bank 0？等等，128/4 = 32，32%32 = 0 → 还是 bank 0。

看起来还是冲突？ 不，关键在于同一时刻要读的地址。SIMT 执行分两步：先读 sdata[tid]，再读 sdata[tid+s]。

第 1 步：warp 里 32 个线程读 sdata[0..31] → 32 个不同 bank，零冲突。

第 2 步：读 sdata[128..159] → 128/4=32, 32%32=0，所以 sdata[128] 是 bank 0，sdata[129] 是 bank 1，...，sdata[159] 是 bank 31 → 还是 32 个不同 bank，零冲突。

关键洞察：只要 warp 内 32 个线程访问的地址连续 32 个，就一定占满 32 个 bank，零冲突。reduce2 通过 tid < s 保证每次读的是连续地址。

后面几轮同理。这就是 reduce2 消除 bank conflict 的本质：让 warp 访问连续的 shared memory 地址。

reduce3：解决 idle 线程
cpp
unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
sdata[tid] = d_in[i] + d_in[i + blockDim.x];
关键改动：每个线程加载 2 个 global 元素，先加一次再放进 shared。

原来一个 block 处理 256 个元素；

现在一个 block 处理 512 个元素；

线程 255 原来只 load 一次就闲置，现在也参与一次加法。

为什么值得：global load 是延迟大的操作。让 256 个线程多干点活，掩盖 load 延迟。这就是博客里说的"让 idle 线程干点活"。

副作用：grid 的 block 数减半（N 不变，每 block 处理 512）。这其实也是好事——减少 block 间原子/写回竞争。

reduce4：最后一维展开，去同步
cpp
__device__ void warpReduce(volatile float* cache, int tid) {
    cache[tid] += cache[tid + 32];
    cache[tid] += cache[tid + 16];
    cache[tid] += cache[tid + 8];
    cache[tid] += cache[tid + 4];
    cache[tid] += cache[tid + 2];
    cache[tid] += cache[tid + 1];
}

for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
    if (tid < s) cache[tid] += cache[tid + s];
    __syncthreads();
}
if (tid < 32) warpReduce(sdata, tid);
关键洞察：当 s ≤ 32 时，只有一个 warp 在工作（warp 0）。warp 内 32 个 lane 是锁步执行的（同一条指令），硬件天然保证它们同步，不需要 __syncthreads()。

__syncthreads() 的代价：它是一条 barrier，要等 block 内所有线程到达。block 有 256 个线程，就算只有 warp0 在干活，也要等另外 7 个 warp 到 barrier。

volatile 关键字：告诉编译器"这个指针指向的内存可能被其他线程改，不要缓存到寄存器"。在这个场景下，warpReduce 内部各个线程会读其他线程写过的 shared 地址，不加 volatile 编译器可能把它优化掉。现代 CUDA 中 __shfl_* 更推荐（见 reduce7），但早期没这选项时 volatile 是必需的。

reduce5：完全展开 for 循环
cpp
if (blockSize >= 512) { if (tid < 256) sdata[tid] += sdata[tid+256]; __syncthreads(); }
if (blockSize >= 256) { if (tid < 128) sdata[tid] += sdata[tid+128]; __syncthreads(); }
if (blockSize >= 128) { if (tid < 64)  sdata[tid] += sdata[tid+64];  __syncthreads(); }
if (tid < 32) warpReduce<blockSize>(sdata, tid);
关键改动：把循环完全展开，每步用 if (blockSize >= X) 判断。

好处：

编译器可以在编译期知道分支是否会出现——blockSize >= 512 是 constexpr，编译器直接删掉不可能的代码；

减少循环控制开销（比较、跳转、迭代变量维护）；

允许编译器做更激进的指令重排。

模板参数 blockSize：让编译器针对每个 block 大小生成一份专用代码。

reduce6：网格跨步 + 合理 block 数
cpp
unsigned int gridSize = blockSize * 2 * gridDim.x;
sdata[tid] = 0;
while (i < n) {
    sdata[tid] += d_in[i] + d_in[i + blockSize];
    i += gridSize;
}
关键改动：用 while 循环让每个 block 处理多段数据。

为什么 block 数量要控制在合理范围：

block 太多（几万个）：调度开销、原子冲突、cache 抖动；

block 太少（SM 数都不到）：无法掩盖延迟；

经验值：SM 数量的几倍到几十倍。V100 有 80 个 SM，取 2048 效果最好。

网格跨步的本质：和昨天你写的 add 里的 for (i = idx; i < n; i += stride) 一样——让少量线程循环处理大量数据。

reduce7：用 shuffle 替代 shared memory
cpp
template <unsigned int blockSize>
__device__ __forceinline__ float warpReduceSum(float sum) {
    if (blockSize >= 32) sum += __shfl_down_sync(0xffffffff, sum, 16);
    if (blockSize >= 16) sum += __shfl_down_sync(0xffffffff, sum, 8);
    if (blockSize >= 8)  sum += __shfl_down_sync(0xffffffff, sum, 4);
    if (blockSize >= 4)  sum += __shfl_down_sync(0xffffffff, sum, 2);
    if (blockSize >= 2)  sum += __shfl_down_sync(0xffffffff, sum, 1);
    return sum;
}
关键洞察：warp 内的数据交换根本不用过 shared memory——用 __shfl_* 指令直接在寄存器间传。

原来：shared_store + barrier + shared_load，3 步、几微秒；

shuffle：1 条指令、几纳秒。

为什么比 shared memory 快：

不过 LSU（Load/Store Unit），不占访存带宽；

不需要 __syncthreads()；

寄存器到寄存器，1 个周期。

这就是现代 reduce 的实现方式。你之前看到的 _warp_shuffle_reduce 就是同一个东西。

第二部分：两种实现的异同
现在对照博客版（reduce0~7）和你之前看的 reduce_sum.cu。

相同点
维度	博客版	reduce_sum.cu
第一阶段结构	block 内归约 → 输出部分和	完全一样
warp 内归约	reduce7 用 shuffle	用 _warp_shuffle_reduce
网格跨步	reduce6 引入	所有 kernel 都有
模板	reduce5 起	全程用
不同点
维度	博客版	reduce_sum.cu
输出策略	每个 block 输出 d_out[blockIdx.x]	atomicAdd 到单一输出
第二阶段	单独 kernel（或同 kernel 跑一遍）	无——atomicAdd 一步到位
dtype 支持	只有 fp32	fp32/fp16/fp8/int8 全支持
向量化	无	fp32x4、fp16x2、fp16x8、i8x16、dp4a
结果确定性	确定（block 顺序无关）	不确定（atomicAdd 顺序随机）
block 上限	无特殊限制	512 或 1024（控制原子冲突）
模板参数	blockSize	block_size + T（类型）
两种设计哲学
博客版 = "阶段分离"：

阶段 1 写 d_out[M]，M = block 数；

阶段 2 对 M 个值再 reduce；

优点：结果确定、易调试；

缺点：两次 kernel launch，M 个中间值要存。

reduce_sum.cu = "原子合并"：

每个 block 直接 atomicAdd 到最终输出；

优点：一次 kernel、代码简洁；

缺点：原子冲突、浮点非确定性。

实战推荐：

大规模 + fp32：用两阶段（博客版），确定性重要；

小规模 + 混合 dtype：用 atomicAdd（reduce_sum.cu），因为结果差几十 ULP 无所谓；

要极致性能：两阶段 + 第二次用 1 个 block 做最后归约。

第三部分：跨 block 通信的真相
这是你今天最关键的疑问，值得掰开揉碎讲。

博客版到底有没有跨 block 通信？
没有！ 严格说：

cpp
if (tid == 0) d_out[blockIdx.x] = sdata[tid];
block 0 写 d_out[0]；

block 1 写 d_out[1]；

...

block 999 写 d_out[999]。

每个 block 写自己专属的地址，互不干扰。这本质上不是"通信"，而是"并行输出"。block 之间完全独立——这才是 GPU 最喜欢的模式。

之所以叫"两阶段"，是因为之后要有第二次 kernel 把 d_out[0..999] 再归约。但第二次 kernel 是另一个独立的 launch，不是同一个 grid 内的 block 通信。

reduce_sum.cu 的跨 block 通信
cpp
if (tid == 0) atomicAdd(b, sum);
所有 block 都写 *b 这一个地址。这才是真正的跨 block 通信——多个 block 同时读写同一内存。

atomicAdd 保证"读-改-写"是原子的，不会丢结果。

为什么"跨 block 不能用 __syncthreads()"，而博客版"看起来简单完成了"？
这是两回事：

博客版没有做真正的跨 block 通信——只是各 block 写各自地址。不需要同步，因为不冲突。

__syncthreads() 的禁止针对的是"让一个 block 等另一个 block 完成"。这是不允许的，因为：

GPU 一次只能同时跑几十个 block，剩下的几千个在排队；

如果 block 0 要等 block 999 完成，但 block 999 根本还没被调度 → 死锁；

所以 CUDA 从硬件层面禁止跨 block 的 __syncthreads()。

atomicAdd 不违反这条规则——它不需要等任何 block 完成。无论谁先谁后，atomicAdd 都能正确累加。最终结果是"所有 block 都到达"时才正确，但每个 block 不需要等待。

一句话总结三种情况
场景	是否需要同步	怎么做
各 block 写各自地址	不需要	直接写
各 block 累加到同一地址	不需要等待，但需要原子性	atomicAdd
block A 要读 block B 的结果	禁止	拆成两个 kernel
用你的 reduce_sum.cu 举例
cpp
for (int i = idx; i < N; i += gridDim.x * blockDim.x) {
    sum += a[i];
}
// ...
if (tid == 0) atomicAdd(b, sum);
这里没有任何跨 block 同步——所有 block 并行跑，每个 block 算完自己的部分和就直接 atomicAdd。谁先谁后都行，硬件调度器自由决定。这正是 GPU 高效的原因。

第四部分：把两个问题合起来看
你之前问的"跨 block 通信简单通过 threadid 和 blockdim 完成了"——现在应该清楚了：

i = blockIdx.x * blockDim.x + threadIdx.x 是在算这个线程该处理哪个 global 元素。它跟"跨 block 通信"没有关系。

真正涉及跨 block 通信的只有两处：

写输出：d_out[blockIdx.x] = sdata[0]——各写各的，无同步；

atomicAdd：atomicAdd(b, sdata[0])——累加到同一地址，需要原子但不需要 barrier。

"跨 block 通信"这个词有些误导——GPU 上不存在"block 间对话"，只有"多个 block 访问同一内存"。处理同一内存有两种策略：分开写（各写各的地址）或原子写（atomicAdd）。前者简单但需要第二阶段，后者直接但结果非确定。

今日验收自检
对着这几个问题讲一遍：

baseline 的三大问题：warp divergence、bank conflict、idle 线程。

reduce1 为什么能消除 divergence：改变索引让每个 warp 内部分支一致。

reduce2 为什么能消除 bank conflict：让 warp 每次访问连续 32 个 shared 地址，占满 32 个 bank。

为什么 s ≤ 32 时不需要 __syncthreads()：只有一个 warp 在工作，warp 内锁步执行。

reduce6 为什么限制 block 数：太多造成调度和原子冲突，太少掩盖不了延迟。

__shfl_down_sync 为什么比 shared memory 快：寄存器直传，不过 LSU，不需要 barrier。

博客版和 reduce_sum.cu 的差异：

博客版：两阶段，写不同地址，结果确定；

reduce_sum.cu：atomicAdd，单 kernel，结果非确定；

都用了 warp shuffle 做 warp 内归约。

跨 block "通信"的真相：

博客版没做真正的跨 block 通信；

atomicAdd 是唯一的真跨 block 通信方式；

__syncthreads() 只作用 block 内，跨 block 同步会死锁。
