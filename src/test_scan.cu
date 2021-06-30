#include <cuda.h>
#include <stdio.h>
#include <math.h>
#include <sys/time.h>

#define gray_t double
#define PI 3.1415926535897
#define mymax(x, y) ((x) < (y) ? (y) : (x))
#define mymin(x, y) ((x) > (y) ? (y) : (x))
#define myabs(x) ((x) < 0? (-(x)) : (x))
// 检查以cuda开头的api调用是否出错或已经出错
#define checkCudaErrors(call)												\
{																			\
	const cudaError_t error = call;											\
	if (error != cudaSuccess) {												\
		printf("Error: %s:%d, ", __FILE__, __LINE__);						\
		printf("code:%d, reason: %s \n", error, cudaGetErrorString(error));	\
		exit(1);															\
	}																		\
}

// from CUDA7.pdf
// 计算 每个block内部的前缀和 并记录block sum（block内前缀和数组的最后一个值）的前缀和
__global__ void scan(gray_t* out, gray_t* block_sums, gray_t* data) {
    extern __shared__ gray_t s_data[];
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    s_data[threadIdx.x] = data[tid];

    for (int stride = 1; stride < blockDim.x; stride <<= 1) {
        __syncthreads();
        gray_t val = (threadIdx.x >= stride) ? s_data[threadIdx.x - stride] : 0;
        __syncthreads();
        s_data[threadIdx.x] += val;
    }

    out[tid] = s_data[threadIdx.x];
    if (threadIdx.x == 0) {
        for (int i = blockIdx.x + 1; i < gridDim.x; ++i) {
            atomicAdd(&block_sums[i], s_data[blockDim.x - 1]);
        } 
    }
}

// 将blocksum前缀和添加到block内部前缀和的每一项
__global__ void scan_update(gray_t* out, gray_t* block_sums) {
    __shared__ gray_t block_sum;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (threadIdx.x == 0) {
        block_sum  = block_sums[blockIdx.x];
    }

    __syncthreads();

    out[idx] += block_sum;
}

int diveup(int total, int threadnum){
    return (total % threadnum ? (total / threadnum + 1) : (total / threadnum));
}

void test_scan() {
    double sigma = 150;
    int filter_size = int(sigma * 3 * 2 + 1) | 1;
    int mid = filter_size >> 1;
    gray_t* filter = new gray_t[mid + 1]; 
    gray_t* pre_filter = new gray_t[mid + 1]; // pre_filter[i]表示sum(filter[0], ..., filter[i])
    gray_t* cpu_pre_filter = new gray_t[mid + 1]; 
    double total = 0;

    for (int i = 0; i < mid + 1; ++i) {
        filter[i] = 1 / (sqrt(2 * PI) * sigma) * exp((- (i - mid) * (i - mid)) / (2 * sigma * sigma));
        total += 2 * filter[i];
    }

    total -= filter[mid];
    for (int i = 0; i < mid + 1; ++i) {
        filter[i] /= total;
    }

    gray_t* device_filter;
    checkCudaErrors(cudaMalloc((void **) &device_filter, sizeof(gray_t) * (mid + 1)));
    checkCudaErrors(cudaMemcpy(device_filter, filter, sizeof(gray_t) * (mid + 1), cudaMemcpyHostToDevice));
    // copy  filter to pre_filter
    int block_num = diveup(mid + 1, 32);
    gray_t* device_pre_filter;
    gray_t* block_sums;
    checkCudaErrors(cudaMalloc((void **) &block_sums, sizeof(gray_t) * (block_num)));
    checkCudaErrors(cudaMalloc((void **) &device_pre_filter, sizeof(gray_t) * (mid + 1)));

    scan<<<block_num, 32, sizeof(gray_t)*32>>>(device_pre_filter, block_sums, device_filter);
    checkCudaErrors(cudaDeviceSynchronize());
    scan_update<<<block_num, 32>>>(device_pre_filter, block_sums);
    checkCudaErrors(cudaDeviceSynchronize());

    checkCudaErrors(cudaMemcpy(pre_filter, device_pre_filter, sizeof(gray_t) * (mid + 1), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaFree(device_filter));
    checkCudaErrors(cudaFree(block_sums));
    checkCudaErrors(cudaFree(device_pre_filter));
    
    cpu_pre_filter[0] = filter[0];
    for (int i = 1; i < mid + 1; ++i) {
        cpu_pre_filter[i] = filter[i] + cpu_pre_filter[i - 1];
    }

    for (int i = 0; i < mid + 1; ++i) {
        printf("%lf ", cpu_pre_filter[i] - pre_filter[i]);
    }
}

__global__ void cal_filter(gray_t* filter, gray_t* total, int mid, double sigma) {
    extern __shared__ gray_t sdata[];
    // 计算filter
    int fi = blockIdx.x * blockDim.x + threadIdx.x;  // not unsigned int
    int tidx = threadIdx.x;
    if (fi <= mid) {
        filter[fi] = 1 / (sqrt(2 * PI) * sigma) * exp((- (fi - mid) * (fi - mid)) / (2 * sigma * sigma));
    }
    if (tidx == 0) {
        total[0] = 0.0;
    }
    __syncthreads(); // 确保当前线程块所需的filter计算完毕
    
    // 将filter加载到共享内存
    sdata[tidx] = fi <= mid ? filter[fi] : 0;
    __syncthreads();
    
    // 规约：把当前线程块的filter之和存放到共享内存sdata[0]
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tidx < s) {
            sdata[tidx] += sdata[tidx + s];
        }
        __syncthreads();
    }
    
    // 把所有线程块结果用原子加法累加到全局遍历total[0]上
    if (tidx == 0) {
        atomicAdd(total, 2.0 * sdata[0]);
    }
    if (fi == mid) {
        atomicAdd(total, - filter[mid]);
    }
}

__global__ void div_filter(gray_t* filter, gray_t* total, int mid, int loop) {
    // 计算filter
    int fi = blockIdx.x * blockDim.x + threadIdx.x;
    int sta = fi * loop;
    int end = mymin(sta + loop, mid + 1);
    for (int i = sta; i < end; ++i) {
        filter[i] /= total[0];
    }
}

void test_reduce() {
    double sigma = 150;
    int filter_size = int(sigma * 3 * 2 + 1) | 1;
    int mid = filter_size >> 1;
    gray_t* filter = new gray_t[mid + 1]; 
    gray_t* cpu_filter = new gray_t[mid + 1]; 
    double total = 0;


    for (int i = 0; i < mid + 1; ++i) {
        cpu_filter[i] = 1 / (sqrt(2 * PI) * sigma) * exp((- (i - mid) * (i - mid)) / (2 * sigma * sigma));
        total += cpu_filter[i];
    }

    total = 2 * total - cpu_filter[mid];
    for (int i = 0; i < mid + 1; ++i) {
        cpu_filter[i] /= total;
    }
    printf("[%lf]", total);
    gray_t* device_total;
    gray_t* device_filter;
    checkCudaErrors(cudaMalloc((void **) &device_total, sizeof(gray_t)));
    checkCudaErrors(cudaMalloc((void **) &device_filter, sizeof(gray_t) * (mid + 1)));
    cal_filter<<<diveup((mid + 1), 32), 32, sizeof(gray_t)*32>>>(device_filter, device_total, mid, sigma);
    checkCudaErrors(cudaDeviceSynchronize());
    div_filter<<<diveup((mid + 1), 32 * 8), 32>>>(device_filter, device_total, mid, 8);
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaMemcpy(filter, device_filter, sizeof(gray_t) * (mid + 1), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(&total, device_total, sizeof(gray_t), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaFree(device_filter));
    checkCudaErrors(cudaFree(device_total));
    for (int i = 0; i < mid + 1; ++i) {
        printf("%lf ", filter[i] - cpu_filter[i]);
    }
}

__global__ void conv1(gray_t* img_src, gray_t* pre_filter, gray_t* filter, gray_t* temp_res, int n, int m, int mid){  
    int i = threadIdx.x+blockDim.x*blockIdx.x;
    int j = threadIdx.y+blockDim.y*blockIdx.y;
    if (i < n && j < m) {
        int pos = j + i * m;
        gray_t temp = 0;
        int i_sta = i - mid;
        int i_end = i + mid;
        // 当前行的卷积范围：[i-mid, i+mid]
        if (i - mid < 0) {
            // 合并计算[i-mid, 0)部分，即原filter的前mid-i个参数与mid-i个填充值（列首元素）的点乘
            temp += pre_filter[mid - i - 1] * img_src[j];
            i_sta = 0;
        }
        if (i + mid >= n) {
            // 合并计算(n-1, i+mid] 部分，即原filter的后xx=i+mid+1-n个参数（由于对称性 等价于前xx个）与xx个填充值（行尾元素）的点乘
            temp += pre_filter[i + mid - n] * img_src[(n - 1) * m + j];
            i_end = n - 1;
        }
        for (int xi = i_sta; xi <= i_end; ++xi) {
            // 第xi个元素离卷积中心i的距离为xi-i 使用的是距离卷积核中心mid距离为xi-i的卷积参数
            temp += filter[mid - myabs(i - xi)] * img_src[xi * m + j];
        }
        temp_res[pos] = temp;
    }
    
}

__global__ void conv2(gray_t* temp_res, gray_t* pre_filter, gray_t* filter, gray_t* img_dst, int n, int m, int mid){  
    int i = threadIdx.x+blockDim.x*blockIdx.x;
    int j = threadIdx.y+blockDim.y*blockIdx.y;
    if (i < n && j < m) {
        int pos = i * m + j;
        gray_t temp = 0;
        int j_sta = j - mid;
        int j_end = j + mid;
        // gray_t tmp = 0;
        // 当前行的卷积范围：[j-mid, j+mid]
        if (j - mid < 0) {
            // 合并计算[j-mid, 0)部分，即原filter的前mid-j个参数与mid-j个填充值（行首元素）的点乘
            temp += pre_filter[mid - j - 1] * temp_res[i * m];
            j_sta = 0;
        }
        if (j + mid >= m) {
            // 合并计算(m-1, j+mid] 部分，即原filter的后xx=j+mid+1-m个参数（由于对称性 等价于前xx个）与xx个填充值（行尾元素）的点乘
            temp += pre_filter[j + mid - m] * temp_res[i * m];
            j_end = m - 1;
        }
        for (int yj = j_sta; yj <= j_end; ++yj) {
            // 第yj个元素离卷积中心j的距离为yj-j 使用的是距离卷积核中心mid距离为yj-j的卷积参数
            temp += filter[mid - myabs(j - yj)] * temp_res[i * m + yj];
        }
        img_dst[pos] = temp;
    }
}

void guassian_smooth(const gray_t* img_src, gray_t** img_dst_ptr, int n, int m, double sigma) {
    gray_t* img_dst = new gray_t[n * m];
    *img_dst_ptr = img_dst;
    // 卷积核：用两次一维卷积分离实现二维卷积 复杂度从 O(m*n*filter_size*filter_size) 降为 O(m*n*filter_size)
    // 1. 根据sigma确定卷积核大小 原理参考https://www.cnblogs.com/shine-lee/p/9671253.html “|1”是为了取邻近的奇数
    int filter_size = int(sigma * 3 * 2 + 1) | 1;
    // 2. 根据高斯分布确定卷积核参数
    int mid = filter_size >> 1;
    gray_t* filter = new gray_t[mid + 1]; // 因为高斯卷积核的对称性 所以只存储前一半加一个参数
    gray_t* pre_filter = new gray_t[mid + 1]; // pre_filter[i]表示sum(filter[0], ..., filter[i])
    double total = 0;
    for (int i = 0; i < mid + 1; ++i) {
        filter[i] = 1 / (sqrt(2 * PI) * sigma) * exp((- (i - mid) * (i - mid)) / (2 * sigma * sigma));
        total += 2 * filter[i];
    }
    total -= filter[mid];
    for (int i = 0; i < mid + 1; ++i) {
        filter[i] /= total;
    }
    pre_filter[0] = filter[0];
    for (int i = 1; i < mid + 1; ++i) {
        pre_filter[i] = filter[i] + pre_filter[i - 1];
    }
    
    // 卷积（卷积核越界部分使用边界填充，保持图片大小不变）
    gray_t* temp_res;  // 存储进行第一维卷积后的结果
    gray_t* device_img_src;
    gray_t* device_pre_filter;
    gray_t* device_filter;
    gray_t* device_img_dst;
    checkCudaErrors(cudaMalloc((void **) &temp_res, sizeof(gray_t) * n * m));
    checkCudaErrors(cudaMalloc((void **) &device_img_src, sizeof(gray_t) * n * m));
    checkCudaErrors(cudaMalloc((void **) &device_pre_filter, sizeof(gray_t) * (mid + 1)));
    checkCudaErrors(cudaMalloc((void **) &device_filter, sizeof(gray_t) * (mid + 1)));
    checkCudaErrors(cudaMalloc((void **) &device_img_dst, sizeof(gray_t) * n * m));
    checkCudaErrors(cudaMemcpy(device_img_src, img_src, sizeof(gray_t) * n * m, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(device_pre_filter, pre_filter, sizeof(gray_t) * (mid + 1), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(device_filter, filter, sizeof(gray_t) * (mid + 1), cudaMemcpyHostToDevice));
    unsigned int grid_rows = diveup(n, 4);
    unsigned int grid_cols = diveup(m, 8);
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(4, 8);
    // 1. 进行第一维卷积
    conv1<<<dimGrid, dimBlock>>>(device_img_src, device_pre_filter, device_filter, temp_res, n, m, mid);
    checkCudaErrors(cudaDeviceSynchronize());
    // 2. 进行第二维卷积
    conv2<<<dimGrid, dimBlock>>>(temp_res, device_pre_filter, device_filter, device_img_dst, n, m, mid);
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaMemcpy(img_dst, device_img_dst, sizeof(gray_t) * n * m, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaFree(temp_res));
    checkCudaErrors(cudaFree(device_img_src));
    checkCudaErrors(cudaFree(device_pre_filter));
    checkCudaErrors(cudaFree(device_filter));
    checkCudaErrors(cudaFree(device_img_dst));
    delete[] filter;
    delete[] pre_filter;
}

void cpu_guassian_smooth(const gray_t* img_src, gray_t** img_dst_ptr, int n, int m, double sigma) {
    gray_t* img_dst = new gray_t[n * m];
    *img_dst_ptr = img_dst;
    // 卷积核：用两次一维卷积分离实现二维卷积 复杂度从 O(m*n*filter_size*filter_size) 降为 O(m*n*filter_size)
    // 1. 根据sigma确定卷积核大小 原理参考https://www.cnblogs.com/shine-lee/p/9671253.html “|1”是为了取邻近的奇数
    int filter_size = int(sigma * 3 * 2 + 1) | 1;
    // 2. 根据高斯分布确定卷积核参数
    int mid = filter_size >> 1;
    gray_t* filter = new gray_t[mid + 1]; // 因为高斯卷积核的对称性 所以只存储前一半加一个参数
    gray_t* pre_filter = new gray_t[mid + 1]; // pre_filter[i]表示sum(filter[0], ..., filter[i])
    double total = 0;
    for (int i = 0; i < mid + 1; ++i) {
        filter[i] = 1 / (sqrt(2 * PI) * sigma) * exp((- (i - mid) * (i - mid)) / (2 * sigma * sigma));
        total += 2 * filter[i];
    }
    total -= filter[mid];
    for (int i = 0; i < mid + 1; ++i) {
        filter[i] /= total;
    }
    pre_filter[0] = filter[0];
    for (int i = 1; i < mid + 1; ++i) {
        pre_filter[i] = filter[i] + pre_filter[i - 1];
    }
    
    // 卷积（卷积核越界部分使用边界填充，保持图片大小不变）
    gray_t* temp_res = new gray_t[n * m];  // 存储进行第一维卷积后的结果
    // 1. 进行第一维卷积
    for (int j = 0; j < m; ++j) {
        for (int i = 0; i < n; ++i) {
            int pos = i * m + j;
            temp_res[pos] = 0;
            int i_sta = i - mid;
            int i_end = i + mid;
            // 当前行的卷积范围：[i-mid, i+mid]
            if (i - mid < 0) {
                // 合并计算[i-mid, 0)部分，即原filter的前mid-i个参数与mid-i个填充值（列首元素）的点乘
                temp_res[pos] += pre_filter[mid - i - 1] * img_src[j];
                i_sta = 0;
            }
            if (i + mid >= n) {
                // 合并计算(n-1, i+mid] 部分，即原filter的后xx=i+mid+1-n个参数（由于对称性 等价于前xx个）与xx个填充值（行尾元素）的点乘
                temp_res[pos] += pre_filter[i + mid - n] * img_src[(n - 1) * m + j];
                i_end = n - 1;
            }
            for (int xi = i_sta; xi <= i_end; ++xi) {
                // 第xi个元素离卷积中心i的距离为xi-i 使用的是距离卷积核中心mid距离为xi-i的卷积参数
                temp_res[pos] += filter[mid - myabs(i - xi)] * img_src[xi * m + j];
            }
        }
    }
    // 2. 进行第二维卷积
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            int pos = i * m + j;
            img_dst[pos] = 0;
            int j_sta = j - mid;
            int j_end = j + mid;
            // 当前行的卷积范围：[j-mid, j+mid]
            if (j - mid < 0) {
                // 合并计算[j-mid, 0)部分，即原filter的前mid-j个参数与mid-j个填充值（行首元素）的点乘
                img_dst[pos] += pre_filter[mid - j - 1] * temp_res[i * m];
                j_sta = 0;
            }
            if (j + mid >= m) {
                // 合并计算(m-1, j+mid] 部分，即原filter的后xx=j+mid+1-m个参数（由于对称性 等价于前xx个）与xx个填充值（行尾元素）的点乘
                img_dst[pos] += pre_filter[j + mid - m] * temp_res[i * m];
                j_end = m - 1;
            }
            for (int yj = j_sta; yj <= j_end; ++yj) {
                // 第yj个元素离卷积中心j的距离为yj-j 使用的是距离卷积核中心mid距离为yj-j的卷积参数
                img_dst[pos] += filter[mid - myabs(j - yj)] * temp_res[i * m + yj];
            }
        }
    }

    delete[] filter;
    delete[] pre_filter;
    delete[] temp_res;
}

double get_time() {
    struct timeval t1;
    gettimeofday(&t1, NULL);
    return t1.tv_sec + t1.tv_usec / 1000000.0;
}

void test_guassian() {
    int n = 1023, m = 888;
    gray_t* src_img = new gray_t[n * m];
    for (int i = 0; i < n * m; ++i) {
        src_img[i] = 0.5 * (i + 1);
    } 
    gray_t sigma = 55.5;
    
    gray_t* cpu_res, *gpu_res;
    double time1 = get_time();
    cpu_guassian_smooth(src_img, &cpu_res, n, m, sigma);
    double time2 = get_time();
    guassian_smooth(src_img, &gpu_res, n, m, sigma);
    double time3 = get_time();
    printf("%lf,%lf\n",time2-time1,time3-time2);
    
    // for (int i = 0; i < n * m; ++i) {
    //     printf("%lf ", gpu_res[i] - cpu_res[i]);
    // } 
}


int main() {
    test_guassian();
}