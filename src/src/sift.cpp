/******************************************************************
/*    姓名: 郭晓龙 何泽
/*    文件说明: 该文件实现了 SIFT on CPU
/********************************************************************/
#include "sift.h"

/******************************************************************
/*	函数: sift
/*	函数描述：提取图像的sift特征 将其绘制在res_img上
/*	参数描述：
/*		gray_img：原图像地址，图像为二维数组，每个元素为灰度整数值
/*  	n、m：图像高、宽
/*  	kr：特征描述：样本点所在region的大小为kr x kr
/*  	ks: 特征描述：region划分为ks x ks个subregion
/*  	ko：特征描述：描述每个subregion的方向直方图的bins个数（均匀划分360度）
/*  	S：金字塔：中间层数（每个octave是S+3层）
/*  	sigma_init：金字塔：第一层使用的高斯模糊参数
/*  	contrast_threshold：样本点检测：用于低对比度样本点过滤的阈值大小 如0.03
/*  	edge_response_threshold：样本点检测：用于边缘响应样本点过滤的阈值大小 如10
/*  	max_iterpolation：样本点检测：最大变换插值次数
/*  	time_arr4：记录每个阶段（金字塔构建、样本点检测、主方向赋值、特征生成）的用时，单位为秒
/********************************************************************/
extern "C" void sift(gray_t* gray_img, gray_t* res_img, int n, int m, int kr, int ks, int ko, int S, double sigma_init, double contrast_threshold, double edge_response_threshold, int max_iterpolation, double* time_arr4) {
	// 高斯金字塔
    time_arr4[0] = - get_time();
	// 1. 高斯层
	std::vector<Layer> Gauss_pyramid;
	build_Gauss_pyramid(gray_img, n, m, Gauss_pyramid, S, sigma_init);
    time_arr4[0] += get_time(); 
    printf("Build Pyramid: %.3lf s\n", time_arr4[0]);
	// 极值点检测
    // 2. 高斯差分层
    time_arr4[1] = - get_time();
	std::vector<Layer> DoG_pyramid;
	build_DoG_pyramid(Gauss_pyramid, DoG_pyramid, S);
	std::vector<KeyPoint> keypoints;
	detect_keypoints(DoG_pyramid, keypoints, S, contrast_threshold, edge_response_threshold, max_iterpolation);
    time_arr4[1] += get_time();
    printf("Number of points: %lu \nKeypoints Detect: %.3lf s\n", keypoints.size(), time_arr4[1]);
	// 主方向提取
    time_arr4[2] = - get_time();
	assign_orient(Gauss_pyramid, keypoints, S);
    time_arr4[2] += get_time(); 
    printf("Orientation Assignment: %.3lf s\n", time_arr4[2]);
	// 描述生成
    time_arr4[3] = - get_time();
	generate_features(keypoints, Gauss_pyramid, kr, ks, ko, S);
    time_arr4[3] += get_time();
    printf("Descriptor Generation: %.3lf s\n", time_arr4[3]);
    // 绘制结果
    draw_keypoints(gray_img, res_img, n, m, keypoints);
    // 释放空间
    free_space(Gauss_pyramid);
    free_space(DoG_pyramid);
    //free_space_feat(keypoints);
}

extern "C" void test(gray_t* gray_img, int n, int m, gray_t* res_img, int* res_n, int* res_m, int res_num, int S, double sigma_init) {
	std::vector<Layer> Gauss_pyramid;
	build_Gauss_pyramid(gray_img, n, m, Gauss_pyramid, S, sigma_init);
    std::vector<Layer> DoG_pyramid;
	build_DoG_pyramid(Gauss_pyramid, DoG_pyramid, S);
    int g_size = Gauss_pyramid.size();
    for (int gi = 0; gi < g_size && gi < res_num; ++gi) {
        res_n[gi] = DoG_pyramid[gi].n;
        res_m[gi] = DoG_pyramid[gi].m;
        for (int i = 0; i < res_n[gi] * res_m[gi]; ++i) {
            res_img[i] = DoG_pyramid[gi].img[i];
        }
        res_img += res_n[gi] * res_m[gi];
    }
    free_space(Gauss_pyramid);
    free_space(DoG_pyramid);
}


/******************************************************************
/*	函数: guassian_smooth
/*	函数描述:对图像进行高斯模糊，平滑参数为sigma，结果保存到img_dst
/*	参数描述:
/*		img_src：原图像地址
/*		img_dst_ptr：保存平滑图像的地址的指针
/*  	n、m：图像高、宽（平滑操作不改变图像大小）
/*  	sigma：高斯模糊参数
/********************************************************************/
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

/******************************************************************
/*	函数: double_sample
/*	函数描述:对图像进行2倍上采样（图像的双线性插值）
/*	参数描述:
/*		img_src：原图像地址
/*		img_dst_ptr：保存上采样图像的地址的指针
/*  	n、m：图像高、宽，更新为上采样图像的大小
/********************************************************************/
void double_sample(const gray_t* img_src, gray_t** img_dst_ptr, int* n, int* m) {
	int scale_x = 2, scale_y = 2;
    int nv = *n, mv = *m;
	int nn = scale_x * nv, nm = scale_y * mv;
    gray_t* img_dst = new gray_t[nn * nm];
    *img_dst_ptr = img_dst;
	for (int dst_x = 0; dst_x < nn; ++dst_x) {
		for (int dst_y = 0; dst_y < nm; ++dst_y) {
			// 中心对齐
			double src_x = (dst_x + 0.5) / scale_x - 0.5;
			double src_y = (dst_y + 0.5) / scale_y - 0.5;
			int src_i = int(src_x);
			int src_j = int(src_y);
            // 双线性插值 原理参考https://blog.csdn.net/qq_37577735/article/details/80041586
			img_dst[dst_x * nm + dst_y] = \
				(src_i + 1  - src_x) * (src_j + 1 - src_y) * img_src[src_i * mv + src_j] \
				+ (src_i + 1  - src_x) * (src_y - src_j) * img_src[src_i * mv + src_j + 1] \
				+ (src_x - src_i) * (src_j + 1 - src_y) * img_src[(src_i + 1) * mv + src_j] \
				+ (src_x - src_i) * (src_y - src_j) * img_src[(src_i + 1) * mv + src_j + 1];
		}
	}
	*n = nn;
	*m = nm;
}

/******************************************************************
/*	函数: half_sample
/*	函数描述:对图像进行1/2下采样
/*	参数描述:
/*		img_src：原图像地址
/*		img_dst_ptr：保存下采样图像的地址的指针
/*  	n、m：图像高、宽，更新为下采样图像的大小
/********************************************************************/
void half_sample(const gray_t* img_src, gray_t** img_dst_ptr, int* n, int* m) {
    int nv = *n, mv = *m;
    int nn = nv / 2, nm = mv / 2;
    gray_t* img_dst = new gray_t[nn * nm];
    *img_dst_ptr = img_dst;
    for (int i = 0; i < nn; ++i) {
        for (int j = 0; j < nm; ++j) {
            // SIFT中的1/2下采样方法：每个维度上每隔两个像素取一个像素
            img_dst[i * nm + j] = img_src[(i << 1) * mv + (j << 1)];
        }
    }
    *n = nn;
    *m = nm;
}

/******************************************************************
/*	函数: build_Gauss_pyramid
/*	函数描述: 构建图像的高斯金字塔层
/*	参数描述:
/*		double_sample_img：输入图像
/*		Gauss_pyramid：保存高斯层的vector容器
/*  	n,m：输入图像大小
/*  	S：中间层数（每个octave是S+3层）
/*  	sigma_init：第一层使用的高斯模糊参数
/********************************************************************/
void build_Gauss_pyramid(gray_t* gray_img, int n, int m, std::vector<Layer>& Gauss_pyramid, int S, double sigma_init) {
    // 至少3*3
    double sigma = sigma_init; // 记录下一层的相对于 原始上采样图像 的平滑参数
    double rela_sigma; // 下一层由当前层以rela_sigma的高斯平滑得到 相当于以sigma从 原始上采样图像 的高斯平滑得到
    double s_rt_2 = pow(2, 1.0 / S); // sigma(i+1) = sigma(i) * s_rt_2 注意1/S是整数0 
    double s_mul = sqrt(s_rt_2 * s_rt_2 - 1); // rela_sigma = sqrt((sigma(i) * s_rt_2) ^ 2 - sigma(i) ^ 2) = sigma(i) * s_mul
    gray_t* cur_img;
    for (int octave = 0; n >= 3 && m >= 3; ++octave) {
        if (octave == 0) {
            // 第一个octave的第一层 将输入图像上采样后进行高斯模糊
            double_sample(gray_img, &gray_img, &n, &m);
            guassian_smooth(gray_img, &cur_img, n, m, sigma);
            delete[] gray_img; // 删除上采样图像
            gray_img = nullptr;
            Gauss_pyramid.push_back(Layer(cur_img, n, m, sigma));
        }
        else {
            // 第二个及之后octave的第一层 从上一个octave的第S层下采样得到
            int last_oct_S = octave * (S + 3) - 3;
            half_sample(Gauss_pyramid[last_oct_S].img, &cur_img, &n, &m);
            sigma = Gauss_pyramid[last_oct_S].sigma;
            Gauss_pyramid.push_back(Layer(cur_img, n, m, sigma));
        }
        rela_sigma = sigma * s_mul;
        sigma *= s_rt_2;
        // 每个octave的后续layer：前后依赖实现
        // for (int layer = 1; layer < S + 3; ++layer) {
        //     guassian_smooth(cur_img, &cur_img, n, m, rela_sigma);
        //     Gauss_pyramid.push_back(Layer(cur_img, n, m, sigma));
        //     rela_sigma = sigma * s_mul;
        //     sigma *= s_rt_2;
        // }
        //每个octave的后续layer：前后独立实现 每一层直接从octave的第一层模糊得到
        for (int layer = 1; layer < S + 3; ++layer) {
            Gauss_pyramid.push_back(Layer(nullptr, n, m, sigma));
            sigma *= s_rt_2;
        }
        int oct_layer = octave * (S + 3);
        //#pragma omp parallel for
        for (int layer = 1; layer < S + 3; ++layer) {
            guassian_smooth(Gauss_pyramid[oct_layer].img, &Gauss_pyramid[oct_layer + layer].img, n, m, Gauss_pyramid[oct_layer].sigma * sqrt(pow(2, 2.0 * layer / S) - 1));
        }
    }
}

/******************************************************************
/*	函数: build_DoG_pyramid
/*	函数描述: 构建图像的高斯差分金字塔层
/*	参数描述:
/*		Gauss_pyramid：输入高斯金字塔层
/*		DoG_pyramid：保存高斯差分层的vector容器
/*  	S：中间层数（每个octave是S+3层）
/********************************************************************/
void build_DoG_pyramid(std::vector<Layer>& Gauss_pyramid, std::vector<Layer>& DoG_pyramid, int S) {
    int n_layer = Gauss_pyramid.size();
    for (int layer = 0; layer < n_layer; layer += S + 3) {
        int n = Gauss_pyramid[layer].n, m = Gauss_pyramid[layer].m;
        for (int layeri = layer + 1; layeri < layer + S + 3; ++ layeri) {
            gray_t* dog_img = new gray_t[n * m];
            for (int i = 0; i < n * m; ++i) {
                dog_img[i] = Gauss_pyramid[layeri].img[i] - Gauss_pyramid[layeri - 1].img[i];
            }
            DoG_pyramid.push_back(Layer(dog_img, n, m, Gauss_pyramid[layeri - 1].sigma));
        }
    }
}

/******************************************************************
/*	函数: detect_keypoints
/*	函数描述: 从高斯差分金字塔中检测极值点
/*	参数描述:
/*		DoG_pyramid：输入的高斯差分金字塔
/*		keypoints：保存极值点
/*  	S：中间层数（每个dog octave是S+2层）
/*  	contrast_threshold：用于低对比度样本点过滤的阈值大小 如0.03
/*  	edge_response_threshold：用于边缘响应样本点过滤的阈值大小 如10
/*  	max_iterpolation：最大变换插值次数
/********************************************************************/
void detect_keypoints(std::vector<Layer>& DoG_pyramid, std::vector<KeyPoint>& keypoints, int S, double contrast_threshold, double edge_response_threshold, int max_iterpolation) {
    int n_layer = DoG_pyramid.size();
    int octave = 0;
    for (int layer = 0; layer < n_layer; layer += S + 2) {
        int n = DoG_pyramid[layer].n, m = DoG_pyramid[layer].m;
        for (int layeri = layer + 1; layeri < layer + S + 1; ++ layeri) {
            // 刚好S个中间层
            gray_t* prev_img = DoG_pyramid[layeri - 1].img;
            gray_t* cur_img = DoG_pyramid[layeri].img;
            gray_t* next_img = DoG_pyramid[layeri + 1].img;
            for (int i = 1; i < n-1; ++i) {
                for (int j = 1; j < m-1; ++j) {
                    gray_t cur = cur_img[i * m + j];
                    int x = 0, y, l;
                    // 如果小于它的9+8+9=26个邻居
                    if (cur < cur_img[(i - 1) * m + j - 1] && cur < cur_img[(i - 1) * m + j] && cur < cur_img[(i - 1) * m + j + 1]\
                           && cur < cur_img[i * m + j - 1] && cur < cur_img[i * m + j + 1]\
                           && cur < cur_img[(i + 1) * m + j - 1] && cur < cur_img[(i + 1) * m + j] && cur < cur_img[(i + 1) * m + j + 1]) {
                        if (cur < prev_img[(i - 1) * m + j - 1] && cur < prev_img[(i - 1) * m + j] && cur < prev_img[(i - 1) * m + j + 1]\
                               && cur < prev_img[i * m + j - 1] && cur < prev_img[i * m + j] && cur < prev_img[i * m + j + 1]\
                               && cur < prev_img[(i + 1) * m + j - 1] && cur < prev_img[(i + 1) * m + j] && cur < prev_img[(i + 1) * m + j + 1]) {
                            if (cur < next_img[(i - 1) * m + j - 1] && cur < next_img[(i - 1) * m + j] && cur < next_img[(i - 1) * m + j + 1]\
                               && cur < next_img[i * m + j - 1] && cur < next_img[i * m + j] && cur < next_img[i * m + j + 1]\
                               && cur < next_img[(i + 1) * m + j - 1] && cur < next_img[(i + 1) * m + j] && cur < next_img[(i + 1) * m + j + 1]) {
                                x = i, y = j, l = layeri - layer;
                            }
                        }
                    }
                    // 或者大于它的9+8+9=26个邻居
                    else if (cur > cur_img[(i - 1) * m + j - 1] && cur > cur_img[(i - 1) * m + j] && cur > cur_img[(i - 1) * m + j + 1]\
                           && cur > cur_img[i * m + j - 1] && cur > cur_img[i * m + j + 1]\
                           && cur > cur_img[(i + 1) * m + j - 1] && cur > cur_img[(i + 1) * m + j] && cur > cur_img[(i + 1) * m + j + 1]) {
                        if (cur > prev_img[(i - 1) * m + j - 1] && cur > prev_img[(i - 1) * m + j] && cur > prev_img[(i - 1) * m + j + 1]\
                               && cur > prev_img[i * m + j - 1] && cur > prev_img[i * m + j] && cur > prev_img[i * m + j + 1]\
                               && cur > prev_img[(i + 1) * m + j - 1] && cur > prev_img[(i + 1) * m + j] && cur > prev_img[(i + 1) * m + j + 1]) {
                            if (cur > next_img[(i - 1) * m + j - 1] && cur > next_img[(i - 1) * m + j] && cur > next_img[(i - 1) * m + j + 1]\
                               && cur > next_img[i * m + j - 1] && cur > next_img[i * m + j] && cur > next_img[i * m + j + 1]\
                               && cur > next_img[(i + 1) * m + j - 1] && cur > next_img[(i + 1) * m + j] && cur > next_img[(i + 1) * m + j + 1]) {
                               x = i, y = j, l = layeri - layer;
                            }
                        }
                    }
                    // 添加极值点
                    if (x != 0) {
                        interpolate_keypoints(DoG_pyramid, S, x, y, l, layer, contrast_threshold, edge_response_threshold, max_iterpolation);
                        if (x != 0) {
                            keypoints.push_back(KeyPoint(x, y, n, m, octave, l, DoG_pyramid[layer + l].sigma));
                        }
                    }
                }
            }
        }
        ++octave;
    }
}

/******************************************************************
/*	函数: interpolate_keypoints
/*	函数描述: 对检测到的极值点进行插值更新、过滤
/*	参数描述:
/*		DoG_pyramid：输入的高斯差分金字塔
/*  	S：中间层数（每个dog octave是S+2层）
/*  	x，y，layeri，layer：样本点所在二维坐标、所在DoG层的octave内编号、与其所在octave的第一层在金字塔中的编号
/*  	contrast_threshold：用于低对比度样本点过滤的阈值大小 如0.03
/*  	edge_response_threshold：用于边缘响应样本点过滤的阈值大小 如10
/*  	max_iterpolation：最大变换插值次数
/********************************************************************/
void interpolate_keypoints(std::vector<Layer>& DoG_pyramid, int S, int& x, int& y, int& layeri, int layer, double contrast_threshold, double edge_response_threshold, int max_iterpolation) {
    int n = DoG_pyramid[layer + layeri].n, m = DoG_pyramid[layer + layeri].m;
    double ex_val; //极值
    double ratio; // 边缘响应
    double ratio_threshold = (edge_response_threshold + 1) * (edge_response_threshold + 1) / edge_response_threshold; // 边缘响应阈值
    double fxyl[3]; // 偏移量
    double he[9]; // hessian矩阵
    double he_inv[9]; // hessian矩阵的逆
    double dxyl[3]; // 一阶导
    gray_t* img[3];
    for (int i = 0; i < max_iterpolation; ++i) {
        img[0] = DoG_pyramid[layer + layeri - 1].img;
        img[1] = DoG_pyramid[layer + layeri].img;
        img[2] = DoG_pyramid[layer + layeri + 1].img;
        // 计算二阶导（hessian矩阵） 原理参考https://blog.csdn.net/saltriver/article/details/78990520
        int xy = x * m + y;
        he[0] = img[1][xy - m] + img[1][xy + m] - 2 * img[1][xy]; // Dxx
        he[4] = img[1][xy + 1] + img[1][xy - 1] - 2 * img[1][xy]; // Dyy
        he[8] = img[0][xy] + img[2][xy] - 2 * img[1][xy]; // Dll
        he[1] = he[3] = img[1][xy + m + 1] - img[1][xy + 1] - img[1][xy + m] + img[1][xy]; //Dxy
        he[2] = he[6] = img[2][xy + m] - img[2][xy] - img[1][xy + m] + img[1][xy]; //Dxl
        he[5] = he[7] = img[2][xy + 1] - img[2][xy] - img[1][xy + 1] + img[1][xy]; //Dyl
        // 计算hessian矩阵的逆 公式见https://blog.csdn.net/feixia_24/article/details/41644335
        double det = he[0] * (he[4] * he[8] - he[5] * he[7]) \
                - he[3] * (he[1] * he[8] - he[2] * he[7]) \
                + he[6] * (he[1] * he[5] - he[2] * he[4]);
        // assert det != 0
        he_inv[0] = (he[4] * he[8] - he[5] * he[7]) / det;
        he_inv[1] = (he[2] * he[7] - he[1] * he[8]) / det;
        he_inv[2] = (he[1] * he[5] - he[2] * he[4]) / det;
        he_inv[3] = (he[5] * he[6] - he[3] * he[8]) / det;
        he_inv[4] = (he[0] * he[8] - he[2] * he[6]) / det;
        he_inv[5] = (he[3] * he[2] - he[0] * he[5]) / det;
        he_inv[6] = (he[3] * he[7] - he[4] * he[6]) / det;
        he_inv[7] = (he[1] * he[6] - he[0] * he[7]) / det;
        he_inv[8] = (he[0] * he[4] - he[3] * he[1]) / det;
        // 计算一阶导
        dxyl[0] = img[1][xy + m] - img[1][xy]; // dx
        dxyl[1] = img[1][xy + 1] - img[1][xy]; // dy
        dxyl[2] = img[2][xy] - img[1][xy]; // dl
        // 计算偏移量
        fxyl[0] = - (he_inv[0] * dxyl[0] + he_inv[1] * dxyl[1] + he_inv[2] * dxyl[2]);
        fxyl[1] = - (he_inv[3] * dxyl[0] + he_inv[4] * dxyl[1] + he_inv[5] * dxyl[2]);
        fxyl[2] = - (he_inv[6] * dxyl[0] + he_inv[7] * dxyl[1] + he_inv[8] * dxyl[2]);
        // 计算极值
        ex_val = img[1][xy] + 0.5 * (dxyl[0] * fxyl[0] + dxyl[1] * fxyl[1] + dxyl[2] * fxyl[2]);
        // 计算边缘响应
        ratio = (he[0] + he[4]) * (he[0] + he[4]) / (he[4] * he[0] - he[3] * he[1]);
        // 1. 如果某一维大于0.5 则更新样本点后重新插值
        if (myabs(fxyl[0]) > 0.5 || myabs(fxyl[1]) > 0.5 || myabs(fxyl[2]) > 0.5) {
            x += int(fxyl[0] + 0.5);
            y += int(fxyl[1] + 0.5);
            layeri += int(fxyl[2] + 0.5);
            // 检查是否在合法样本点范围 如果不在 取消该样本点
            if (x < 1 || x > n - 2|| y < 1 || y > m - 2 || layeri < 1 || layeri > S) {
                break;
            }
        }
        // 2. 否则 如果极值小于阈值（低对比度） 取消该样本点
        else if (ex_val < contrast_threshold) {
            break;
        }
        // 3. 否则 如果边缘响应过大 取消该样本点
        else if (ratio > ratio_threshold) {
            break;
        }
        // 4. 否则 保留该样本点
        else {
            return;
        }
    }

    x = 0;
}


/******************************************************************
/*	函数: assign_orient
/*	函数描述: 为样本点赋值主方向
/*	参数描述:
/*		Gauss_pyramid：输入的高斯金字塔
/*  	keypoints：已检测到的样本点
/*  	S：金字塔中每隔octave的中间层数
/********************************************************************/
void assign_orient(std::vector<Layer>& Gauss_pyramid, std::vector<KeyPoint>& keypoints, int S) {
    int old_size = keypoints.size();
    for (int ki = 0; ki < old_size; ++ki) {
        double bins[36] = {0};
        double sigma = 1.5 * keypoints[ki].sigma;
        int layer = keypoints[ki].octave * (S+3) + keypoints[ki].layer;
        gray_t* img = Gauss_pyramid[layer].img;
        int n = Gauss_pyramid[layer].n, m = Gauss_pyramid[layer].m;
        int x = keypoints[ki].x, y = keypoints[ki].y;
        int win_radius = (int(sigma * 3 * 2 + 1) | 1) >> 1; // 高斯权重窗口
        // 统计高斯窗口内的梯度方向分布（将360度划分为36个bins），每个像素的梯度权重为其梯度大小乘以高斯权重大小
        for (int i = - win_radius; i <= win_radius; ++i) {
            for (int j = - win_radius; j <= win_radius; ++j) {
                int xi = i + x, yj = j + y;
                if (xi > 0 && xi < n - 1 && yj > 0 && yj < m - 1) {
                    double dx = img[(xi + 1) * m + yj] - img[(xi - 1) * m + yj];
                    double dy = img[xi * m + yj + 1] - img[xi * m + yj - 1];
                    double magnitude = sqrt(dx * dx + dy * dy);
                    double gaussian = 1/ (2 * PI * sigma * sigma) * exp(- (i * i + j * j) / (2 * sigma * sigma));
                    // dy有可能等于0 需要加上eps数值稳定
                    int theta = (int)((atan(dx / (dy + EPS)) * 180 / PI) + 180);
                    bins[theta /  10] += gaussian * magnitude;
                }
            }
        }
        // 梯度值最大的作为主方向
        double max_theta_val = bins[0];
        int max_theta = 0;
        for (int i = 1; i < 36; ++i) {
            if (bins[i] > max_theta_val) {
                max_theta = i * 10; // 单位：度
                max_theta_val = bins[i];
            }
        }
        keypoints[ki].ori = max_theta;
        // 增强稳定性：大于主方向的梯度值的80%的那些方向也用于创建新的样本点 分别以这些方向为主方向
        for (int i = 1; i < 36; ++i) {
            if (bins[i] > 0.8 * max_theta_val) {
                KeyPoint dup_key(keypoints[ki]);
                dup_key.ori = i * 10;
                keypoints.push_back(dup_key);
            }
        }
    }
}

/******************************************************************
/*	函数: generate_features
/*	函数描述: 生成样本点的特征描述
/*	参数描述:
/*		Gauss_pyramid：高斯金字塔
/*  	keypoints：已检测到的样本点
/*  	kr：样本点所在region的大小为kr x kr
/*  	ks: region划分为ks x ks个subregion
/*  	ko：描述每个subregion的方向直方图的bins个数（均匀划分360度）
/*  	S：金字塔中每隔octave的中间层数
/********************************************************************/
void generate_features(std::vector<KeyPoint>& keypoints, std::vector<Layer>& Gauss_pyramid, int kr, int ks, int ko, int S) {
    int key_size = keypoints.size();
    // assert kr % ks = 0
    int rs = kr / ks; // subregion为rsxrs大小
    int radius =  kr >> 1; // region的半径 即region为(2radius)x(2radius)
    int f_size = ks * ks * ko; // 特征的维度
    double bs = 360.0 / ko;  // 方向直方图每个bin的区间长度
    for (int ki = 0; ki < key_size; ++ki) {
        int layer = keypoints[ki].octave * (S+3) + keypoints[ki].layer;
        gray_t* img = Gauss_pyramid[layer].img;
        int n = keypoints[ki].n, m = keypoints[ki].m;
        int x = keypoints[ki].x, y = keypoints[ki].y;
        int ori = keypoints[ki].ori;
        double sigma = 1.5 * kr;
        double* feature = new double[f_size];
        // 初始化所有直方图
        for (int i = 0; i < f_size; ++i) {
            feature[i] = 0;
        }
        for (int si = 0; si < ks; ++ si) {
            for (int sj = 0; sj < ks; ++sj) {
                // 第sub个（行优先）subregion
                double* sub_feature = feature + (si * ks + sj) * ko; 
                // 左上角坐标为(xsi, ysj)
                int xsi = x - radius + si * rs;
                int ysj = y - radius + sj * rs;
                // 统计该subregion中的梯度方向信息 所有subregion的直方图拼接为feature
                for (int i = xsi; i < xsi + rs; ++i) {
                    for (int j = ysj; j < ysj + rs; ++j) {
                        if (i > 0 && i < n - 1 && j > 0 && j < m - 1) {
                            // 当前像素的图像梯度
                            double dx = img[(i + 1) * m + j] - img[(i - 1) * m + j];
                            double dy = img[i * m + j + 1] - img[i * m + j - 1];
                            double magnitude = sqrt(dx * dx + dy * dy);
                            // 当前像素的高斯权重
                            int di = i - x, dj = j - y;
                            double gaussian = 1/ (2 * PI * sigma * sigma) * exp(- (di * di + dj * dj) / (2 * sigma * sigma));
                            // 当前像素的相对于主方向的方向
                            // dy有可能等于0 需要加上eps数值稳定
                            int theta = atan(dx / (dy + EPS)) * 180 / PI + 180 - ori;
                            int b = int(theta / bs + 0.5);
                            sub_feature[b] += magnitude * gaussian;
                        }
                    }
                }
            }
        }
        // 归一化
        double total = 0;
        for (int i = 0; i < f_size; ++i) {
            total += feature[i];
        }
        for (int i = 0; i < f_size; ++i) {
            feature[i] /= total;
        }
        // 限制最大值为0.2
        for (int i = 0; i < f_size; ++i) {
            feature[i] = feature[i] > 0.2 ? 0.2 : feature[i];
        }
        // 再归一化
        total = 0;
        for (int i = 0; i < f_size; ++i) {
            total += feature[i];
        }
        for (int i = 0; i < f_size; ++i) {
            feature[i] /= total;
        }
        
        keypoints[ki].feature = feature;
        keypoints[ki].feat_len = f_size;
    }
}

/******************************************************************
/*	函数: free_space
/*	函数描述: 释放金字塔空间
/*	参数描述:
/*		pyramid：金字塔
/********************************************************************/
void free_space(std::vector<Layer>& pyramid) {
    int p_size = pyramid.size();
    for (int i = 0; i < p_size; ++i) {
        delete[] pyramid[i].img;
        pyramid[i].img = nullptr;
    }
}

/******************************************************************
/*	函数: free_space_feat
/*	函数描述: 释放样本点特征
/*	参数描述:
/*		keypoints：样本点信息
/********************************************************************/
void free_space_feat(std::vector<KeyPoint>& keypoints) {
    int k_size = keypoints.size();
    for (int i = 0; i < k_size; ++i) {
        delete[] keypoints[i].feature;
        keypoints[i].feature = nullptr;
    }
}

/******************************************************************
/*	函数: draw_keypoints
/*	函数描述: 将样本点绘制在图片上
/*	参数描述:
/*		src_img：原图
/*		res_img：结果图
/*		n，m：图片大小
/*  	keypoints：已检测到的样本点
/********************************************************************/
void draw_keypoints(gray_t* src_img, gray_t* res_img, int n, int m, std::vector<KeyPoint>& keypoints) {
    for (int i = 0; i< n * m; ++i) {
        res_img[i] = src_img[i];
    }
    int key_size = keypoints.size();
    int radius = 2;
    int color = 255;
    for (int ki = 0; ki < key_size; ++ki) {
        int x = n * keypoints[ki].x / keypoints[ki].n;
        int y = m * keypoints[ki].y / keypoints[ki].m;
        // printf("attemp draw (%d, %d) at (%d, %d)\n", x, y, n, m);
        for (int i = x - radius; i <= x + radius; ++i) {
            for (int j = y - radius; j <= y + radius; ++j) {
                if (i >= 0 && i < n && j >= 0 && j < m) {
                    res_img[i * m + j] = color;
                }
            }
        }
    }
}

/******************************************************************
/*	函数: get_time
/*	函数描述: 返回gettimeofday的时间 单位为秒
/*	参数描述:
/********************************************************************/
double get_time() {
    struct timeval t1;
    gettimeofday(&t1, NULL);
    return t1.tv_sec + t1.tv_usec / 1000000.0;
}