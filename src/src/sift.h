#ifndef SIFT_H
#define SIFT_H

#include <vector>
#include <stdio.h>
#include <math.h>
#include <sys/time.h>
#include "omp.h"

#define gray_t double
#define PI 3.1415926535897
#define EPS (1e-10)
#define myabs(x) ((x) < 0? (-(x)) : (x))

struct Layer {
	gray_t* img;
	int n, m;
    double sigma;
    Layer(gray_t* _img, int _n, int _m, double _sigma): img(_img), n(_n), m(_m), sigma(_sigma) {}
};

struct KeyPoint{
	int x, y; // 二维坐标
    int n, m; // 所在层的图像大小
    int octave; // 所在octave编号
    int layer; // octave内编号
    double sigma; // 所在层的尺度
    double ori; // 主方向
    double* feature; // 特征
    int feat_len; // 特征长度
    KeyPoint(int _x, int _y, int _n, int _m, int _octave, int _layer, double _sigma): x(_x), y(_y), n(_n), m(_m), octave(_octave), layer(_layer), sigma(_sigma) {}
    KeyPoint(const KeyPoint& other): x(other.x), y(other.y), n(other.n), m(other.m), octave(other.octave), layer(other.layer), sigma(other.sigma) {}
};

// 基本图像操作
void guassian_smooth(const gray_t* img_src, gray_t** img_dst_ptr, int n, int m, double sigma);
void double_sample(const gray_t* img_src, gray_t** img_dst_ptr, int* n, int* m) ;
void half_sample(const gray_t* img_src, gray_t** img_dst_ptr, int* n, int* m);
// 构建高斯金字塔
void build_Gauss_pyramid(gray_t* gray_img, int n, int m, std::vector<Layer>& Gauss_pyramid, int S, double sigma_init);
void build_DoG_pyramid(std::vector<Layer>& Gauss_pyramid, std::vector<Layer>& DoG_pyramid, int S);
// 极值点检测
void detect_keypoints(std::vector<Layer>& DoG_pyramid, std::vector<KeyPoint>& keypoints, int S, double contrast_threshold, double edge_response_threshold, int max_iterpolation);
// 极值点筛选
void interpolate_keypoints(std::vector<Layer>& DoG_pyramid, int S, int& x, int& y, int& layeri, int layer, double contrast_threshold, double edge_response_threshold, int max_iterpolation);
// 主方向
void assign_orient(std::vector<Layer>& Gauss_pyramid, std::vector<KeyPoint>& keypoints, int S);
// 描述符
void generate_features(std::vector<KeyPoint>& keypoints, std::vector<Layer>& Gauss_pyramid, int kr, int ks, int ko, int S);
// 其他函数
void draw_keypoints(gray_t* src_img, gray_t* res_img, int n, int m, std::vector<KeyPoint>& keypoints);
void free_space(std::vector<Layer>& pyramid);
void free_space_feat(std::vector<KeyPoint>& keypoints);
double get_time();
// sift
extern "C" void sift(gray_t* gray_img, gray_t* res_img, int n, int m, int kr, int ks, int ko, int S, double sigma_init, double contrast_threshold, double edge_response_threshold, int max_iterpolation, double* time);
// test
extern "C" void test(gray_t* gray_img, int n, int m, gray_t* res_img, int* res_n, int* res_m, int res_num, int S, double sigma_init);
#endif

