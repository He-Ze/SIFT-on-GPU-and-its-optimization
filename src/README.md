## 文件说明

- `src`目录下为源代码文件（每个版本均在前一个版本的基础上完成）：
    - `sift.cpp`：原始CPU实现
    - `sift_omp.cpp`：使用OpenMP优化单个octave
    - `sift.cu`：使用GPU计算卷积
    - `sift_share.cu`：使用了共享内存
    - `sift_device.cu`：显存拷贝优化与下采样的cuda优化
    - `sift_final.cu`：卷积核计算的规约与扫描cuda优化
- `lib`目录下为编译出的`.so`库文件
- `test`目录下为对应名字cuda程序的的测试程序
- `res`目录下为运行之后输出的结果图片
- `run.sh`为编译运行所有程序的shell脚本
- `test.ipynb`文件包含了单元测试的例子（绘制高斯金字塔的前几层、绘制特征点）