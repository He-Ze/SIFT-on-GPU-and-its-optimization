set -v
g++ ./src/sift.cpp -fPIC -shared -o ./lib/siftcpu.so 
nvcc ./src/sift.cu -Xcompiler -fopenmp -Xcompiler -fPIC -shared -o ./lib/siftgpu.so
nvcc ./src/sift_share.cu -Xcompiler -fopenmp -Xcompiler -fPIC -shared -o ./lib/siftshare.so
nvcc ./src/sift_device.cu -Xcompiler -fopenmp -Xcompiler -fPIC -shared -o ./lib/siftdevice.so
nvcc ./src/sift_final.cu -arch=sm_70 -Xcompiler -fopenmp -Xcompiler -fPIC -shared -o ./lib/siftfinal.so
python test.py 
CUDA_VISIBLE_DEVICES=1 python testgpu.py
CUDA_VISIBLE_DEVICES=1 python testshare.py
CUDA_VISIBLE_DEVICES=1 python testdevice.py
CUDA_VISIBLE_DEVICES=1 python testfinal.py
