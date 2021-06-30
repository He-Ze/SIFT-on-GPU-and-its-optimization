echo "================================================编译================================================="
echo "g++ ./src/sift.cpp -fPIC -shared -o ./lib/siftcpu.so "
g++ ./src/sift.cpp -fPIC -shared -o ./lib/siftcpu.so 
echo "g++ ./src/sift_omp.cpp -fPIC -fopenmp -shared -o ./lib/siftomp.so "
g++ ./src/sift_omp.cpp -fPIC -fopenmp -shared -o ./lib/siftomp.so 
echo "nvcc ./src/sift.cu -Xcompiler -fopenmp -Xcompiler -fPIC -shared -o ./lib/siftgpu.so"
nvcc ./src/sift.cu -Xcompiler -fopenmp -Xcompiler -fPIC -shared -o ./lib/siftgpu.so
echo "nvcc ./src/sift_share.cu -Xcompiler -fopenmp -Xcompiler -fPIC -shared -o ./lib/siftshare.so"
nvcc ./src/sift_share.cu -Xcompiler -fopenmp -Xcompiler -fPIC -shared -o ./lib/siftshare.so
echo "nvcc ./src/sift_device.cu -Xcompiler -fopenmp -Xcompiler -fPIC -shared -o ./lib/siftdevice.so"
nvcc ./src/sift_device.cu -Xcompiler -fopenmp -Xcompiler -fPIC -shared -o ./lib/siftdevice.so
echo "nvcc ./src/sift_final.cu -arch=sm_70 -Xcompiler -fopenmp -Xcompiler -fPIC -shared -o ./lib/siftfinal.so"
nvcc ./src/sift_final.cu -arch=sm_70 -Xcompiler -fopenmp -Xcompiler -fPIC -shared -o ./lib/siftfinal.so
echo "================================================运行================================================="
echo "-----------------------------------------------CPU版本-----------------------------------------------"
echo "python ./test/test_cpu.py "
python ./test/test_cpu.py 
echo "---------------------------------------------OpenMP版本----------------------------------------------"
echo "python ./test/test_omp.py"
python ./test/test_omp.py
echo "-----------------------------------------------GPU版本-----------------------------------------------"
echo "CUDA_VISIBLE_DEVICES=6 python ./test/test_gpu.py"
CUDA_VISIBLE_DEVICES=6 python ./test/test_gpu.py
echo "---------------------------------------------共享内存版本---------------------------------------------"
echo "CUDA_VISIBLE_DEVICES=6 python ./test/test_share.py"
CUDA_VISIBLE_DEVICES=6 python ./test/test_share.py
echo "--------------------------------------------减少显存拷贝版本-------------------------------------------"
echo "CUDA_VISIBLE_DEVICES=6 python ./test/test_device.py"
CUDA_VISIBLE_DEVICES=6 python ./test/test_device.py
echo "-----------------------------------------------规约版本-----------------------------------------------"
echo "CUDA_VISIBLE_DEVICES=6 python ./test/test_final.py"
CUDA_VISIBLE_DEVICES=6 python ./test/test_final.py
