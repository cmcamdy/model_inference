# model_inference
Model inference acceleration


## cmake_demo
- 用于提供一个构建cmake工程的模板，用法：
```
cd cmake_demo
mkdir build && cd build
cmake .. && make -j16
./example/add/add 
```
- 成功的话会输出：1+2=3

## kernel 
- 添加了gtest，可以用submodule的方式，也可以用安装直接find的方式
    - git submodule add https://github.com/google/googletest.git kernel/third_party/gtest
- 添加了CUDA组件，待测试（CUDAToolkit需要提前安装，特别注意，如果是WSL需要去官网下载安装包）