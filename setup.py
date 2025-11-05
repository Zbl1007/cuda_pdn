from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension
import os

setup(
    name="sim_cpp",
    ext_modules=[
        CppExtension(
            "op_sim_cpp",
            ["op_sim.cpp"],
            include_dirs=["/usr/include/eigen3", "/opt/intel/mkl/include"],
            library_dirs=["/opt/intel/mkl/lib/intel64"],
            libraries=["mkl_gf_lp64", "mkl_sequential", "mkl_core", "m", "gfortran"],
            extra_link_args=["-fopenmp"],
        ),
        CppExtension(
            "ac_sim_cpp",
            ["ac_sim.cpp"],
            include_dirs=["/usr/include/eigen3", "/opt/intel/mkl/include"],
            library_dirs=["/opt/intel/mkl/lib/intel64"],
            libraries=["mkl_gf_lp64", "mkl_sequential", "mkl_core", "m", "gfortran"],
            extra_link_args=["-fopenmp"],
        ),
        CUDAExtension(
            'op_sim_cuda_cpp',
            ['op_sim_cuda.cu',],
        # 添加 CUDA 库的路径
        library_dirs=[os.path.join('/usr/local/cuda', 'lib64')],
        # 添加需要链接的库
        libraries=['cudart', 'cusolver', 'cusparse', 'cudss'],
        # 可以添加一些编译器参数
        extra_compile_args={'cxx': ['-O3'], 'nvcc': ['-O3']}
        ),
        CUDAExtension(
            'ac_sim_cuda_cpp',
            ['ac_sim_cuda.cu',],
        # 添加 CUDA 库的路径
        library_dirs=[os.path.join('/usr/local/cuda', 'lib64')],
        # 添加需要链接的库
        libraries=['cudart', 'cusolver', 'cusparse', 'cudss'],
        # 可以添加一些编译器参数
        extra_compile_args={'cxx': ['-O3'], 'nvcc': ['-O3']}
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
