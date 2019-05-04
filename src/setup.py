from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension

setup(
    name="dtw_cpp",
    ext_modules=[
        CppExtension(
            "dtw_cpp",
            ["dtw.cpp"],
            extra_compile_args=["-fopenmp", "-g", "-O0"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
