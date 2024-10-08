from setuptools import setup, Extension
from torch.utils import cpp_extension
import os
module_path = os.path.dirname(__file__)
setup(name='op_cpp',
      ext_modules=[cpp_extension.CUDAExtension(name="fused",
                            sources=["fused_bias_act.cpp", "fused_bias_act_kernel.cu"], include_dirs=cpp_extension.include_paths(),),
                   cpp_extension.CUDAExtension(name="upfirdn2d",
                             sources=["upfirdn2d.cpp", "upfirdn2d_kernel.cu"],include_dirs=cpp_extension.include_paths(),),
                   ],
      cmdclass={'build_ext': cpp_extension.BuildExtension})