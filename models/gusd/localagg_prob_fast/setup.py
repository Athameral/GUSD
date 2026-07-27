#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os
os.path.dirname(os.path.abspath(__file__))

num_class = 17

setup(
    name="local_aggregate_prob_fast",
    packages=['local_aggregate_prob_fast'],
    ext_modules=[
        CUDAExtension(
            name="local_aggregate_prob_fast._C",
            sources=[
            "src/aggregator_impl.cu",
            "src/forward.cu",
            "src/backward.cu",
            "local_aggregate.cu",
            "ext.cpp"],
            # extra_compile_args={"nvcc": ["-I" + os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party/glm/")]})
            # extra_compile_args={"nvcc": ["-g", "-G", "-Xcompiler", "-fno-gnu-unique","-I" + os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party/glm/")]})
            # extra_compile_args={"nvcc": ["-Xcompiler", "-fno-gnu-unique","-I" + os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party/glm/")]})
            extra_compile_args={"nvcc": ["-Xcompiler", "-fno-gnu-unique",
                    "-gencode=arch=compute_60,code=sm_60",
                    "-gencode=arch=compute_61,code=sm_61",
                    "-gencode=arch=compute_70,code=sm_70",
                    "-gencode=arch=compute_75,code=sm_75",
                    "-gencode=arch=compute_80,code=sm_80",
                    "-gencode=arch=compute_86,code=sm_86",
                    "-gencode=arch=compute_86,code=compute_86",
                    ]},
            define_macros=[
                ('NUM_LOGITS_CHANNELS', str(num_class))
            ],
            ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
