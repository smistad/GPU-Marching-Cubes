#ifndef OPENCL_UTILITIES_H
#define OPENCL_UTILITIES_H

#define __NO_STD_VECTOR // Use cl::vector instead of STL version
#define __CL_ENABLE_EXCEPTIONS

#include <CL/cl.hpp>
#include <string>
#include <iostream>
#include <fstream>


enum cl_vendor {
    ALL,
    NVIDIA,
    AMD,
    INTEL
};

cl::Context createCLContext(cl_device_type type, bool GLInterop = false, cl_vendor vendor = ALL);

cl::Program buildProgramFromSource(cl::Context context, std::string filename);

char *getCLErrorString(cl_int err);

#endif
