#ifndef OPENCL_UTILITIES_H
#define OPENCL_UTILITIES_H

#define __NO_STD_VECTOR // Use cl::vector instead of STL version
#define __CL_ENABLE_EXCEPTIONS


#if defined(__APPLE__) || defined(__MACOSX)
    #include <OpenCL/cl.hpp>
#else
    #include <CL/cl.hpp>
#endif


#include <string>
#include <iostream>
#include <fstream>


enum cl_vendor {
    VENDOR_ANY,
    VENDOR_NVIDIA,
    VENDOR_AMD,
    VENDOR_INTEL
};

cl::Context createCLContext(cl_device_type type = CL_DEVICE_TYPE_DEFAULT, cl_vendor vendor = VENDOR_ANY);

cl::Platform getPlatform(cl_device_type, cl_vendor vendor = VENDOR_ANY); 

cl::Program buildProgramFromSource(cl::Context context, std::string filename);

char *getCLErrorString(cl_int err);

#endif
