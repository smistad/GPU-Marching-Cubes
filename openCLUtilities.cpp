#include "openCLUtilities.hpp"

cl::Platform getPlatform(cl_device_type type, cl_vendor vendor) {
    // Get available platforms
    cl::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);

    if(platforms.size() == 0)
        throw cl::Error(1, "No OpenCL platforms were found");

    int platformID = -1;
    if(vendor != VENDOR_ANY) {
        std::string find;
        switch(vendor) {
            case VENDOR_NVIDIA:
                find = "NVIDIA";
            break;
            case VENDOR_AMD:
                find = "Advanced Micro Devices";
            break;
            case VENDOR_INTEL:
                find = "Intel";
            break;
            default:
                throw cl::Error(1, "Invalid vendor specified");
            break;
        }
        for(unsigned int i = 0; i < platforms.size(); i++) {
            if(platforms[i].getInfo<CL_PLATFORM_VENDOR>().find(find) != std::string::npos) {
                try {
                    cl::vector<cl::Device> devices;
                    platforms[i].getDevices(type, &devices);
                    platformID = i;
                    break;
                } catch(cl::Error e) {
                   continue; 
                }
            }
        }
    } else {
        for(unsigned int i = 0; i < platforms.size(); i++) {
            try {
                cl::vector<cl::Device> devices;
                platforms[i].getDevices(type, &devices);
                platformID = i;
                break;
            } catch(cl::Error e) {
               continue; 
            }
        }
    }

    if(platformID == -1) 
        throw cl::Error(1, "No compatible OpenCL platform found");

    cl::Platform platform = platforms[platformID];
    std::cout << "Using platform vendor: " << platform.getInfo<CL_PLATFORM_VENDOR>() << std::endl;
    return platform;
}

cl::Context createCLContext(cl_device_type type, cl_vendor vendor) {

    cl::Platform platform = getPlatform(type, vendor);

    // Use the preferred platform and create a context
    cl_context_properties cps[] = {
        CL_CONTEXT_PLATFORM, 
        (cl_context_properties)(platform)(),
        0
    };

    try {
        cl::Context context = cl::Context(type, cps);

        return context;
    } catch(cl::Error error) {
        throw cl::Error(1, "Failed to create an OpenCL context!");
    }
}

cl::Program buildProgramFromSource(cl::Context context, std::string filename) {
        // Read source file
        std::ifstream sourceFile(filename.c_str());
        if(sourceFile.fail()) 
            throw cl::Error(1, "Failed to open OpenCL source file");
        std::string sourceCode(
            std::istreambuf_iterator<char>(sourceFile),
            (std::istreambuf_iterator<char>()));
        cl::Program::Sources source(1, std::make_pair(sourceCode.c_str(), sourceCode.length()+1));

        // Make program of the source code in the context
        cl::Program program = cl::Program(context, source);

        cl::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();
    
        // Build program for these specific devices
        try{
            program.build(devices);
        } catch(cl::Error error) {
            if(error.err() == CL_BUILD_PROGRAM_FAILURE) {
                std::cout << "Build log:" << std::endl << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0]) << std::endl;
            }   
            throw error;
        } 
        return program;

}

char *getCLErrorString(cl_int err) {
    switch (err) {
        case CL_SUCCESS:                          return (char *) "Success!";
        case CL_DEVICE_NOT_FOUND:                 return (char *) "Device not found.";
        case CL_DEVICE_NOT_AVAILABLE:             return (char *) "Device not available";
        case CL_COMPILER_NOT_AVAILABLE:           return (char *) "Compiler not available";
        case CL_MEM_OBJECT_ALLOCATION_FAILURE:    return (char *) "Memory object allocation failure";
        case CL_OUT_OF_RESOURCES:                 return (char *) "Out of resources";
        case CL_OUT_OF_HOST_MEMORY:               return (char *) "Out of host memory";
        case CL_PROFILING_INFO_NOT_AVAILABLE:     return (char *) "Profiling information not available";
        case CL_MEM_COPY_OVERLAP:                 return (char *) "Memory copy overlap";
        case CL_IMAGE_FORMAT_MISMATCH:            return (char *) "Image format mismatch";
        case CL_IMAGE_FORMAT_NOT_SUPPORTED:       return (char *) "Image format not supported";
        case CL_BUILD_PROGRAM_FAILURE:            return (char *) "Program build failure";
        case CL_MAP_FAILURE:                      return (char *) "Map failure";
        case CL_INVALID_VALUE:                    return (char *) "Invalid value";
        case CL_INVALID_DEVICE_TYPE:              return (char *) "Invalid device type";
        case CL_INVALID_PLATFORM:                 return (char *) "Invalid platform";
        case CL_INVALID_DEVICE:                   return (char *) "Invalid device";
        case CL_INVALID_CONTEXT:                  return (char *) "Invalid context";
        case CL_INVALID_QUEUE_PROPERTIES:         return (char *) "Invalid queue properties";
        case CL_INVALID_COMMAND_QUEUE:            return (char *) "Invalid command queue";
        case CL_INVALID_HOST_PTR:                 return (char *) "Invalid host pointer";
        case CL_INVALID_MEM_OBJECT:               return (char *) "Invalid memory object";
        case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:  return (char *) "Invalid image format descriptor";
        case CL_INVALID_IMAGE_SIZE:               return (char *) "Invalid image size";
        case CL_INVALID_SAMPLER:                  return (char *) "Invalid sampler";
        case CL_INVALID_BINARY:                   return (char *) "Invalid binary";
        case CL_INVALID_BUILD_OPTIONS:            return (char *) "Invalid build options";
        case CL_INVALID_PROGRAM:                  return (char *) "Invalid program";
        case CL_INVALID_PROGRAM_EXECUTABLE:       return (char *) "Invalid program executable";
        case CL_INVALID_KERNEL_NAME:              return (char *) "Invalid kernel name";
        case CL_INVALID_KERNEL_DEFINITION:        return (char *) "Invalid kernel definition";
        case CL_INVALID_KERNEL:                   return (char *) "Invalid kernel";
        case CL_INVALID_ARG_INDEX:                return (char *) "Invalid argument index";
        case CL_INVALID_ARG_VALUE:                return (char *) "Invalid argument value";
        case CL_INVALID_ARG_SIZE:                 return (char *) "Invalid argument size";
        case CL_INVALID_KERNEL_ARGS:              return (char *) "Invalid kernel arguments";
        case CL_INVALID_WORK_DIMENSION:           return (char *) "Invalid work dimension";
        case CL_INVALID_WORK_GROUP_SIZE:          return (char *) "Invalid work group size";
        case CL_INVALID_WORK_ITEM_SIZE:           return (char *) "Invalid work item size";
        case CL_INVALID_GLOBAL_OFFSET:            return (char *) "Invalid global offset";
        case CL_INVALID_EVENT_WAIT_LIST:          return (char *) "Invalid event wait list";
        case CL_INVALID_EVENT:                    return (char *) "Invalid event";
        case CL_INVALID_OPERATION:                return (char *) "Invalid operation";
        case CL_INVALID_GL_OBJECT:                return (char *) "Invalid OpenGL object";
        case CL_INVALID_BUFFER_SIZE:              return (char *) "Invalid buffer size";
        case CL_INVALID_MIP_LEVEL:                return (char *) "Invalid mip-map level";
        default:                                  return (char *) "Unknown";
    }
}
