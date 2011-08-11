#include "openCLUtilities.hpp"

#if defined __APPLE__ || defined(MACOSX)
#else
    #if defined WIN32
    #else
        #include <GL/glx.h>
    #endif
#endif

cl::Context createCLContext(cl_device_type type, bool GLInterop, cl_vendor vendor) {
    // Get available platforms
    cl::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);

    // TODO: create vendor selecting mechanism

    if(platforms.size() == 0)
        throw cl::Error(1, "No OpenCL platforms were found");

    cl::Platform platform = platforms[0];

    // Use the preferred platform and create a context
    cl_context_properties * cps;
    if(GLInterop) {
        #ifdef __APPLE || defined(MACOSX)
        cps = new cl_context_properties[5];
        //TODO: Apple OpenGL interop
        #else
        cps = new cl_context_properties[7];
        #ifdef WIN32
        cps[0] = CL_GL_CONTEXT_KHR; 
        cps[1] = (cl_context_properties)wglGetCurrentContext();
        cps[2] = CL_WGL_HDC_KHR;
        cps[3] = (cl_context_properties)wglGetCurrentDC();
        #else
        cps[0] = CL_GL_CONTEXT_KHR; 
        cps[1] = (cl_context_properties)glXGetCurrentContext();
        cps[2] = CL_GLX_DISPLAY_KHR;
        cps[3] = (cl_context_properties)glXGetCurrentDisplay();
        #endif
        #endif // end if apple
        cps[4] = CL_CONTEXT_PLATFORM;
        cps[5] = (cl_context_properties)(platform)();
        cps[6] = 0;
    } else {
        cps = new cl_context_properties[3];
        cps[0] = CL_CONTEXT_PLATFORM;
        cps[1] = (cl_context_properties)(platform)();
        cps[2] = 0;
    }
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
