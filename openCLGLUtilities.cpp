#include "openCLGLUtilities.hpp"


cl::Context createCLGLContext(cl_device_type type, cl_vendor vendor) {

    cl::Platform platform = getPlatform(type, vendor);

#if defined(__APPLE__) || defined(__MACOSX)
    // Apple (untested)
    cl_context_properties cps[] = {
        CL_CGL_SHAREGROUP_KHR,
        (cl_context_properties)CGLGetShareGroup(CGLGetCurrentContext()),
        CL_CONTEXT_PLATFORM,
        (cl_context_properties)(platform)(),
        0
    };
#else
#ifdef _WIN32
    // Windows
    cl_context_properties cps[] = {
        CL_GL_CONTEXT_KHR,
        (cl_context_properties)wglGetCurrentContext(),
        CL_WGL_HDC_KHR,
        (cl_context_properties)wglGetCurrentDC(),
        CL_CONTEXT_PLATFORM,
        (cl_context_properties)(platform)(),
        0
    };
#else
    // Linux
    cl_context_properties cps[] = {
        CL_GL_CONTEXT_KHR,
        (cl_context_properties)glXGetCurrentContext(),
        CL_GLX_DISPLAY_KHR,
        (cl_context_properties)glXGetCurrentDisplay(),
        CL_CONTEXT_PLATFORM,
        (cl_context_properties)(platform)(),
        0
    };
#endif
#endif

    try {
        cl::Context context = cl::Context(type, cps);

        return context;
    } catch(cl::Error error) {
        throw cl::Error(1, "Failed to create an OpenCL context!");
    }
}
