About
========================================
A GPU implementation of the Marching Cubes algorithm for extracting surfaces from volumes using OpenCL and OpenGL.
See http://www.thebigblob.com/marching-cubes-implementation-using-opencl-and-opengl/ for detailed information about this implementation.

Dependencies
========================================
OpenGL, GLUT, GLU, GLEW and OpenCL (with C++ bindings)

Notes about implementation
========================================
* Example 3D RAW files can be downloaded from www.volvis.org
* Currently only supports 8 bit raw files, but should be easily extended to other types
* Due to the lack of 3D texture write support on NVIDIA GPUs a slower version is used on NVIDIA GPUs. This version uses regular buffers and morton codes to emulate 3D caching. (not optimal, but at least it works)
* See LICENCE file for license information
* If you clone the project, remember to run git submodule init and git submodule update to fetch the contents of the OpenCLUtilities submodule

Compiling
========================================
Use the attached CMakeLists.txt to compile the program:
cmake CMakeLists.txt

Usage
========================================
Run the program with the following arguments:

filename.raw sizeX sizeY sizeZ [stepSizeX stepSizeY stepSizeZ] [spacingX spacingY spacingZ]
