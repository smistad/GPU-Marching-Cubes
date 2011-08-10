#ifndef HPMC_H
#define HPMC_H

#define GL_SHARING_EXTENSION "cl_khr_gl_sharing"
#define __NO_STD_VECTOR // Use cl::vector instead of STL version
#define __CL_ENABLE_EXCEPTIONS

#include <iostream>
#include <fstream>
#include <utility>
#include <string>
#include <GL/glew.h>
#include <CL/cl.hpp>
#include <CL/cl_gl_ext.h>
#include <GL/glut.h>

#if defined __APPLE__ || defined(MACOSX)
#else
    #if defined WIN32
    #else
        #include <GL/glx.h>
    #endif
#endif
#include <math.h>

using namespace cl;

void setupOpenGL(int *, char **);
void setupOpenCL(unsigned char * voxels, int sizeX, int sizeY, int sizeZ);
void run();
void renderScene();
void idle();
void reshape(int width, int height);
void keyboard(unsigned char key, int x, int y);
void mouseMovement(int x, int y);

void updateScalarField();
void histoPyramidConstruction();
void histoPyramidTraversal(int sum);

void parseRawFile(char * filename);


char * getCLErrorString(cl_int error);

typedef unsigned int uint;
typedef unsigned char uchar;

typedef struct {
    int x,y,z;
} Size;

typedef struct {
    float x,y,z;
} Sizef;

#define BUFFER_OFFSET(i) ((char *)NULL + (i))

#endif
