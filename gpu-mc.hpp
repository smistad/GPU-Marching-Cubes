#ifndef HPMC_H
#define HPMC_H

#define __NO_STD_VECTOR // Use cl::vector instead of STL version
#define __CL_ENABLE_EXCEPTIONS
#define __USE_GL_INTEROP

#include <iostream>
#include <fstream>
#include <utility>
#include <string>
#include <GL/glew.h>
#include <GL/glut.h>
#include "OpenCLUtilities/openCLGLUtilities.hpp"
#include <math.h>

using namespace cl;

typedef unsigned int uint;
typedef unsigned char uchar;

void setupOpenGL(int * argc, char ** argv, int size, int sizeX, int sizeY, int sizeZ, float spacingX, float spacingY, float spacingZ); 
void setupOpenCL(unsigned char * voxels, int size);
void run();
void renderScene();
void idle();
void reshape(int width, int height);
void keyboard(unsigned char key, int x, int y);
void mouseMovement(int x, int y);

int prepareDataset(uchar ** voxels, int sizeX, int sizeY, int sizeZ);

void updateScalarField();
void histoPyramidConstruction();
void histoPyramidTraversal(int sum);

char * getCLErrorString(cl_int error);


typedef struct {
    int x,y,z;
} Size;

typedef struct {
    float x,y,z;
} Sizef;

#define BUFFER_OFFSET(i) ((char *)NULL + (i))

#endif
