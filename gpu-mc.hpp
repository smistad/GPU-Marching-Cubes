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
#include <math.h>

using namespace cl;

typedef unsigned int uint;
typedef unsigned char uchar;

void setupOpenGL(int *, char **, int, int, int, int);
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
