#include "gpu-mc.hpp"
#include "rawUtilities.hpp"
#include <iostream>
using namespace std;

int main(int argc, char ** argv) {

    // Process arguments
    if(argc == 5 || argc == 8) {
        char * filename = argv[1];
        int sizeX = atoi(argv[2]);
        int sizeY = atoi(argv[3]);
        int sizeZ = atoi(argv[4]);
        int stepSizeX = 1;
        int stepSizeY = 1;
        int stepSizeZ = 1;
        if(argc == 8) {
            stepSizeX = atoi(argv[5]);
            stepSizeY = atoi(argv[6]);
            stepSizeZ = atoi(argv[7]);
        }
        unsigned char * voxels = readRawFile(filename, sizeX, sizeY, sizeZ, stepSizeX, stepSizeY, stepSizeZ);
        if(voxels == NULL) {
            cout << "File '" << filename << "' not found!" << endl;
            return EXIT_FAILURE;
        }
        int size = prepareDataset(voxels, sizeX/stepSizeX, sizeY/stepSizeY, sizeZ/stepSizeZ);
        setupOpenGL(&argc,argv);
        setupOpenCL(voxels, sizeX/stepSizeX, sizeY/stepSizeY, sizeZ/stepSizeZ);
        run();
    } else {
        cout << "usage: filename.raw sizeX sizeY sizeZ [stepSizeX stepSizeY stepSizeZ]" << endl;
    }


            



    return 0;
}
