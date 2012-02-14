#include "rawUtilities.hpp"

unsigned char * readRawFile(char * filename, int sizeX, int sizeY, int sizeZ, int stepSizeX, int stepSizeY, int stepSizeZ) {
    // Parse the specified raw file
    int rawDataSize = sizeX*sizeY*sizeZ;
    unsigned char * rawVoxels = new unsigned char[rawDataSize];
    FILE * file = fopen(filename, "rb");
    if(file == NULL)
        return NULL;

    fread(rawVoxels, sizeof(unsigned char), rawDataSize, file);
	if(stepSizeX == 1 && stepSizeY == 1 && stepSizeZ == 1) 
        return rawVoxels;

    unsigned char * voxels = new unsigned char[rawDataSize / ( stepSizeX*stepSizeY*stepSizeZ)];
    int i = 0;
    for(int z = 0; z < sizeZ; z += stepSizeZ) {
        for(int y = 0; y < sizeY; y += stepSizeY) {
            for(int x = 0; x < sizeX; x += stepSizeX) {
                voxels[i] = rawVoxels[x + y*sizeX + z*sizeX*sizeY];
                i++;
            }
        }
    }
    delete rawVoxels;
    return voxels;
}
