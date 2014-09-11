#include "gpu-mc.hpp"

// Define some globals
GLuint VBO_ID = 0;
Program program;
CommandQueue queue;
Context context;
bool writingTo3DTextures;
bool extractSurfaceOnEveryFrame;
bool extractSurface;

int SIZE;
int isolevel = 51;
int windowWidth, windowHeight;
int windowID;

Image3D rawData;
Image3D cubeIndexesImage;
Buffer cubeIndexesBuffer;
Kernel constructHPLevelKernel;
Kernel constructHPLevelCharCharKernel;
Kernel constructHPLevelCharShortKernel;
Kernel constructHPLevelShortShortKernel;
Kernel constructHPLevelShortIntKernel;
Kernel classifyCubesKernel;
Kernel traverseHPKernel;
vector<Image3D> images;
vector<Buffer> buffers;

Sizef scalingFactor;
Sizef translation;

float camX, camY, camZ = 4.0f; //X, Y, and Z
float lastx, lasty, xrot, yrot, xrotrad, yrotrad; //Last pos and rotation
float speed = 0.1f; //Movement speed

inline double log2(double x) {
    return log(x)/log(2.0);
}

void mouseMovement(int x, int y) {
    int cx = windowWidth/2;
    int cy = windowHeight/2;
     
    if(x == cx && y == cy){ //The if cursor is in the middle
        return;
    }
     
    int diffx=x-cx; //check the difference between the current x and the last x position
    int diffy=y-cy; //check the difference between the current y and the last y position
    xrot += (float)diffy/2; //set the xrot to xrot with the addition of the difference in the y position
    yrot += (float)diffx/2;// set the xrot to yrot with the addition of the difference in the x position
    glutWarpPointer(cx, cy); //Bring the cursor to the middle
}

void renderBitmapString(float x, float y, void *font, char *string) {  
    char *c;
    glRasterPos2f(x, y);
    for(c = string; *c != '\0'; c++) {
        glutBitmapCharacter(font, *c);
    }
}

int frame = 0;
int timebase = 0;
char s[100];
int previousTime = 0;
void drawFPSCounter(int sum) {
	frame++;

    int time = glutGet(GLUT_ELAPSED_TIME);
	if (time - timebase > 1000) { // 1 times per second
		sprintf(s,"Triangles: %d FPS: %4.2f. Speed: %d ms. Isovalue: %4.3f", sum, frame*1000.0/(time-timebase), (int)round(time - previousTime), (float)isolevel / 255.0f);
		timebase = time;
		frame = 0;
	}

	previousTime = time;
    glDisable( GL_DEPTH_TEST );
    glDisable(GL_LIGHTING);
    renderBitmapString(-0.99f, 0.95f, GLUT_BITMAP_8_BY_13, s);
    renderBitmapString(-0.99f, 0.9f, GLUT_BITMAP_8_BY_13, "+/-: Change isovalue. W,A,S,D: Move object. Mouse: Rotate object.");
    if(extractSurfaceOnEveryFrame) {
        renderBitmapString(-0.99f, 0.85, GLUT_BITMAP_8_BY_13, "Extracting surfaces on every frame. Press 'e' to change.");
    }else{
        renderBitmapString(-0.99f, 0.85f, GLUT_BITMAP_8_BY_13, "Extracting surfaces only on isovalue change. Press 'e' to change.");
    }
    glEnable(GL_LIGHTING);
    glEnable(GL_DEPTH_TEST);
}

void idle() {
    glutPostRedisplay();
}

void reshape(int width, int height) {
    windowWidth = width;
    windowHeight = height;
	glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glViewport(0, 0, width, height);
	gluPerspective(45.0f, (GLfloat)width/(GLfloat)height, 0.5f, 10.0f);
}
cl::size_t<3> origin; 
cl::size_t<3> region;
int totalSum;
BufferGL *VBOBuffer;
void renderScene() {
    glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);
    if(extractSurfaceOnEveryFrame || extractSurface) {
        // Delete VBO buffer
        glDeleteBuffers(1, &VBO_ID);
        histoPyramidConstruction();

        // Read top of histoPyramid an use this size to allocate VBO below
        int * sum = new int[8];
        if(writingTo3DTextures) {
            queue.enqueueReadImage(images[images.size()-1], CL_FALSE, origin, region, 0, 0, sum);
        } else {
            queue.enqueueReadBuffer(buffers[buffers.size()-1], CL_FALSE, 0, sizeof(int)*8, sum);
        }

        queue.finish();
        totalSum = sum[0] + sum[1] + sum[2] + sum[3] + sum[4] + sum[5] + sum[6] + sum[7] ;
        
        if(totalSum == 0) {
            std::cout << "No triangles were extracted. Check isovalue." << std::endl;
            return;
        }
        
        // Create new VBO
        glGenBuffers(1, &VBO_ID);
        glBindBuffer(GL_ARRAY_BUFFER, VBO_ID);
        glBufferData(GL_ARRAY_BUFFER, totalSum*18*sizeof(cl_float), NULL, GL_STATIC_DRAW);
        glBindBuffer(GL_ARRAY_BUFFER, 0);

        // Traverse the histoPyramid and fill VBO
        histoPyramidTraversal(totalSum);
        queue.flush();
    }

    // Render VBO
    reshape(windowWidth,windowHeight);
    glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
   
	glTranslatef(-camX, -camY, -camZ);

	glRotatef(xrot,1.0,0.0,0.0);
	glRotatef(yrot,0.0, 1.0, 0.0);

    glPushMatrix();
    glColor3f(1.0f, 1.0f, 1.0f);
    glScalef(scalingFactor.x, scalingFactor.y, scalingFactor.z);
    glTranslatef(translation.x, translation.y, translation.z);

    glRotatef(90.0f, 0.0f, 0.0f, 1.0f);
    // Normal Buffer
    glBindBuffer(GL_ARRAY_BUFFER, VBO_ID);
    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_NORMAL_ARRAY);

    glVertexPointer(3, GL_FLOAT, 24, BUFFER_OFFSET(0));
	glNormalPointer(GL_FLOAT, 24, BUFFER_OFFSET(12));    

    if(extractSurfaceOnEveryFrame || extractSurface) 
        queue.finish();
	//glWaitSync(traversalSync, 0, GL_TIMEOUT_IGNORED);
    glDrawArrays(GL_TRIANGLES, 0, totalSum*3);
	
    // Release buffer
    glBindBuffer(GL_ARRAY_BUFFER, 0); 
    glDisableClientState(GL_VERTEX_ARRAY);
    glDisableClientState(GL_NORMAL_ARRAY);

    glPopMatrix();

    // Render text
    glPushMatrix();
    glMatrixMode( GL_PROJECTION );
    glLoadIdentity();
    glMatrixMode( GL_MODELVIEW );
    glLoadIdentity();
    glColor3f(1.0f, 1.0f, 0.0f);
	drawFPSCounter(totalSum);
    glPopMatrix();


    glutSwapBuffers();
    extractSurface = false;
}

void run() {
    glutMainLoop();
}

void setupOpenGL(int * argc, char ** argv, int size, int sizeX, int sizeY, int sizeZ, float spacingX, float spacingY, float spacingZ) {
    /* Initialize GLUT */
    glutInit(argc, argv);
    glutInitDisplayMode(GLUT_DEPTH | GLUT_DOUBLE | GLUT_RGBA);
    glutInitWindowPosition(0, 0);
    //glutInitWindowSize(glutGet(GLUT_SCREEN_WIDTH),glutGet(GLUT_SCREEN_HEIGHT));
    glutInitWindowSize(800, 800);
    windowID = glutCreateWindow("GPU Marching Cubes");
    //glutFullScreen();	
    glutDisplayFunc(renderScene);
    glutIdleFunc(idle);
    glutReshapeFunc(reshape);
	glutKeyboardFunc(keyboard);
	glutMotionFunc(mouseMovement);

    glewInit();
	glEnable(GL_NORMALIZE);
	glEnable(GL_DEPTH_TEST);
	glShadeModel(GL_SMOOTH);
	glEnable(GL_LIGHT0);
	glEnable(GL_LIGHTING);

	// Set material properties which will be assigned by glColor
	GLfloat color[] = { 1.0f, 1.0f, 1.0f, 1.0f };
	glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, color);
    GLfloat specReflection[] = { 0.8f, 0.8f, 0.8f, 1.0f };
    glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, specReflection);
    GLfloat shininess[] = { 16.0f };
    glMaterialfv(GL_FRONT_AND_BACK, GL_SHININESS, shininess);

    // Create light components
    GLfloat ambientLight[] = { 0.3f, 0.3f, 0.3f, 1.0f };
    GLfloat diffuseLight[] = { 0.7f, 0.7f, 0.7f, 1.0f };
    GLfloat specularLight[] = { 1.0f, 1.0f, 1.0f, 1.0f };
    GLfloat position[] = { -0.0f, 4.0f, 1.0f, 1.0f };
     
    // Assign created components to GL_LIGHT0
    glLightfv(GL_LIGHT0, GL_AMBIENT, ambientLight);
    glLightfv(GL_LIGHT0, GL_DIFFUSE, diffuseLight);
    glLightfv(GL_LIGHT0, GL_SPECULAR, specularLight);
    glLightfv(GL_LIGHT0, GL_POSITION, position);

	origin[0] = 0;
	origin[1] = 0;
	origin[2] = 0; 
	region[0] = 2;
	region[1] = 2;
	region[2] = 2;
    scalingFactor.x = spacingX*1.5f/size;
    scalingFactor.y = spacingY*1.5f/size;
    scalingFactor.z = spacingZ*1.5f/size;
    
    translation.x = (float)sizeX/2.0f;
    translation.y = -(float)sizeY/2.0f;
    translation.z = -(float)sizeZ/2.0f;

    extractSurface = true;
    extractSurfaceOnEveryFrame = false;
}

void keyboard(unsigned char key, int x, int y) {
	switch(key) {
		case '+':
			isolevel ++;
            if(!extractSurfaceOnEveryFrame)
                extractSurface = true;
		break;
		case '-':
			isolevel --;
            if(!extractSurfaceOnEveryFrame)
                extractSurface = true;
		break;
        //WASD movement
        case 'w':
            camZ -= 0.1f;
        break;
        case 's':
            camZ += 0.1f;
        break;
        case 'a':
            camX -= 0.1f;
        break;
        case 'd':
            camX += 0.1f;
        break;
        case 'e':
            extractSurfaceOnEveryFrame = !extractSurfaceOnEveryFrame;
        break;
        case 27:
        case 'q':
            //TODO some clean up
            glutDestroyWindow(windowID);
        break;
	}
}

int max(int a, int b) {
    return a > b ? a:b;
}

int prepareDataset(uchar ** voxels, int sizeX, int sizeY, int sizeZ) {
    // If all equal and power of two exit
    if(sizeX == sizeY && sizeY == sizeZ && sizeX == pow(2, log2(sizeX)))
        return sizeX;

    // Find largest size and find closest power of two
    int largestSize = max(sizeX, max(sizeY, sizeZ));
    int size = 0;
    int i = 1;
    while(pow(2, i) < largestSize)
        i++;
    size = pow(2, i);

    // Make new voxel array of this size and fill it with zeros
    uchar * newVoxels = new uchar[size*size*size];
    for(int j = 0; j < size*size*size; j++) 
        newVoxels[j] = 0;

    // Fill the voxel array with previous data
    for(int x = 0; x < sizeX; x++) {
        for(int y = 0; y < sizeY; y++) {
            for(int z = 0; z <sizeZ; z++) {
                newVoxels[x + y*size + z*size*size] = voxels[0][x + y*sizeX + z*sizeX*sizeY];
            }
        }
    }
    delete[] voxels[0];
    voxels[0] = newVoxels;
    return size;
}

#include <sstream>

template <class T>
inline std::string to_string(const T& t) {
    std::stringstream ss;
    ss << t;
    return ss.str();
}

void setupOpenCL(uchar * voxels, int size) {
    SIZE = size; 
   try { 
        // Create a context that use a GPU and OpenGL interop.
		context = createCLGLContext(CL_DEVICE_TYPE_GPU, VENDOR_ANY);

        // Get a list of devices on this platform
		vector<Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();

        // Create a command queue and use the first device
        queue = CommandQueue(context, devices[0]);

        // Check if writing to 3D textures are supported
        std::string sourceFilename;
        if((int)devices[0].getInfo<CL_DEVICE_EXTENSIONS>().find("cl_khr_3d_image_writes") > -1) {
            writingTo3DTextures = true;
            sourceFilename = "gpu-mc.cl";
        } else {
            std::cout << "Writing to 3D textures is not supported on this device. Writing to regular buffers instead." << std::endl;
            std::cout << "Note that this is a bit slower." << std::endl;
            writingTo3DTextures = false;
            sourceFilename = "gpu-mc-morton.cl";
        }

        // Read source file
        std::ifstream sourceFile(sourceFilename.c_str());
        if(sourceFile.fail()) {
            std::cout << "Failed to open OpenCL source file" << std::endl;
            exit(-1);
        }
        std::string sourceCode(
            std::istreambuf_iterator<char>(sourceFile),
            (std::istreambuf_iterator<char>()));
        
        // Insert size
        Program::Sources source(1, std::make_pair(sourceCode.c_str(), sourceCode.length()+1));

        // Make program of the source code in the context
        program = Program(context, source);
    
        // Build program for these specific devices
        try{
            std::string buildOptions = "-DSIZE=" + to_string(SIZE);
            program.build(devices, buildOptions.c_str());
        } catch(Error error) {
            if(error.err() == CL_BUILD_PROGRAM_FAILURE) {
                std::cout << "Build log:\t" << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0]) << std::endl;
            }   
            throw error;
        } 

        if(writingTo3DTextures) {
            // Create images for the HistogramPyramid
            int bufferSize = SIZE;
            // Make the two first buffers use INT8
            images.push_back(Image3D(context, CL_MEM_READ_WRITE, ImageFormat(CL_RGBA, CL_UNSIGNED_INT8), bufferSize, bufferSize, bufferSize));
            bufferSize /= 2;
            images.push_back(Image3D(context, CL_MEM_READ_WRITE, ImageFormat(CL_R, CL_UNSIGNED_INT8), bufferSize, bufferSize, bufferSize));
            bufferSize /= 2;
            // And the third, fourth and fifth INT16
            images.push_back(Image3D(context, CL_MEM_READ_WRITE, ImageFormat(CL_R, CL_UNSIGNED_INT16), bufferSize, bufferSize, bufferSize));
            bufferSize /= 2;
            images.push_back(Image3D(context, CL_MEM_READ_WRITE, ImageFormat(CL_R, CL_UNSIGNED_INT16), bufferSize, bufferSize, bufferSize));
            bufferSize /= 2;
            images.push_back(Image3D(context, CL_MEM_READ_WRITE, ImageFormat(CL_R, CL_UNSIGNED_INT16), bufferSize, bufferSize, bufferSize));
            bufferSize /= 2;
            // The rest will use INT32
            for(int i = 5; i < (log2((float)SIZE)); i ++) {
                if(bufferSize == 1)
                    bufferSize = 2; // Image cant be 1x1x1
                images.push_back(Image3D(context, CL_MEM_READ_WRITE, ImageFormat(CL_R, CL_UNSIGNED_INT32), bufferSize, bufferSize, bufferSize));
                bufferSize /= 2;
            }

            // If writing to 3D textures is not supported we to create buffers to write to 
       } else {
            int bufferSize = SIZE*SIZE*SIZE;
            buffers.push_back(Buffer(context, CL_MEM_READ_WRITE, sizeof(char)*bufferSize));
            bufferSize /= 8;
            buffers.push_back(Buffer(context, CL_MEM_READ_WRITE, sizeof(char)*bufferSize));
            bufferSize /= 8;
            buffers.push_back(Buffer(context, CL_MEM_READ_WRITE, sizeof(short)*bufferSize));
            bufferSize /= 8;
            buffers.push_back(Buffer(context, CL_MEM_READ_WRITE, sizeof(short)*bufferSize));
            bufferSize /= 8;
            buffers.push_back(Buffer(context, CL_MEM_READ_WRITE, sizeof(short)*bufferSize));
            bufferSize /= 8;
            for(int i = 5; i < log2((float)SIZE); i ++) {
                buffers.push_back(Buffer(context, CL_MEM_READ_WRITE, sizeof(int)*bufferSize));
                bufferSize /= 8;
            }

            cubeIndexesBuffer = Buffer(context, CL_MEM_WRITE_ONLY, sizeof(char)*SIZE*SIZE*SIZE);
            cubeIndexesImage = Image3D(context, CL_MEM_READ_ONLY, 
                    ImageFormat(CL_R, CL_UNSIGNED_INT8),
                    SIZE, SIZE, SIZE);
        }

        // Transfer dataset to device
		rawData = Image3D(
                context, 
                CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                ImageFormat(CL_R, CL_UNSIGNED_INT8), 
                SIZE, SIZE, SIZE,
                0, 0, voxels
        );
        delete[] voxels;

		// Make kernels
		constructHPLevelKernel = Kernel(program, "constructHPLevel");
		classifyCubesKernel = Kernel(program, "classifyCubes");
		traverseHPKernel = Kernel(program, "traverseHP");

        if(!writingTo3DTextures) {
            constructHPLevelCharCharKernel = Kernel(program, "constructHPLevelCharChar");
            constructHPLevelCharShortKernel = Kernel(program, "constructHPLevelCharShort");
            constructHPLevelShortShortKernel = Kernel(program, "constructHPLevelShortShort");
            constructHPLevelShortIntKernel = Kernel(program, "constructHPLevelShortInt");
        }

    } catch(Error error) {
       std::cout << error.what() << "(" << error.err() << ")" << std::endl;
       std::cout << getCLErrorString(error.err()) << std::endl;
    }
}


void histoPyramidConstruction() {

    updateScalarField();

    if(writingTo3DTextures) {
        // Run base to first level
		constructHPLevelKernel.setArg(0, images[0]);
		constructHPLevelKernel.setArg(1, images[1]);

        queue.enqueueNDRangeKernel(
			constructHPLevelKernel, 
			NullRange, 
			NDRange(SIZE/2, SIZE/2, SIZE/2), 
			NullRange
		);

        int previous = SIZE / 2;
        // Run level 2 to top level
        for(int i = 1; i < log2((float)SIZE)-1; i++) {
			constructHPLevelKernel.setArg(0, images[i]);
			constructHPLevelKernel.setArg(1, images[i+1]);
			previous /= 2;
            queue.enqueueNDRangeKernel(
				constructHPLevelKernel, 
				NullRange, 
				NDRange(previous, previous, previous), 
                NullRange
			);
        }
    } else {

        // Run base to first level
		constructHPLevelCharCharKernel.setArg(0, buffers[0]);
		constructHPLevelCharCharKernel.setArg(1, buffers[1]);

        queue.enqueueNDRangeKernel(
			constructHPLevelCharCharKernel, 
			NullRange, 
			NDRange(SIZE/2, SIZE/2, SIZE/2), 
			NullRange
		);

        int previous = SIZE / 2;

		constructHPLevelCharShortKernel.setArg(0, buffers[1]);
		constructHPLevelCharShortKernel.setArg(1, buffers[2]);

        queue.enqueueNDRangeKernel(
			constructHPLevelCharShortKernel, 
			NullRange, 
			NDRange(previous/2, previous/2, previous/2), 
			NullRange
		);

        previous /= 2;

		constructHPLevelShortShortKernel.setArg(0, buffers[2]);
		constructHPLevelShortShortKernel.setArg(1, buffers[3]);

        queue.enqueueNDRangeKernel(
			constructHPLevelShortShortKernel, 
			NullRange, 
			NDRange(previous/2, previous/2, previous/2), 
			NullRange
		);

        previous /= 2;

		constructHPLevelShortShortKernel.setArg(0, buffers[3]);
		constructHPLevelShortShortKernel.setArg(1, buffers[4]);

        queue.enqueueNDRangeKernel(
			constructHPLevelShortShortKernel, 
			NullRange, 
			NDRange(previous/2, previous/2, previous/2), 
			NullRange
		);

        previous /= 2;

        constructHPLevelShortIntKernel.setArg(0, buffers[4]);
		constructHPLevelShortIntKernel.setArg(1, buffers[5]);

        queue.enqueueNDRangeKernel(
			constructHPLevelShortIntKernel, 
			NullRange, 
			NDRange(previous/2, previous/2, previous/2), 
			NullRange
		);

        previous /= 2;

        // Run level 2 to top level
        for(int i = 5; i <  log2(SIZE)-1; i++) {
			constructHPLevelKernel.setArg(0, buffers[i]);
			constructHPLevelKernel.setArg(1, buffers[i+1]);
			previous /= 2;
            queue.enqueueNDRangeKernel(
				constructHPLevelKernel, 
				NullRange, 
				NDRange(previous, previous, previous), 
                NullRange
			);
        }
    }
}

void updateScalarField() {
    if(writingTo3DTextures) {
        classifyCubesKernel.setArg(0, images[0]);
        classifyCubesKernel.setArg(1, rawData);
        classifyCubesKernel.setArg(2, isolevel);
        queue.enqueueNDRangeKernel(
                classifyCubesKernel, 
                NullRange, 
                NDRange(SIZE, SIZE, SIZE),
                NullRange
        );
    } else {
        classifyCubesKernel.setArg(0, buffers[0]);
        classifyCubesKernel.setArg(1, cubeIndexesBuffer);
        classifyCubesKernel.setArg(2, rawData);
        classifyCubesKernel.setArg(3, isolevel);
        queue.enqueueNDRangeKernel(
                classifyCubesKernel, 
                NullRange, 
                NDRange(SIZE, SIZE, SIZE),
                NullRange
        );

        cl::size_t<3> offset;
        offset[0] = 0;
        offset[1] = 0;
        offset[2] = 0;
        cl::size_t<3> region;
        region[0] = SIZE;
        region[1] = SIZE;
        region[2] = SIZE;

        // Copy buffer to image
        queue.enqueueCopyBufferToImage(cubeIndexesBuffer, cubeIndexesImage, 0, offset, region);
    }
}

void histoPyramidTraversal(int sum) {
    // Make OpenCL buffer from OpenGL buffer
	unsigned int i = 0;
    if(writingTo3DTextures) {
        for(i = 0; i < images.size(); i++) {
            traverseHPKernel.setArg(i, images[i]);
        }
    } else {
        traverseHPKernel.setArg(0, rawData);
        traverseHPKernel.setArg(1, cubeIndexesImage);
        for(i = 0; i < buffers.size(); i++) {
            traverseHPKernel.setArg(i+2, buffers[i]);
        }
        i += 2;
    }
	
	VBOBuffer = new BufferGL(context, CL_MEM_WRITE_ONLY, VBO_ID);
    traverseHPKernel.setArg(i, *VBOBuffer);
	traverseHPKernel.setArg(i+1, isolevel);
	traverseHPKernel.setArg(i+2, sum);
	//cl_event syncEvent = clCreateEventFromGLsyncKHR((cl_context)context(), (cl_GLsync)glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0), 0);
	//glFinish();
	vector<Memory> v;
	v.push_back(*VBOBuffer);
	//vector<Event> events;
	//Event e;
	//events.push_back(Event(syncEvent));
    queue.enqueueAcquireGLObjects(&v);

	// Increase the global_work_size so that it is divideable by 64
	int global_work_size = sum + 64 - (sum - 64*(sum / 64));
    // Run a NDRange kernel over this buffer which traverses back to the base level
    queue.enqueueNDRangeKernel(traverseHPKernel, NullRange, NDRange(global_work_size), NDRange(64));

	Event traversalEvent;	
    queue.enqueueReleaseGLObjects(&v, 0, &traversalEvent);
//	traversalSync = glCreateSyncFromCLeventARB((cl_context)context(), (cl_event)traversalEvent(), 0); // Need the GL_ARB_cl_event extension
    queue.flush();
}
