#include "hpmc.hpp"

int main(int argc, char ** argv) {
    /* Initialize GLUT */
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DEPTH | GLUT_DOUBLE | GLUT_RGBA);
    glutInitWindowPosition(0, 0);
    glutInitWindowSize(glutGet(GLUT_SCREEN_WIDTH),glutGet(GLUT_SCREEN_HEIGHT));
    glutCreateWindow("HistoPyramid Marching Cubes with OpenCL");
    glutFullScreen();	
    glutDisplayFunc(renderScene);
    glutIdleFunc(idle);
    glutReshapeFunc(reshape);
	glutKeyboardFunc(keyboard);
	glutMotionFunc(mouseMovement);
    


    setupOpenGL();
    setupOpenCL();

    glutMainLoop();

    return 0;
}
