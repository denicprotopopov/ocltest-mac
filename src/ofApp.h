#pragma once

#include "ofMain.h"
#include <OpenCL/opencl.h>

struct Particle {
    float x, y;
    float vx, vy;
};

class ofApp : public ofBaseApp {
public:
    void setup();
    void update();
    void draw();

    void setupOpenCL();
    void moveParticles();

    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel;
    std::vector<Particle> particles;
    cl_mem particleBuffer;
    const int numParticles = 1000;
};
