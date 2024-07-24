#include "ofApp.h"
#include <fstream>
#include <sstream>

std::string loadKernel(const std::string& filename) {
    // Try to load from the current working directory
    std::ifstream file(filename);
    
    // If that fails, try to load from the data directory
    if (!file.is_open()) {
        std::string dataPath = ofToDataPath(filename);
        file.open(dataPath);
    }
    
    if (!file.is_open()) {
        throw std::runtime_error("Could not open kernel file: " + filename +
                                 ". Current working directory: " + ofFilePath::getCurrentWorkingDirectory());
    }
    
    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

void ofApp::setup() {
//    ofSetFrameRate(30);

    setupOpenCL();

    // Initialize particles
    particles.resize(numParticles);
    for (auto& p : particles) {
        p.x = ofRandom(ofGetWidth());
        p.y = ofRandom(ofGetHeight());
        p.vx = ofRandom(-1, 1);
        p.vy = ofRandom(-1, 1);
    }

    // Create buffer for particles
    cl_int err;
    particleBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                    sizeof(Particle) * numParticles, particles.data(), &err);
    if (err != CL_SUCCESS) {
        ofLogError() << "Failed to create particle buffer: " << err;
    }
}

void ofApp::setupOpenCL() {
    cl_int err;

    // Get platform
    cl_uint numPlatforms;
    cl_platform_id platform = NULL;
    err = clGetPlatformIDs(1, &platform, &numPlatforms);
    if (err != CL_SUCCESS) {
        ofLogError() << "Failed to get platform ID: " << err;
        return;
    }

    // Get device
    cl_device_id device = NULL;
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    if (err != CL_SUCCESS) {
        ofLogError() << "Failed to get device ID: " << err;
        return;
    }

    // Create context
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    if (err != CL_SUCCESS) {
        ofLogError() << "Failed to create context: " << err;
        return;
    }

    // Create command queue
    queue = clCreateCommandQueue(context, device, 0, &err);
    if (err != CL_SUCCESS) {
        ofLogError() << "Failed to create command queue: " << err;
        return;
    }

    // Load kernel code from file
    std::string kernelCode = loadKernel("moveParticles.cl");
    const char* kernelSource = kernelCode.c_str();
    size_t kernelLength = kernelCode.length();

    // Create program
    program = clCreateProgramWithSource(context, 1, &kernelSource, &kernelLength, &err);
    if (err != CL_SUCCESS) {
        ofLogError() << "Failed to create program: " << err;
        return;
    }

    // Build program
    err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        ofLogError() << "Failed to build program: " << err;
        size_t logSize;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &logSize);
        std::vector<char> buildLog(logSize);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, logSize, buildLog.data(), NULL);
        ofLogError() << "Build log: " << buildLog.data();
        return;
    }

    // Create kernel
    kernel = clCreateKernel(program, "moveParticles", &err);
    if (err != CL_SUCCESS) {
        ofLogError() << "Failed to create kernel: " << err;
        return;
    }
}

void ofApp::moveParticles() {
    cl_int err;

    // Set kernel arguments
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &particleBuffer);
    if (err != CL_SUCCESS) {
        ofLogError() << "Failed to set kernel argument: " << err;
        return;
    }

    // Enqueue kernel execution
    size_t globalWorkSize = numParticles;
    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalWorkSize, NULL, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        ofLogError() << "Failed to enqueue kernel: " << err;
        return;
    }

    // Read the result back to host
    err = clEnqueueReadBuffer(queue, particleBuffer, CL_TRUE, 0,
                              sizeof(Particle) * numParticles, particles.data(), 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        ofLogError() << "Failed to read buffer: " << err;
        return;
    }
}

void ofApp::update() {
    moveParticles();
}

void ofApp::draw() {
    ofBackground(0);
    ofSetColor(255);
    for (auto& p : particles) {
        ofDrawCircle(p.x, p.y, 2);
    }
    ofDrawBitmapString(ofGetFrameRate(), 10, 10);
}
