struct Particle {
    float x, y;
    float vx, vy;
};

kernel void moveParticles(global struct Particle* particles) {
    int id = get_global_id(0);

    particles[id].x += particles[id].vx;
    particles[id].y += particles[id].vy;

    //if (particles[id].x < 0 || particles[id].x >= 1024) particles[id].vx *= -1;
    //if (particles[id].y < 0 || particles[id].y >= 768) particles[id].vy *= -1;

    // screen-wrap
    if (particles[id].x < 0)    particles[id].x = 1024;
    if (particles[id].x > 1024) particles[id].x = 0;
    if (particles[id].y < 0)    particles[id].y = 768;
    if (particles[id].y > 768) particles[id].y = 0;

    //if (particles[id].y < 0 || particles[id].y >= 768) particles[id].vy *= -1;

    //for (int i = 1; i <= sizeof(particles); i++) {
    //    if (particles[id].x - particles[i].x < 1.0 &&
    //        particles[id].y - particles[i].y < 1.0) {
    //        particles[id].vx *= -1;
    //        particles[id].vy *= -1;
    //    }
    //}
}
