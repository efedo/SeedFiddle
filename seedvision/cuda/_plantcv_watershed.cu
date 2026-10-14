// Pinned scikit-image 0.25.2 watershed/heap behavior, compactness=0, wsl=false.
// Copyright 2003-2009 Massachusetts Institute of Technology; 2009-2011 Broad
// Institute; 2003 Lee Kamentsky; 2003-2005 Peter J. Verveer; scikit-image team.
// BSD-3-Clause: see docs/SCIKIT_IMAGE_WATERSHED_NOTICES.md.

__device__ bool before(float av, int aa, float bv, int ba) {
    return av != bv ? av < bv : aa < ba;
}

__device__ void exchange(float* values, int* ages, int* indices, int a, int b) {
    float value = values[a]; values[a] = values[b]; values[b] = value;
    int age = ages[a]; ages[a] = ages[b]; ages[b] = age;
    int index = indices[a]; indices[a] = indices[b]; indices[b] = index;
}

__device__ void push(float* values, int* ages, int* indices, int* state,
                     int capacity, float value, int age, int index) {
    int child = state[0];
    if (child >= capacity) { state[3] = 1; return; }
    values[child] = value; ages[child] = age; indices[child] = index;
    state[0]++;
    while (child > 0) {
        int parent = (child-1)/2;
        if (!before(values[child], ages[child], values[parent], ages[parent])) break;
        exchange(values, ages, indices, child, parent);
        child = parent;
    }
}

extern "C" __global__ void seedvision_ws_init(const float* elevation, const int* markers,
    const unsigned char* mask, int* output, float* values, int* ages, int* indices,
    int* state, int height, int width, int capacity, int batch) {
    int end = state[2]+batch;
    int pixels = height*width;
    if (end > pixels) end = pixels;
    for (int index = state[2]; index < end; ++index) {
        int label = mask[index] ? markers[index] : 0;
        output[index] = label;
        if (label) push(values, ages, indices, state, capacity, elevation[index], 0, index);
        if (state[3]) break;
    }
    state[2] = end;
}

extern "C" __global__ void seedvision_ws_step(const float* elevation, const int* markers,
    const unsigned char* mask, int* output, float* values, int* ages, int* indices,
    int* state, int height, int width, int capacity, int batch) {
    for (int step = 0; step < batch && state[0] > 0 && !state[3]; ++step) {
        float value = values[0];
        int index = indices[0];
        state[0]--;
        if (state[0] > 0) {
            exchange(values, ages, indices, 0, state[0]);
            int parent = 0;
            while (true) {
                int smallest = parent;
                int left = parent*2+1, right = left+1;
                if (left >= state[0]) break;
                if (before(values[left], ages[left], values[parent], ages[parent])) smallest = left;
                if (right < state[0] && before(values[right], ages[right], values[smallest], ages[smallest])) smallest = right;
                if (smallest == parent) break;
                exchange(values, ages, indices, parent, smallest);
                parent = smallest;
            }
        }
        int row = index/width, column = index%width;
        // The pinned default cross neighborhood is up, left, right, down.
        int neighbors[4] = {row > 0 ? index-width : -1, column > 0 ? index-1 : -1,
            column+1 < width ? index+1 : -1, row+1 < height ? index+width : -1};
        for (int direction = 0; direction < 4; ++direction) {
            int neighbor = neighbors[direction];
            if (neighbor < 0 || !mask[neighbor] || output[neighbor]) continue;
            state[1]++;
            output[neighbor] = output[index];
            float next = elevation[neighbor];
            if (next < value) next = value;
            push(values, ages, indices, state, capacity, next, state[1], neighbor);
            if (state[3]) break;
        }
    }
}
