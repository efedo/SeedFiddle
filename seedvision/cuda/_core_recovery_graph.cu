// Native experimental graph producers; four-connected eroded-core identities.
extern "C" __global__ void recovery_graph_validate(const unsigned char* mask,
    int* state, int start, int count) {
    int j = blockIdx.x*blockDim.x+threadIdx.x;
    if (j >= count) return;
    unsigned char value = mask[start+j];
    if (value != 0 && value != 255) atomicOr(state+5, 1);
    if (value == 255) atomicAdd(state+7, 1);
}

extern "C" __global__ void recovery_graph_cores(const unsigned char* mask,
    int* labels, int* queue, int* ids, int* areas, int* state,
    int height, int width, int capacity, int label_cap, int batch) {
    int pixels = height*width;
    for (int step=0; step<batch && !state[5]; ++step) {
        if (state[1] < state[2]) {
            int i = queue[state[1]++], row = i/width, col = i%width;
            int neighbors[4] = {row > 0 ? i-width : -1, col > 0 ? i-1 : -1,
                col+1 < width ? i+1 : -1, row+1 < height ? i+width : -1};
            for (int d=0; d<4; ++d) {
                int n = neighbors[d];
                if (n < 0 || !mask[n] || labels[n]) continue;
                if (state[2] >= capacity) { state[5] = 4; return; }
                labels[n] = state[3]; queue[state[2]++] = n;
                areas[state[4]-1]++; state[6]++;
            }
        } else {
            if (state[0] >= pixels) return;
            int i = state[0]++;
            if (!mask[i] || labels[i]) continue;
            if (state[4] >= label_cap) { state[5] = 2; return; }
            state[3] = i+1;
            ids[state[4]] = i+1; areas[state[4]] = 1; state[4]++;
            labels[i] = i+1; state[1] = 0; state[2] = 1; queue[0] = i;
            state[6]++;
        }
    }
}

extern "C" __global__ void recovery_graph_validity(unsigned char* valid,
    int height, int width, int x0, int y0, int x1, int y1) {
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i >= height*width) return;
    int row = i/width, col = i%width;
    valid[i] = col >= x0 && col < x1 && row >= y0 && row < y1 ? 255 : 0;
}

extern "C" __global__ void recovery_graph_clip(const int* markers,
    const unsigned char* valid, int* clipped, int pixels) {
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i < pixels) clipped[i] = valid[i] == 255 ? markers[i] : 0;
}
