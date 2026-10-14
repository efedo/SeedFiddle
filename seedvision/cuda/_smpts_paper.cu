// Declared SMPTS paper reconstruction; these are not original capsule sources.
extern "C" __global__ void paper_initialize(const unsigned char* mask,
    int* parent, int* state, int start, int count) {
    int j = blockIdx.x*blockDim.x+threadIdx.x;
    if (j >= count) return;
    int i = start+j;
    unsigned char v = mask[i];
    if (v != 0 && v != 255) atomicOr(state, 1);
    parent[i] = v == 255 ? i+1 : 0;
}

extern "C" __global__ void paper_reset(int* state) {
    state[0] = 0;
}

extern "C" __global__ void paper_jump(const int* parent, int* next,
    int start, int count) {
    int j = blockIdx.x*blockDim.x+threadIdx.x;
    if (j >= count) return;
    int i = start+j, p = parent[i];
    next[i] = p ? parent[p-1] : 0;
}

extern "C" __global__ void paper_hook(int* parent, int* state,
    int height, int width, int start, int count) {
    int j = blockIdx.x*blockDim.x+threadIdx.x;
    if (j >= count) return;
    int i = start+j, y = i/width, x = i%width;
    int neighbors[4] = {x ? i-1 : -1,
        y && x ? i-width-1 : -1, y ? i-width : -1,
        y && x+1 < width ? i-width+1 : -1};
    for (int d=0; d<4; ++d) {
        int n = neighbors[d];
        if (n < 0) continue;
        // Every concurrent access to the hooked array is atomic.
        int a = atomicAdd(parent+i, 0), b = atomicAdd(parent+n, 0);
        if (!a || !b || a == b) continue;
        int high = a > b ? a : b, low = a < b ? a : b;
        int previous = atomicMin(parent+high-1, low);
        if (previous > low) atomicExch(state, 1);
    }
}

extern "C" __global__ void paper_verify_roots(const int* parent,
    unsigned char* flags, int* state, int height, int width, int start, int count) {
    int j = blockIdx.x*blockDim.x+threadIdx.x;
    if (j >= count) return;
    int i = start+j, p = parent[i], y = i/width, x = i%width;
    flags[i] = p == i+1;
    if (!p) return;
    if (p < 1 || p > i+1 || parent[p-1] != p) { atomicOr(state, 1); return; }
    int neighbors[4] = {x ? i-1 : -1,
        y && x ? i-width-1 : -1, y ? i-width : -1,
        y && x+1 < width ? i-width+1 : -1};
    for (int d=0; d<4; ++d) {
        int n = neighbors[d];
        if (n >= 0 && parent[n] && parent[n] != p) atomicOr(state, 1);
    }
}

extern "C" __global__ void paper_compact(const int* parent, const int* prefix,
    int* labels, int start, int count) {
    int j = blockIdx.x*blockDim.x+threadIdx.x;
    if (j < count) { int i = start+j; labels[i] = parent[i] ? prefix[parent[i]-1] : 0; }
}

extern "C" __global__ void paper_stats_initialize(int* stats, int slots,
    int height, int width) {
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i >= slots) return;
    stats[5*i] = 0; stats[5*i+1] = width; stats[5*i+2] = height;
    stats[5*i+3] = 0; stats[5*i+4] = 0;
}

extern "C" __global__ void paper_stats(const int* labels, int* stats, int* state,
    int slots, int width, int start, int count) {
    int j = blockIdx.x*blockDim.x+threadIdx.x;
    if (j >= count) return;
    int i = start+j, id = labels[i];
    if (id < 0 || id >= slots) { atomicOr(state, 1); return; }
    if (!id) return;
    int x = i%width, y = i/width;
    atomicAdd(stats+5*id, 1);
    atomicMin(stats+5*id+1, x); atomicMin(stats+5*id+2, y);
    atomicMax(stats+5*id+3, x+1); atomicMax(stats+5*id+4, y+1);
}

extern "C" __global__ void paper_median(const unsigned char* mask,
    unsigned char* result, int* state, int height, int width, int size,
    int start, int count) {
    int j = blockIdx.x*blockDim.x+threadIdx.x;
    if (j >= count) return;
    int i = start+j, y = i/width, x = i%width, r = size/2, sum = 0;
    for (int dy=-r; dy<=r; ++dy) for (int dx=-r; dx<=r; ++dx) {
        int yy = y+dy, xx = x+dx;
        yy = yy < 0 ? 0 : (yy >= height ? height-1 : yy);
        xx = xx < 0 ? 0 : (xx >= width ? width-1 : xx);
        unsigned char v = mask[yy*width+xx];
        if (v != 0 && v != 255) atomicOr(state, 1);
        sum += v == 255;
    }
    result[i] = sum > size*size/2 ? 255 : 0;
}

extern "C" __global__ void paper_intersection(const unsigned char* a,
    const unsigned char* b, unsigned char* result, int* state, int start, int count) {
    int j = blockIdx.x*blockDim.x+threadIdx.x;
    if (j >= count) return;
    int i = start+j;
    if ((a[i] != 0 && a[i] != 255) || (b[i] != 0 && b[i] != 255)) atomicOr(state, 1);
    result[i] = a[i] == 255 && b[i] == 255 ? 255 : 0;
}

extern "C" __global__ void paper_select(const unsigned char* mask,
    const int* labels, const unsigned char* selected, unsigned char* target,
    int* state, int slots, int start, int count) {
    int j = blockIdx.x*blockDim.x+threadIdx.x;
    if (j >= count) return;
    int i = start+j, id = labels[i];
    if (id < 0 || id >= slots || (mask[i] != 0 && mask[i] != 255)) {
        atomicOr(state, 1); target[i] = 0; return;
    }
    if ((mask[i] == 255) != (id > 0)) atomicOr(state, 1);
    target[i] = selected[id] && mask[i] == 255 ? 255 : 0;
}

extern "C" __global__ void paper_erode_axis(const unsigned char* mask,
    unsigned char* result, int height, int width, int size, int horizontal,
    int start, int count) {
    int j = blockIdx.x*blockDim.x+threadIdx.x;
    if (j >= count) return;
    int i = start+j, y = i/width, x = i%width, r = size/2;
    unsigned char v = 255;
    for (int d=-r; d<=r; ++d) {
        int yy = y+(horizontal ? 0 : d), xx = x+(horizontal ? d : 0);
        yy = yy < 0 ? 0 : (yy >= height ? height-1 : yy);
        xx = xx < 0 ? 0 : (xx >= width ? width-1 : xx);
        unsigned char sample = mask[yy*width+xx];
        v = sample < v ? sample : v;
    }
    result[i] = v;
}

extern "C" __global__ void paper_merge(const unsigned char* original,
    const int* labels, const unsigned char* selected, const unsigned char* eroded,
    unsigned char* result, int start, int count) {
    int j = blockIdx.x*blockDim.x+threadIdx.x;
    if (j < count) { int i = start+j; result[i] = selected[labels[i]] ? eroded[i] : original[i]; }
}

extern "C" __global__ void paper_parent_initialize(int* minimum, int* maximum,
    int slots, int source_slots) {
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i < slots) { minimum[i] = source_slots; maximum[i] = 0; }
}

extern "C" __global__ void paper_parents(const int* source, const int* final,
    int* minimum, int* maximum, int* state, int slots, int source_slots,
    int start, int count) {
    int j = blockIdx.x*blockDim.x+threadIdx.x;
    if (j >= count) return;
    int i = start+j, id = final[i], parent = source[i];
    if (id < 0 || id >= slots || parent < 0 || parent >= source_slots) {
        atomicOr(state, 1); return;
    }
    if (!id) return;
    if (!parent) { atomicOr(state, 1); return; }
    atomicMin(minimum+id, parent); atomicMax(maximum+id, parent);
}

extern "C" __global__ void paper_crop(const unsigned char* rgb, const int* labels,
    unsigned char* packed_rgb, unsigned char* packed_mask, int width, int label,
    int x0, int y0, int crop_width, int offset, int start, int count) {
    int j = blockIdx.x*blockDim.x+threadIdx.x;
    if (j >= count) return;
    int k = start+j, source = (y0+k/crop_width)*width+x0+k%crop_width;
    bool keep = labels[source] == label;
    int out = offset+k;
    packed_mask[out] = keep ? 255 : 0;
    for (int c=0; c<3; ++c) packed_rgb[3*out+c] = keep ? rgb[3*source+c] : 0;
}
