// Experimental recovery bookkeeping. Dense products stay on the CUDA device.
// The partition itself uses the separately noticed pinned watershed kernel.

__device__ unsigned int recovery_slot(unsigned int x, int slots) {
    x ^= x >> 16; x *= 0x7feb352dU;
    x ^= x >> 15; x *= 0x846ca68bU; x ^= x >> 16;
    return x & (slots - 1);
}

extern "C" __global__ void recovery_validate(const unsigned char* fg,
    const int* markers, const unsigned char* valid, const float* elevation,
    int* table, int* state, int start, int count, int slots, int probes, int cap) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= count) return;
    int i = start+j, label = markers[i];
    bool inside = fg[i] == 255 && valid[i] == 255;
    if (fg[i] != 0 && fg[i] != 255) atomicOr(state, 1);
    if (valid[i] != 0 && valid[i] != 255) atomicOr(state, 2);
    if (label < 0) atomicOr(state, 4);
    if ((__float_as_uint(elevation[i]) & 0x7f800000U) == 0x7f800000U)
        atomicOr(state, 8);
    if (label > 0 && !inside) atomicOr(state, 16);
    if (inside) atomicAdd(state+1, 1);
    if (fg[i] == 255) atomicAdd(state+4, 1);
    if (label <= 0 || !inside) return;
    atomicAdd(state+5, 1);
    unsigned int slot = recovery_slot((unsigned int)label, slots);
    for (int p=0; p<probes; ++p) {
        int old = atomicCAS(table+slot, 0, label);
        if (old == label) return;
        if (old == 0) {
            if (atomicAdd(state+2, 1) >= cap) atomicOr(state, 64);
            return;
        }
        slot = (slot+1) & (slots-1);
    }
    atomicOr(state, 32);
}

extern "C" __global__ void recovery_compact(const int* table, int* ids,
    int* state, int slots, int capacity) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= slots || !table[i]) return;
    int j = atomicAdd(state+3, 1);
    if (j < capacity) ids[j] = table[i];
    else atomicOr(state, 64);
}

extern "C" __global__ void recovery_domain(const unsigned char* fg,
    const unsigned char* valid, unsigned char* domain, int pixels) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < pixels) domain[i] = fg[i] == 255 && valid[i] == 255;
}

extern "C" __global__ void recovery_stats_init(int* stats, int labels,
    int height, int width) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= labels) return;
    int* row = stats+i*11;
    for (int j=0; j<11; ++j) row[j] = 0;
    row[3] = width; row[4] = height; row[5] = -1; row[6] = -1;
}

__device__ int recovery_id_index(const int* ids, int size, int label) {
    int low = 0, high = size;
    while (low < high) {
        int middle = low+(high-low)/2;
        if (ids[middle] < label) low = middle+1;
        else high = middle;
    }
    return low < size && ids[low] == label ? low : -1;
}

extern "C" __global__ void recovery_summarize(const unsigned char* fg,
    const int* markers, const unsigned char* valid, const int* labels,
    const unsigned char* domain, unsigned char* unresolved,
    unsigned char* interfaces, const unsigned char* band, const int* ids,
    int* stats, int* state, int height, int width, int size, int start,
    int count, int publish_products) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= count) return;
    int i = start+j, label = labels[i], row = i/width, col = i%width;
    bool missing = domain[i] && label == 0;
    bool contact = false, invalid = false, next_missing = false;
    int neighbors[4] = {row > 0 ? i-width : -1, col > 0 ? i-1 : -1,
        col+1 < width ? i+1 : -1, row+1 < height ? i+width : -1};
    for (int d=0; d<4; ++d) {
        int n = neighbors[d];
        if (n < 0) continue;
        contact |= label > 0 && labels[n] > 0 && label != labels[n];
        invalid |= fg[n] == 255 && valid[n] == 0;
        next_missing |= domain[n] && labels[n] == 0;
    }
    if (publish_products) {
        unresolved[i] = missing; interfaces[i] = contact;
    }
    if (missing) atomicAdd(state+6, 1);
    if (contact) atomicAdd(state+8, 1);
    if (markers[i] > 0 && markers[i] != label) atomicOr(state, 256);
    if (label < 0 || (label > 0 && !domain[i])) atomicOr(state, 512);
    if (label <= 0) return;
    int k = recovery_id_index(ids, size, label);
    if (k < 0) { atomicOr(state, 128); return; }
    int* facts = stats+k*11;
    if (markers[i] == label) atomicAdd(facts, 1);
    atomicAdd(facts+1, 1);
    if (contact) atomicAdd(facts+2, 1);
    atomicMin(facts+3, col); atomicMin(facts+4, row);
    atomicMax(facts+5, col); atomicMax(facts+6, row);
    if (row == 0 || col == 0 || row+1 == height || col+1 == width)
        atomicOr(facts+7, 1);
    if (invalid) atomicOr(facts+8, 1);
    if (next_missing) atomicOr(facts+9, 1);
    // Base publication may alias band with interfaces, but only reads its own pixel.
    if (publish_products ? contact : band[i] != 0) atomicAdd(facts+10, 1);
}

extern "C" __global__ void recovery_dilate(const unsigned char* previous,
    const unsigned char* domain, unsigned char* next, int height, int width) {
    int i = blockIdx.x * blockDim.x + threadIdx.x, pixels = height*width;
    if (i >= pixels) return;
    int row = i/width, col = i%width;
    bool value = previous[i];
    if (row > 0) value |= previous[i-width] != 0;
    if (col > 0) value |= previous[i-1] != 0;
    if (col+1 < width) value |= previous[i+1] != 0;
    if (row+1 < height) value |= previous[i+width] != 0;
    next[i] = domain[i] && value;
}
