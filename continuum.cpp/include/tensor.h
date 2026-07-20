/*
 * tensor.h — Minimal tensor library for Continuum SLM C++ inference.
 *
 * Design goals:
 * - Zero heap allocation during inference (arena-based)
 * - Row-major float32 storage
 * - 4D tensor shape (batch, rows, cols, channels) matching GGML convention
 * - SIMD-friendly contiguous memory
 */

#ifndef CONTINUUM_TENSOR_H
#define CONTINUUM_TENSOR_H

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <algorithm>

namespace continuum {

// ============================================================================
// FP16 Half-Precision Utilities (Phase 8)
// ============================================================================

// IEEE 754 half-precision ↔ single-precision conversion
float fp16_to_fp32(uint16_t h);
uint16_t fp32_to_fp16(float f);

// HalfStorage: FP16 weight container for reduced memory bandwidth
// Weights are stored as uint16_t (half the size of float).
// Dequantization creates an FP32 copy in the arena for one forward pass.
struct HalfStorage {
    uint16_t* data = nullptr;
    size_t count = 0;

    ~HalfStorage() { if (data) free(data); }
    HalfStorage() = default;
    HalfStorage(const HalfStorage&) = delete;
    HalfStorage& operator=(const HalfStorage&) = delete;
    HalfStorage(HalfStorage&& o) noexcept : data(o.data), count(o.count) { o.data = nullptr; o.count = 0; }
    HalfStorage& operator=(HalfStorage&& o) noexcept {
        if (this != &o) { if (data) free(data); data = o.data; count = o.count; o.data = nullptr; o.count = 0; }
        return *this;
    }

    // Create from FP32 data
    static HalfStorage from_fp32(const float* fp32_data, size_t n);

    // Dequantize into FP32 buffer (must be pre-allocated with count floats)
    void dequantize(float* fp32_out) const;

    bool empty() const { return data == nullptr || count == 0; }
};

// ============================================================================
// Tensor shape: up to 4 dimensions
// ============================================================================
struct TensorShape {
    int32_t ne[4] = {1, 1, 1, 1};  // ne[0]=cols, ne[1]=rows, ne[2]=channels, ne[3]=batch
    int32_t nb[4] = {0, 0, 0, 0};  // strides in bytes

    TensorShape() = default;
    TensorShape(int32_t d0) : ne{d0, 1, 1, 1} { calc_strides(); }
    TensorShape(int32_t d0, int32_t d1) : ne{d0, d1, 1, 1} { calc_strides(); }
    TensorShape(int32_t d0, int32_t d1, int32_t d2) : ne{d0, d1, d2, 1} { calc_strides(); }
    TensorShape(int32_t d0, int32_t d1, int32_t d2, int32_t d3) : ne{d0, d1, d2, d3} { calc_strides(); }

    void calc_strides() {
        nb[0] = sizeof(float);       // col stride = 4 bytes
        nb[1] = nb[0] * ne[0];       // row stride
        nb[2] = nb[1] * ne[1];       // channel stride
        nb[3] = nb[2] * ne[2];       // batch stride
    }

    int32_t n_dims() const {
        if (ne[3] > 1) return 4;
        if (ne[2] > 1) return 3;
        if (ne[1] > 1) return 2;
        return 1;
    }

    size_t n_elements() const {
        return (size_t)ne[0] * ne[1] * ne[2] * ne[3];
    }

    size_t n_bytes() const {
        return n_elements() * sizeof(float);
    }
};

// ============================================================================
// Tensor: float32 multi-dimensional array
// ============================================================================
struct Tensor {
    float* data = nullptr;
    TensorShape shape;
    bool owns_data = false;

    Tensor() = default;
    ~Tensor() { if (owns_data && data) free(data); }

    // Disable copy, enable move
    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;
    Tensor(Tensor&& other) noexcept
        : data(other.data), shape(other.shape), owns_data(other.owns_data) {
        other.data = nullptr;
        other.owns_data = false;
    }
    Tensor& operator=(Tensor&& other) noexcept {
        if (this != &other) {
            if (owns_data && data) free(data);
            data = other.data;
            shape = other.shape;
            owns_data = other.owns_data;
            other.data = nullptr;
            other.owns_data = false;
        }
        return *this;
    }

    // Create with owned memory
    static Tensor create(const TensorShape& s) {
        Tensor t;
        t.shape = s;
        t.data = (float*)aligned_alloc(64, s.n_bytes());
        t.owns_data = true;
        memset(t.data, 0, s.n_bytes());
        return t;
    }

    // View into existing memory (no ownership)
    static Tensor view(float* ptr, const TensorShape& s) {
        Tensor t;
        t.data = ptr;
        t.shape = s;
        t.owns_data = false;
        return t;
    }

    // Accessors
    inline float& at(int32_t i0) const {
        return data[i0];
    }
    inline float& at(int32_t i0, int32_t i1) const {
        return data[i1 * shape.ne[0] + i0];
    }
    inline float& at(int32_t i0, int32_t i1, int32_t i2) const {
        return data[(i2 * shape.ne[1] + i1) * shape.ne[0] + i0];
    }
    inline float& at(int32_t i0, int32_t i1, int32_t i2, int32_t i3) const {
        return data[((i3 * shape.ne[2] + i2) * shape.ne[1] + i1) * shape.ne[0] + i0];
    }

    size_t n_elements() const { return shape.n_elements(); }
    size_t n_bytes() const { return shape.n_bytes(); }
    int32_t n_dims() const { return shape.n_dims(); }

    // Fill with constant value
    void fill(float val) {
        size_t n = n_elements();
        for (size_t i = 0; i < n; i++) data[i] = val;
    }

    // Copy from another tensor (must have same shape)
    void copy_from(const Tensor& src) {
        memcpy(data, src.data, n_bytes());
    }
};

// ============================================================================
// Memory Arena: bump allocator for inference-time tensors
// ============================================================================
class Arena {
    float* base;
    size_t size;
    size_t offset;

public:
    Arena(size_t total_bytes) {
        size = total_bytes;
        base = (float*)aligned_alloc(64, total_bytes);
        offset = 0;
    }

    ~Arena() { if (base) free(base); }

    float* alloc(size_t n_floats) {
        size_t bytes = n_floats * sizeof(float);
        bytes = (bytes + 63) & ~63;  // 64-byte align
        if (offset + bytes > size) return nullptr;
        float* ptr = base + (offset / sizeof(float));
        offset += bytes;
        return ptr;
    }

    Tensor alloc_tensor(const TensorShape& shape) {
        float* ptr = alloc(shape.n_elements());
        return Tensor::view(ptr, shape);
    }

    void reset() { offset = 0; memset(base, 0, size); }
    size_t used() const { return offset; }
    size_t capacity() const { return size; }
};

// ============================================================================
// Basic tensor operations (in-place and creating new)
// ============================================================================

// Element-wise add: dst = a + b
void tensor_add(Tensor& dst, const Tensor& a, const Tensor& b);

// Element-wise multiply: dst = a * b
void tensor_mul(Tensor& dst, const Tensor& a, const Tensor& b);

// dst = a + b * c (fused multiply-add)
void tensor_fma(Tensor& dst, const Tensor& a, const Tensor& b, const Tensor& c);

// Matrix multiply: C[M,N] = A[M,K] * B[K,N]
void tensor_matmul(Tensor& dst, const Tensor& a, const Tensor& b);

// Batch matmul: C[B,M,N] = A[B,M,K] * B[B,K,N]
void tensor_batch_matmul(Tensor& dst, const Tensor& a, const Tensor& b);

// Outer product: C[B,K,K] = A[B,K,1] * B[B,1,K]
void tensor_outer_product(Tensor& dst, const Tensor& a, const Tensor& b);

// RMS Normalization along last dimension
void tensor_rms_norm(Tensor& dst, const Tensor& x, const Tensor& scale, float eps);

// Softmax along last dimension (in-place on logits)
void tensor_softmax_inplace(Tensor& x);

// Sigmoid activation (in-place)
void tensor_sigmoid_inplace(Tensor& x);

// SiLU activation (in-place)
void tensor_silu_inplace(Tensor& x);

// GELU activation (in-place)
void tensor_gelu_inplace(Tensor& x);

// Transpose last two dims: dst[...,j,i] = src[...,i,j]
void tensor_transpose_last2(Tensor& dst, const Tensor& src);

// Reshape (view into same data)
Tensor tensor_reshape(const Tensor& src, const TensorShape& new_shape);

// Slice along given dimension
Tensor tensor_slice(const Tensor& src, int dim, int start, int length);

// Copy tensor data
void tensor_copy(Tensor& dst, const Tensor& src);

// Scale tensor by scalar
void tensor_scale(Tensor& x, float scale);

} // namespace continuum

#endif // CONTINUUM_TENSOR_H
