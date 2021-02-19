/**
Copyright (c) 2021, Ernir Erlingsson
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
 */

#ifndef NEXTDBSCAN20_DATA_PROCESS_H
#define NEXTDBSCAN20_DATA_PROCESS_H

#ifdef CUDA_ON
__device__
#endif
static const int NO_CLUSTER = INT32_MAX;

#ifdef CUDA_ON
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
template <typename T>
using h_vec = thrust::host_vector<T>;
template <typename T>
using d_vec = thrust::device_vector<T>;
#else
#include <vector>
template <typename T>
using h_vec = std::vector<T>;
template <typename T>
using d_vec = std::vector<T>;
#endif
#include <cmath>
#include "magma_mpi.h"

class data_process {
private:
    int const m, n_dim;
    std::size_t const n_coord;
    std::size_t const n_total_coord;
    float const e, e2;
    bool const is_approximate;

public:
    unsigned long long allocated_bytes;
    d_vec<float> v_coord;
    d_vec<int> v_coord_id;
    d_vec<int> v_coord_nn;
    d_vec<int> v_coord_cluster_index;
    d_vec<int> v_cluster_label;
#ifdef CUDA_ON
    explicit data_process(h_vec<float> &v_coord, int const m, float const e, int const n_dim, int const n_total_coord, bool const is_approximate)
        : m(m), n_dim(n_dim), n_coord(v_coord.size()/n_dim), e(e), e2(e*e), n_total_coord(n_total_coord), is_approximate(is_approximate), v_coord(v_coord) {
        allocated_bytes = n_coord * n_dim * sizeof(float);
    }
#else
    explicit data_process(
            h_vec<float> &v_coord,
            int const m,
            float const e,
            int const n_dim,
            int const n_total_coord,
            bool const is_approximate)
        : m(m), n_dim(n_dim), n_coord(v_coord.size()/n_dim), e(e), e2(e*e), n_total_coord(n_total_coord),
          is_approximate(is_approximate), v_coord(std::move(v_coord)) {
        allocated_bytes = n_coord * n_dim * sizeof(float);
    }
#endif

    template <typename T, typename std::enable_if<std::is_arithmetic<T>::value>::type* = nullptr>
    void vector_resize(d_vec<T> &vec, std::size_t const new_size, T const val) {
        if (new_size > vec.size()) {
            this->allocated_bytes += ((new_size - vec.size()) * sizeof(T));
        }
        vec.resize(new_size, val);
    }

    void process_points(d_vec<int> const &v_point_id, d_vec<float> const &v_point_data) noexcept;

    void select_and_process(magmaMPI mpi) noexcept;

    void select_cores_and_process(magmaMPI mpi) noexcept;

    void process_cores(d_vec<int> const &v_core_id, d_vec<float> const &v_core_data, magmaMPI mpi) noexcept;

    void get_result_meta(
            int &cores,
            int &noise,
            int &clusters,
            int &n,
            magmaMPI mpi) noexcept;

};


#endif //NEXTDBSCAN20_DATA_PROCESS_H
