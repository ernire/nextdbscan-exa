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

#ifndef NEXTDBSCAN20_NEXTDBSCAN_H
#define NEXTDBSCAN20_NEXTDBSCAN_H

#include <string>
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

#include "magma_mpi.h"

namespace nextdbscan {

    struct result {
        int clusters;
        int noise;
        int core_count;
        int n;
    };

    result start(int m, float e, int n_thread, const std::string &in_file, magmaMPI mpi, bool is_approximate) noexcept;

}

#endif //NEXTDBSCAN20_NEXTDBSCAN_H
