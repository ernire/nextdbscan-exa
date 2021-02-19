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

#ifndef EXAFOUNDRY_EXA_H
#define EXAFOUNDRY_EXA_H

#include <vector>
#include <algorithm>
#include <cassert>
#include <vector>
#include <utility>
#include <numeric>
#include <limits>
#include <cmath>
#include <iostream>
#include <omp.h>
#include "magma_util.h"

template <typename T>
using h_vec = std::vector<T>;
template <typename T>
using d_vec = std::vector<T>;

namespace exa {

    template <typename T, typename std::enable_if<std::is_arithmetic<T>::value>::type* = nullptr>
    void fill(d_vec<T> &v, std::size_t const begin, std::size_t const end, T const val) noexcept {
#ifdef DEBUG_ON
        assert(begin <= end);
#endif
        #pragma omp parallel for
        for (std::size_t i = begin; i < end; ++i) {
            v[i] = val;
        }
    }

    template <typename T, typename std::enable_if<std::is_arithmetic<T>::value>::type* = nullptr>
    void iota(d_vec<T> &v, std::size_t const begin, std::size_t const end, std::size_t const startval) noexcept {
#ifdef DEBUG_ON
        assert(begin <= end);
#endif
        #pragma omp parallel for
        for (std::size_t i = begin; i < end; ++i) {
            v[i] = startval + i - begin;
        }
    }

    template <typename T, typename std::enable_if<std::is_arithmetic<T>::value>::type* = nullptr>
    T reduce(d_vec<T> &v, std::size_t const begin, std::size_t const end, T const startval) noexcept {
#ifdef DEBUG_ON
        assert(begin <= end);
        assert((end - begin) <= (v.size() - begin));
#endif
        T sum = startval;
        #pragma omp parallel for reduction(+: sum)
        for (std::size_t i = begin; i < end; ++i) {
            sum += v[i];
        }
        return sum;
    }

    template <typename T, typename F, typename std::enable_if<std::is_arithmetic<T>::value>::type* = nullptr>
    std::size_t count_if(d_vec<T> &v, std::size_t const begin, std::size_t const end, F const &functor) noexcept {
#ifdef DEBUG_ON
        assert(begin <= end);
#endif
        std::size_t cnt = 0;
        #pragma omp parallel for reduction(+:cnt)
        for (std::size_t i = begin; i < end; ++i) {
            if (functor(v[i]))
                ++cnt;
        }
        return cnt;
    }

    template <typename T, typename std::enable_if<std::is_arithmetic<T>::value>::type* = nullptr>
    void exclusive_scan(d_vec<T> const &v_input, std::size_t const in_begin, std::size_t const in_end,
            d_vec<T> &v_output, std::size_t const out_begin, T const init) noexcept {
#ifdef DEBUG_ON
        assert(in_begin <= in_end);
#endif
        d_vec<T> v_t_size;
        d_vec<T> v_t_offset;
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            #pragma omp single
            {
                v_t_size.resize(omp_get_num_threads(), 0);
            }
            int t_size = magma_util::get_block_size(tid, static_cast<int>(in_end - in_begin), static_cast<int>(v_t_size.size()));
            int t_offset = magma_util::get_block_offset(tid, static_cast<int>(in_end - in_begin), static_cast<int>(v_t_size.size()));
            T size_sum = 0;
            for (auto i = t_offset; i < t_offset + t_size; ++i) {
                size_sum += v_input[i + in_begin];
            }
            v_t_size[tid] = size_sum;
            #pragma omp barrier
            #pragma omp single
            {
                v_t_offset = v_t_size;
                v_t_offset[0] = init;
                for (std::size_t i = 1; i < v_t_offset.size(); ++i) {
                    v_t_offset[i] = v_t_offset[i - 1] + v_t_size[i - 1];
                }

            }
            v_output[t_offset] = v_t_offset[tid];
            for (auto i = t_offset + 1; i < t_offset + t_size; ++i) {
                v_output[out_begin + i] = v_input[i + in_begin - 1] + v_output[out_begin + i - 1];
            }
        }
    }

    template <typename T, typename F, typename std::enable_if<std::is_arithmetic<T>::value>::type* = nullptr>
    void copy_if(d_vec<T> &v_input, std::size_t const in_begin, std::size_t const in_end, d_vec<T> &v_output,
            std::size_t const out_begin, F const &functor) noexcept {
#ifdef DEBUG_ON
        assert(in_begin <= in_end);
#endif
        d_vec<int> v_t_val;
        int n_thread;
        #pragma omp parallel
        {
            #pragma omp single
            {
                n_thread = omp_get_num_threads();
                v_t_val.resize(n_thread * 2);
            }
            int tid = omp_get_thread_num();
            int size = magma_util::get_block_size(tid, static_cast<int>(in_end - in_begin), n_thread);
            int offset = magma_util::get_block_offset(tid, static_cast<int>(in_end - in_begin), n_thread);
            auto it_end = std::copy_if(std::next(v_input.begin(), offset + in_begin),
                    std::next(v_input.begin(), offset + size + in_begin),
                    std::next(v_output.begin(), offset + out_begin), functor);
            v_t_val[tid * 2] = offset + out_begin;
            v_t_val[tid * 2 + 1] = it_end - v_output.begin();
        }
        auto it_out = std::next(v_output.begin(), v_t_val[1]);
        for (int t = 1; t < n_thread; ++t) {
            it_out = std::move(std::next(v_output.begin(), v_t_val[t * 2]),
                    std::next(v_output.begin(), v_t_val[t * 2 + 1]),
                    it_out);
        }
        v_output.resize(std::distance(v_output.begin(), it_out));
    }

    template <typename F>
    void for_each(std::size_t const begin, std::size_t const end, F const &functor) noexcept {
#ifdef DEBUG_ON
        assert(begin <= end);
#endif
        #pragma omp parallel for
        for (std::size_t i = begin; i < end; ++i) {
            functor(i);
        }
    }

    template <typename T, typename std::enable_if<std::is_arithmetic<T>::value>::type* = nullptr>
    std::size_t lower_bound(d_vec<T> const &v_input, std::size_t const begin, std::size_t const end, T const val) {
        return std::lower_bound(std::next(v_input.begin(), begin), std::next(v_input.begin(), end), val) - v_input.begin();
    }

    template <typename T, typename std::enable_if<std::is_arithmetic<T>::value>::type* = nullptr>
    void lower_bound(d_vec<T> &v_input, std::size_t const in_begin, std::size_t const in_end,
            d_vec<T> &v_value, std::size_t const value_begin, std::size_t const value_end,
            d_vec<int> &v_output, std::size_t const out_begin, int const stride) noexcept {
#ifdef DEBUG_ON
        assert(in_begin <= in_end);
        assert(value_begin <= value_end);
        assert(v_input.size() >= (in_end - in_begin));
        assert(v_output.size() >= ((value_end - value_begin - 1) * (stride + 1)) + out_begin);
#endif
        auto it_input_begin = std::next(v_input.begin(), in_begin);
        auto it_input_end = std::next(v_input.begin(), in_end);
        #pragma omp parallel for
        for (std::size_t i = 0; i < value_end - value_begin; ++i) {
            v_output[out_begin + (i * (stride + 1))] = std::lower_bound(it_input_begin, it_input_end, v_value[i + value_begin]) - it_input_begin;
        }
    }

    template <typename T, typename F, typename std::enable_if<std::is_arithmetic<T>::value>::type* = nullptr>
    void lower_bound(d_vec<T> &v_input, std::size_t const in_begin, std::size_t const in_end,
            d_vec<T> &v_value, std::size_t const value_begin, std::size_t const value_end,
            d_vec<int> &v_output, std::size_t const out_begin, int const stride, F const &functor) noexcept {
#ifdef DEBUG_ON
        assert(in_begin <= in_end);
        assert(value_begin <= value_end);
        assert(v_input.size() >= (in_end - in_begin));
        assert(v_output.size() >= ((value_end - value_begin - 1) * (stride + 1)) + out_begin);
#endif
        auto it_input_begin = std::next(v_input.begin(), in_begin);
        auto it_input_end = std::next(v_input.begin(), in_end);
#pragma omp parallel for
        for (std::size_t i = 0; i < value_end - value_begin; ++i) {
            v_output[out_begin + (i * (stride + 1))] =
                    std::lower_bound(it_input_begin, it_input_end, v_value[i + value_begin], functor) - it_input_begin;
        }
    }

    template <typename T, typename T2, typename std::enable_if<std::is_arithmetic<T>::value>::type* = nullptr>
    void upper_bound(d_vec<T> &v_input, std::size_t const in_begin, std::size_t const in_end,
            d_vec<T> &v_value, std::size_t const value_begin, std::size_t const value_end,
            d_vec<T2> &v_output, std::size_t const out_begin, int const stride) noexcept {
#ifdef DEBUG_ON
        assert(in_begin <= in_end);
        assert(value_begin <= value_end);
        assert(v_input.size() >= (in_end - in_begin));
        assert(v_output.size() >= ((value_end - value_begin - 1) * (stride + 1)) + out_begin);
#endif
        auto it_input_begin = std::next(v_input.begin(), in_begin);
        auto it_input_end = std::next(v_input.begin(), in_end);
#pragma omp parallel for
        for (std::size_t i = 0; i < value_end - value_begin; ++i) {
            v_output[out_begin + (i * (stride + 1))] = std::upper_bound(it_input_begin, it_input_end, v_value[i + value_begin]) - it_input_begin;
        }
    }

    template <typename T, typename F, typename std::enable_if<std::is_arithmetic<T>::value>::type* = nullptr>
    void upper_bound(d_vec<T> &v_input, std::size_t const in_begin, std::size_t const in_end,
            d_vec<T> &v_value, std::size_t const value_begin, std::size_t const value_end,
            d_vec<int> &v_output, std::size_t const out_begin, int const stride, F const &functor) noexcept {
#ifdef DEBUG_ON
        assert(in_begin <= in_end);
        assert(value_begin <= value_end);
        assert(v_input.size() >= (in_end - in_begin));
        assert(v_output.size() >= ((value_end - value_begin - 1) * (stride + 1)) + out_begin);
#endif
        auto it_input_begin = std::next(v_input.begin(), in_begin);
        auto it_input_end = std::next(v_input.begin(), in_end);
#pragma omp parallel for
        for (std::size_t i = 0; i < value_end - value_begin; ++i) {
            v_output[out_begin + (i * (stride + 1))] =
                    std::upper_bound(it_input_begin, it_input_end, v_value[i + value_begin], functor) - it_input_begin;
        }
    }

    template <typename T, typename F, typename std::enable_if<std::is_arithmetic<T>::value>::type* = nullptr>
    std::pair<T, T> minmax_element(d_vec<T> &v, std::size_t const begin, std::size_t const end, F const &functor) noexcept {
#ifdef DEBUG_ON
        assert(begin <= end);
#endif
        if (v.size() < 256) {
            auto pair = std::minmax_element(std::next(v.begin(), begin), std::next(v.begin(), end), functor);
            return std::make_pair(*pair.first, *pair.second);
        }
        d_vec<T> v_t_min_max;
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            #pragma omp single
            {
                v_t_min_max.resize(omp_get_num_threads()*2, 0);
            }
            int size = magma_util::get_block_size(tid, (int)(end - begin), static_cast<int>(v_t_min_max.size() / 2));
            int offset = magma_util::get_block_offset(tid, (int)(end - begin), static_cast<int>(v_t_min_max.size() / 2));
            auto pair = std::minmax_element(std::next(v.begin(), offset + begin), std::next(v.begin(), offset + size + begin),
                    functor);
            v_t_min_max[tid * 2] = *pair.first;
            v_t_min_max[tid * 2 + 1] = *pair.second;
        };
        std::sort(v_t_min_max.begin(), v_t_min_max.end(), functor);
        return std::make_pair(v_t_min_max[0], v_t_min_max[v_t_min_max.size()-1]);
    }

    template <typename T, typename F, typename std::enable_if<std::is_arithmetic<T>::value>::type* = nullptr>
    void sort(d_vec<T> &v, std::size_t const begin, std::size_t const end, F const &functor) noexcept {
#ifdef DEBUG_ON
        assert(begin <= end);
        assert((end - begin) <= (v.size() - begin));
#endif

        // TODO improve heuristic
        if (end - begin < 100000) {
            std::sort(std::next(v.begin(), begin), std::next(v.begin(), end), functor);
            return;
        }
        auto const range_begin = std::next(v.begin(), begin);
        auto const range_size = end - begin;

        d_vec<T> v_tmp(range_size);
        d_vec<T> v_samples;
        d_vec<int> v_bucket_size;
        d_vec<int> v_bucket_offset;
        d_vec<int> v_par_bucket_size;
        d_vec<int> v_par_bucket_offset;
        // optimize to only use needed space
        d_vec<int> v_bucket_index(range_size);
        int n_thread = 0, n_bucket = 0;
        #pragma omp parallel
        {
            #pragma omp single
            {
                n_thread = omp_get_num_threads();
                int n_samples = n_thread * log10f(range_size);
                v_samples.reserve(n_samples);
                for (int i = 0; i < n_samples; ++i) {
                    v_samples.push_back(v[(((range_size - 1) / n_samples) * i) + begin]);
                }
//                std::cout << "sample size: " << n_samples << std::endl;
                std::sort(v_samples.begin(), v_samples.end(), functor);
                n_bucket = v_samples.size() + 1;
                v_bucket_size.resize(n_bucket, 0);
                v_bucket_offset.resize(n_bucket);
                v_par_bucket_size.resize(n_bucket * n_thread, -1);
                v_par_bucket_offset.resize(n_bucket * n_thread);
            }
            d_vec<int> v_t_size(n_bucket, 0);
            d_vec<int> v_t_offset(n_bucket);
            int tid = omp_get_thread_num();
            int t_size = magma_util::get_block_size(tid, static_cast<int>(range_size), n_thread);
            int t_offset = magma_util::get_block_offset(tid, static_cast<int>(range_size), n_thread);
            for (int i = t_offset; i < t_offset + t_size; ++i) {
                v_bucket_index[i] = std::lower_bound(v_samples.begin(), v_samples.end(), v[i + begin], functor)
                        - v_samples.begin();
                ++v_t_size[v_bucket_index[i]];
            }
            for (int i = 0; i < n_bucket; ++i) {
                v_par_bucket_size[(i * n_thread) + tid] = v_t_size[i];
            }
            #pragma omp barrier
            for (int i = 0; i < v_bucket_size.size(); ++i) {
                #pragma omp atomic
                v_bucket_size[i] += v_t_size[i];
            }
            #pragma omp barrier
            #pragma omp single
            {
                exclusive_scan(v_par_bucket_size, 0, v_par_bucket_size.size(), v_par_bucket_offset, 0, 0);
                exclusive_scan(v_bucket_size, 0, v_bucket_size.size(), v_bucket_offset, 0, 0);
            }
            for (int i = 0; i < n_bucket; ++i) {
                v_t_offset[i] = v_par_bucket_offset[i * n_thread + tid];
            }
            auto v_t_offset_cpy = v_t_offset;
            for (std::size_t i = t_offset; i < t_offset + t_size; ++i) {
                v_tmp[v_t_offset[v_bucket_index[i]]] = v[i + begin];
                ++v_t_offset[v_bucket_index[i]];
            }
            #pragma omp barrier
            #pragma omp for schedule(dynamic)
            for (int i = 0; i < v_bucket_size.size(); ++i) {
                std::sort(std::next(v_tmp.begin(), v_bucket_offset[i]), std::next(v_tmp.begin(),
                        v_bucket_offset[i] + v_bucket_size[i]), functor);
            }
        }
        std::copy(v_tmp.begin(), v_tmp.end(), range_begin);
    }

    template <typename T1, typename T2, typename F, typename std::enable_if<std::is_arithmetic<T1>::value>::type* = nullptr>
    void unique(d_vec<T1> &v_input, std::size_t const in_begin, std::size_t const in_end, d_vec<T2> &v_output,
            std::size_t const out_begin, F const &functor) noexcept {
#ifdef DEBUG_ON
        assert(in_begin <= in_end);
#endif
//        v_output.resize(1, 0);
        v_output[0] = 0;
        exa::copy_if(v_input, 1, v_input.size(), v_output, 1, functor);
    }

    template <typename T1, typename T2, typename F, typename std::enable_if<std::is_arithmetic<T1>::value>::type* = nullptr>
    void transform(d_vec<T1> const &v_input, std::size_t const in_begin, std::size_t const in_end, d_vec<T2> &v_output,
            std::size_t const out_begin, F const &functor) noexcept {
#ifdef DEBUG_ON
        assert(in_begin <= in_end);
#endif
        #pragma omp parallel for
        for (std::size_t i = in_begin; i < in_end; ++i) {
            v_output[out_begin + i - in_begin] = functor(v_input[i]);
        }
    }

    template<class T, class O>
    void _atomic_op(T* address, T value, O op) {
        T previous = __sync_fetch_and_add(address, 0);

        while (op(value, previous)) {
            if  (__sync_bool_compare_and_swap(address, previous, value)) {
                break;
            } else {
                previous = __sync_fetch_and_add(address, 0);
            }
        }
    }

    template <typename T, typename std::enable_if<std::is_arithmetic<T>::value>::type* = nullptr>
    void atomic_min(T* address, T val) {
        _atomic_op(address, val, std::less<>());
    }

    template <typename T, typename std::enable_if<std::is_arithmetic<T>::value>::type* = nullptr>
    void atomic_max(T* address, T val) {
        _atomic_op(address, val, std::greater<>());
    }

    template <typename T, typename std::enable_if<std::is_arithmetic<T>::value>::type* = nullptr>
    void atomic_add(T* address, T const val) {
        #pragma omp atomic
        *address += val;
    }

    template <typename T, typename std::enable_if<std::is_arithmetic<T>::value>::type* = nullptr>
    void copy(d_vec<T> const &v_input, std::size_t const in_begin, std::size_t const in_end, d_vec<T> &v_output,
            std::size_t const out_begin) {
        std::copy(std::next(v_input.begin(), in_begin), std::next(v_input.begin(), in_end),
                std::next(v_output.begin(), out_begin));
    }
}

#endif //EXAFOUNDRY_EXA_H
