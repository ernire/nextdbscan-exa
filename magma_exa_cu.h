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

#ifndef NEXTDBSCAN20_MAGMA_EXA_CU_H
#define NEXTDBSCAN20_MAGMA_EXA_CU_H

#include <cassert>
#include <iostream>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
template <typename T>
using h_vec = thrust::host_vector<T>;
template <typename T>
using d_vec = thrust::device_vector<T>;
#include <thrust/binary_search.h>
#include <thrust/functional.h>

namespace exa {

    template <typename T, typename std::enable_if<std::is_arithmetic<T>::value>::type* = nullptr>
    void fill(d_vec<T> &v, std::size_t const begin, std::size_t const end, T const val) noexcept {
#ifdef DEBUG_ON
        assert(begin <= end);
#endif
        thrust::fill(v.begin() + begin, v.begin() + end, val);
    }

    template <typename T, typename std::enable_if<std::is_arithmetic<T>::value>::type* = nullptr>
    void iota(d_vec<T> &v, std::size_t const begin, std::size_t const end, std::size_t const startval) noexcept {
#ifdef DEBUG_ON
        assert(begin <= end);
#endif
        thrust::sequence(v.begin() + begin, v.begin() + end, startval);
    }

    template <typename T, typename std::enable_if<std::is_arithmetic<T>::value>::type* = nullptr>
    T reduce(d_vec<T> &v, std::size_t const begin, std::size_t const end, T const startval) noexcept {
#ifdef DEBUG_ON
        assert(begin <= end);
        assert((end - begin) <= (v.size() - begin));
#endif
        return thrust::reduce(v.begin() + begin, v.begin() + end, startval);
    }

    template <typename T, typename F, typename std::enable_if<std::is_arithmetic<T>::value>::type* = nullptr>
    std::size_t count_if(d_vec<T> &v, std::size_t const begin, std::size_t const end, F const &functor) noexcept {
#ifdef DEBUG_ON
        assert(begin <= end);
#endif
        auto iter1 = v.begin();
        thrust::advance(iter1, begin);
        auto iter2 = v.begin();
        thrust::advance(iter2, end);
//        return thrust::count_if(v.begin() + begin, v.begin() + end, functor);
        return thrust::count_if(iter1, iter2, functor);
    }

    template <typename T, typename std::enable_if<std::is_arithmetic<T>::value>::type* = nullptr>
    void exclusive_scan(d_vec<T> const &v_input, std::size_t const in_begin, std::size_t const in_end,
            d_vec<T> &v_output, std::size_t const out_begin, T const init) noexcept {
#ifdef DEBUG_ON
        assert(in_begin <= in_end);
#endif
        thrust::exclusive_scan(v_input.begin() + in_begin, v_input.begin() + in_end, v_output.begin() + out_begin, init);
    }

    template <typename T, typename F, typename std::enable_if<std::is_arithmetic<T>::value>::type* = nullptr>
    void copy_if(d_vec<T> &v_input, std::size_t const in_begin, std::size_t const in_end, d_vec<T> &v_output,
            std::size_t const out_begin, F const &functor) noexcept {
#ifdef DEBUG_ON
        assert(in_begin <= in_end);
#endif
        auto it = thrust::copy_if(v_input.begin() + in_begin, v_input.begin() + in_end, v_output.begin() + out_begin,
                functor);
        v_output.resize(thrust::distance(v_output.begin(), it));
    }

    template <typename F>
    void for_each(std::size_t const begin, std::size_t const end, F const &functor) noexcept {
#ifdef DEBUG_ON
        assert(begin <= end);
#endif
//        thrust::counting_iterator<std::size_t> it_cnt_begin(begin);
//        thrust::counting_iterator<std::size_t> it_cnt_end = it_cnt_begin + (end - begin);
//        thrust::for_each(it_cnt_begin, it_cnt_begin + (end - begin), functor);
//        thrust::make_counting_iterator(0),
        thrust::for_each(thrust::make_counting_iterator(begin), thrust::make_counting_iterator(begin + (end - begin)), functor);
//        thrust::for_each(it_cnt_begin, it_cnt_end, functor);
    }

    template <typename F>
    void for_each_experimental(std::size_t const begin, std::size_t const end, F const &functor) noexcept {
#ifdef DEBUG_ON
        assert(begin <= end);
#endif
        thrust::counting_iterator<int> it_cnt_begin(begin);
        thrust::counting_iterator<int> it_cnt_end = it_cnt_begin + (end - begin);
        thrust::for_each(thrust::device, it_cnt_begin, it_cnt_end, [=]__device__(auto const &i) {
            functor(i);
        });
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
//        thrust::counting_iterator<int> it_cnt_begin(in_begin);
        thrust::counting_iterator<int> it_cnt_begin(0);
        auto it_trans_begin = thrust::make_transform_iterator(it_cnt_begin, (thrust::placeholders::_1 * (stride + 1)) + out_begin);
        auto it_perm_begin = thrust::make_permutation_iterator(v_output.begin(), it_trans_begin);
        thrust::lower_bound(v_input.begin() + in_begin, v_input.begin() + in_end, v_value.begin() + value_begin,
                v_value.begin() + value_end, it_perm_begin);
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
//        thrust::counting_iterator<int> it_cnt_begin(in_begin);
        thrust::counting_iterator<int> it_cnt_begin(0);
        auto it_trans_begin = thrust::make_transform_iterator(it_cnt_begin, (thrust::placeholders::_1 * (stride + 1)) + out_begin);
        auto it_perm_begin = thrust::make_permutation_iterator(v_output.begin(), it_trans_begin);
        thrust::lower_bound(v_input.begin() + in_begin, v_input.begin() + in_end, v_value.begin() + value_begin,
                v_value.begin() + value_end, it_perm_begin, functor);
    }



    template <typename T, typename std::enable_if<std::is_arithmetic<T>::value>::type* = nullptr>
    void upper_bound(d_vec<T> &v_input, std::size_t const in_begin, std::size_t const in_end,
            d_vec<T> &v_value, std::size_t const value_begin, std::size_t const value_end,
            d_vec<int> &v_output, std::size_t const out_begin, int const stride) noexcept {
#ifdef DEBUG_ON
        assert(in_begin <= in_end);
        assert(value_begin <= value_end);
        assert(v_input.size() >= (in_end - in_begin));
        assert(v_output.size() >= ((value_end - value_begin - 1) * (stride + 1)) + out_begin);
#endif
        thrust::counting_iterator<int> it_cnt_begin(0);
        auto it_trans_begin = thrust::make_transform_iterator(it_cnt_begin, (thrust::placeholders::_1 * (stride + 1)) + out_begin);
        auto it_perm_begin = thrust::make_permutation_iterator(v_output.begin(), it_trans_begin);
        thrust::upper_bound(v_input.begin() + in_begin, v_input.begin() + in_end, v_value.begin() + value_begin,
                v_value.begin() + value_end, it_perm_begin);
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
        thrust::counting_iterator<int> it_cnt_begin(0);
        auto it_trans_begin = thrust::make_transform_iterator(it_cnt_begin, (thrust::placeholders::_1 * (stride + 1)) + out_begin);
        auto it_perm_begin = thrust::make_permutation_iterator(v_output.begin(), it_trans_begin);
        thrust::upper_bound(v_input.begin() + in_begin, v_input.begin() + in_end, v_value.begin() + value_begin,
                v_value.begin() + value_end, it_perm_begin, functor);
    }

    template <typename T, typename F, typename std::enable_if<std::is_arithmetic<T>::value>::type* = nullptr>
    thrust::pair<T, T> minmax_element(d_vec<T> &v, std::size_t const begin, std::size_t const end, F const &functor) noexcept {
#ifdef DEBUG_ON
        assert(begin <= end);
#endif
        auto pair = thrust::minmax_element(v.begin() + begin, v.begin() + end, functor);
        return thrust::make_pair(*pair.first, *pair.second);
    }

    template <typename T, typename F, typename std::enable_if<std::is_arithmetic<T>::value>::type* = nullptr>
    void sort(d_vec<T> &v, std::size_t const begin, std::size_t const end, F const &functor) noexcept {
#ifdef DEBUG_ON
        assert(begin <= end);
        assert((end - begin) <= (v.size() - begin));
#endif
        thrust::sort(v.begin() + begin, v.begin() + end, functor);
    }

    template <typename T1, typename T2, typename F, typename std::enable_if<std::is_arithmetic<T1>::value>::type* = nullptr>
    void unique(d_vec<T1> &v_input, d_vec<T2> &v_output, std::size_t const in_begin, std::size_t const in_end,
            std::size_t const out_begin, F const &functor) noexcept {
#ifdef DEBUG_ON
        assert(in_begin <= in_end);
#endif
    }

    template <typename T1, typename T2, typename F, typename std::enable_if<std::is_arithmetic<T1>::value>::type* = nullptr>
    void transform(d_vec<T1> const &v_input, std::size_t const in_begin, std::size_t const in_end, d_vec<T2> &v_output,
            std::size_t const out_begin, F const &functor) noexcept {
#ifdef DEBUG_ON
        assert(in_begin <= in_end);
#endif
        thrust::transform(v_input.begin() + in_begin, v_input.begin() + in_end, v_output.begin() + out_begin, functor);
    }

    template <typename T, typename std::enable_if<std::is_arithmetic<T>::value>::type* = nullptr>
    __device__
    void atomic_add(thrust::device_ptr<T> v, T const val) {
        T * ptr = thrust::raw_pointer_cast(v);
        atomicAdd(ptr, val);
    }

    template <typename T, typename std::enable_if<std::is_arithmetic<T>::value>::type* = nullptr>
    __device__
    void atomic_min(thrust::device_ptr<T> v, thrust::device_reference<T> val) {
        T * ptr = thrust::raw_pointer_cast(v);
        atomicMin(ptr, val);
    }

    template <typename T, typename std::enable_if<std::is_arithmetic<T>::value>::type* = nullptr>
    __device__
    void atomic_min(thrust::device_ptr<T> v, T val) {
        T * ptr = thrust::raw_pointer_cast(v);
        atomicMin(ptr, val);
    }

    template <typename T, typename std::enable_if<std::is_arithmetic<T>::value>::type* = nullptr>
    void copy(d_vec<T> const &v_input, std::size_t const in_begin, std::size_t const in_end, d_vec<T> &v_output,
            std::size_t const out_begin) {
        thrust::copy(v_input.begin() + in_begin, v_input.begin() + in_end, v_output.begin() + out_begin);
    }

};
#endif //NEXTDBSCAN20_MAGMA_EXA_CU_H
