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

namespace exa {
    template <typename T, typename F, typename std::enable_if<std::is_arithmetic<T>::value>::type* = nullptr>
    void copy_if(d_vec<T> &v_input, std::size_t const in_begin, std::size_t const in_end, d_vec<T> &v_output,
            std::size_t const out_begin, F const &functor) {
#ifdef DEBUG_ON
        assert(in_begin <= in_end);
#endif
        if (v_output.size() < v_input.size() + out_begin) {
            v_output.resize(out_begin + in_end - in_begin);
        }
        auto it = std::copy_if(std::next(v_input.begin(), in_begin),
                std::next(v_input.begin(), in_end), std::next(v_output.begin(), out_begin), functor);
        v_output.resize(std::distance(v_output.begin(), it));
    }

    template <typename T, typename F, typename std::enable_if<std::is_arithmetic<T>::value>::type* = nullptr>
    T count_if(d_vec<T> &v, std::size_t const begin, std::size_t const end, F const &functor) {
#ifdef DEBUG_ON
        assert(begin <= end);
#endif
        return std::count_if(std::next(v.begin(), begin),std::next(v.begin(), end), functor);
    }

    template <typename T, typename std::enable_if<std::is_arithmetic<T>::value>::type* = nullptr>
    void fill(d_vec<T> &v, std::size_t const begin, std::size_t const end, T const val) noexcept {
#ifdef DEBUG_ON
        assert(begin <= end);
#endif
        std::fill(std::next(v.begin(), begin), std::next(v.begin(), end), val);
    }

    template <typename F>
    void for_each(std::size_t const begin, std::size_t const end, F const &functor) {
#ifdef DEBUG_ON
        assert(begin <= end);
#endif
        for (std::size_t i = begin; i < end; ++i) {
            functor(i);
        }
    }

    template <typename F>
    void for_each_dynamic(std::size_t const begin, std::size_t const end, F const &functor) {
#ifdef DEBUG_ON
        assert(begin <= end);
#endif
        for (std::size_t i = begin; i < end; ++i) {
            functor(i);
        }
    }

    template <typename T, typename std::enable_if<std::is_arithmetic<T>::value>::type* = nullptr>
    void exclusive_scan(d_vec<T> &v_input, std::size_t const in_begin, std::size_t const in_end,
            d_vec<T> &v_output, std::size_t const out_begin, T const init) noexcept {
#ifdef DEBUG_ON
        assert(in_begin <= in_end);
#endif
        auto val = init;
        auto out = out_begin;
        v_output[out++] = val;
        for (int i = in_begin + 1; i < in_end; ++i, ++out) {
            v_output[out] = v_input[i - 1] + v_output[out - 1];
        }
    }

    template <typename T, typename std::enable_if<std::is_arithmetic<T>::value>::type* = nullptr>
    void iota(d_vec<T> &v, std::size_t const begin, std::size_t const end, int const startval) noexcept {
#ifdef DEBUG_ON
        assert(begin <= end);
#endif
        std::iota(std::next(v.begin(), begin), std::next(v.begin(), end), startval);
    }

    template <typename T, typename F, typename std::enable_if<std::is_arithmetic<T>::value>::type* = nullptr>
    std::pair<T, T> minmax_element(d_vec<T> &v, std::size_t const begin, std::size_t const end, F const &functor) noexcept {
#ifdef DEBUG_ON
        assert(begin <= end);
#endif
        auto minmax = std::minmax_element(std::next(v.begin(), begin), std::next(v.begin(), end), functor);
        return std::make_pair(*minmax.first, *minmax.second);
//        std::iota(std::next(v.begin(), begin), std::next(v.begin(), end), startval);
    }

    template <typename T, typename std::enable_if<std::is_arithmetic<T>::value>::type* = nullptr>
    T reduce(d_vec<T> &v, std::size_t const begin, std::size_t const end, T const startval) {
#ifdef DEBUG_ON
        assert(begin <= end);
        assert((end - begin) <= (v.size() - begin));
#endif
        auto val = startval;
        for (int i = begin; i < end; ++i) {
            val += v[i];
        }
        return val;
    }

    template <typename T, typename F, typename std::enable_if<std::is_arithmetic<T>::value>::type* = nullptr>
    void sort(d_vec<T> &v, std::size_t const begin, std::size_t const end, F const &functor) noexcept {
#ifdef DEBUG_ON
        assert(begin <= end);
        assert((end - begin) <= (v.size() - begin));
#endif
        std::sort(std::next(v.begin(), begin), std::next(v.begin(), end), functor);
    }

    template <typename T1, typename T2, typename F, typename std::enable_if<std::is_arithmetic<T1>::value>::type* = nullptr>
    void unique(d_vec<T1> &v_input, d_vec<T2> &v_output, std::size_t const in_begin, std::size_t const in_end,
            std::size_t const out_begin, F const &functor) {
#ifdef DEBUG_ON
        assert(in_begin <= in_end);
#endif
        v_output.resize(1, 0);
        exa::copy_if(v_input, 1, v_input.size(), v_output, 1, functor);
    }

    template <typename T1, typename T2, typename F, typename std::enable_if<std::is_arithmetic<T1>::value>::type* = nullptr>
    void transform(d_vec<T1> const &v_input, std::size_t const in_begin, std::size_t const in_end, d_vec<T2> &v_output,
            std::size_t const out_begin, F const &functor) noexcept {
#ifdef DEBUG_ON
        assert(in_begin <= in_end);
#endif
        std::transform(std::next(v_input.begin(), in_begin),
                std::next(v_input.begin(), in_end), std::next(v_output.begin(), out_begin), functor);
    }

    template <typename T, typename std::enable_if<std::is_arithmetic<T>::value>::type* = nullptr>
    void atomic_min(T* address, T val) {
        *address = val;
    }

    template <typename T, typename std::enable_if<std::is_arithmetic<T>::value>::type* = nullptr>
    std::size_t lower_bound(d_vec<T> const &v_input, std::size_t const begin, std::size_t const end, T const val) {
        return std::lower_bound(std::next(v_input.begin(), begin), std::next(v_input.begin(), end), val) - v_input.begin();
    }

    template <typename T, typename std::enable_if<std::is_arithmetic<T>::value>::type* = nullptr>
    void lower_bound(d_vec<T> &v_input, std::size_t const in_begin, std::size_t const in_end,
            d_vec<T> &v_value, std::size_t const value_begin, std::size_t const value_end,
            d_vec<T> &v_output, std::size_t const out_begin, int const stride) noexcept {
#ifdef DEBUG_ON
        assert(in_begin <= in_end);
        assert(value_begin <= value_end);
        assert(v_input.size() >= (in_end - in_begin));
        assert(v_output.size() >= ((value_end - value_begin - 1) * (stride + 1)) + out_begin);
#endif
        auto it_input_begin = std::next(v_input.begin(), in_begin);
        auto it_input_end = std::next(v_input.begin(), in_end);
        for (T i = value_begin; i < value_end; ++i) {
            v_output[out_begin + (i * (stride + 1))] = std::lower_bound(it_input_begin, it_input_end, v_value[i]) - v_input.begin();
        }
    }

template <typename T, typename std::enable_if<std::is_arithmetic<T>::value>::type* = nullptr>
void copy(d_vec<T> const &v_input, std::size_t const in_begin, std::size_t const in_end, d_vec<T> &v_output,
        std::size_t const out_begin) {
std::copy(std::next(v_input.begin(), in_begin), std::next(v_input.begin(), in_end),
        std::next(v_output.begin(), out_begin));
}
}

#endif //EXAFOUNDRY_EXA_H
