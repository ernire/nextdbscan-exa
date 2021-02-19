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

#ifndef NEXTDBSCAN20_MAGMA_UTIL_H
#define NEXTDBSCAN20_MAGMA_UTIL_H

#include <iostream>
#include <random>
#include <vector>
#include <chrono>
#include <functional>

namespace magma_util {

    template<class T, typename std::enable_if<std::is_arithmetic<T>::value>::type* = nullptr>
    T get_block_size(T block_index, T number_of_samples, T number_of_blocks) noexcept {
        T block = (number_of_samples / number_of_blocks);
        T reserve = number_of_samples % number_of_blocks;
        //    Some processes will need one more sample if the data size does not fit completely
        if (reserve > 0 && block_index < reserve) {
            return block + 1;
        }
        return block;
    }

    template<class T, typename std::enable_if<std::is_arithmetic<T>::value>::type* = nullptr>
    T get_block_offset(T block_index, T number_of_samples, T number_of_blocks) noexcept {
        T offset = 0;
        for (T i = 0; i < block_index; i++) {
            offset += get_block_size(i, number_of_samples, number_of_blocks);
        }
        return offset;
    }

    template<class F>
    long long measure_duration(std::string const &name, bool const is_verbose, F const &functor) noexcept {
        if (is_verbose) {
            std::cout << name << std::flush;
        }
        auto start_timestamp = std::chrono::high_resolution_clock::now();
        functor();
        auto end_timestamp = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_timestamp - start_timestamp).count();
        if (is_verbose) {
            std::cout << duration << " milliseconds" << std::endl;
        }
        return duration;
    }

    template<class T>
    void print_v(const std::string &name, T *v, std::size_t size) noexcept {
        std::cout << name;
        for (std::size_t i = 0; i < size; ++i) {
            std::cout << v[i] << " ";
        }
        std::cout << std::endl;
    }

}

#endif //NEXTDBSCAN20_MAGMA_UTIL_H
