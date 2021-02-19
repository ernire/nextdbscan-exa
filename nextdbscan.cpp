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

#include <iostream>
#include <chrono>
#include "nextdbscan.h"
#include "data_process.h"
#include "magma_input.h"
#include "magma_util.h"


nextdbscan::result nextdbscan::start(int const m, float const e, int const n_thread, std::string const &in_file,
        magmaMPI mpi, bool const is_approximate) noexcept {

    if (mpi.rank == 0) {
        std::cout << "Total of " << (n_thread * mpi.n_nodes) << " cores used on " << mpi.n_nodes << " node(s)." << std::endl;
    }

    auto start_timestamp = std::chrono::high_resolution_clock::now();

    h_vec<float> v_coord;
    int n_total_coord = -1, n_dim = -1;
    magma_util::measure_duration("Read Input Data: ", mpi.rank == 0, [&]() -> void {
        magma_input::read_input(in_file, v_coord, n_total_coord, n_dim, mpi.n_nodes, mpi.rank);
    });
    if (mpi.rank == 0) {
        std::cout << "Read " << n_total_coord << " aggregated points with " << n_dim << " dimensions. " << std::endl;
    }
    auto start_timestamp_no_io = std::chrono::high_resolution_clock::now();
    data_process dp(v_coord, m, e, n_dim, n_total_coord, is_approximate);

    magma_util::measure_duration("Classify Points: ", mpi.rank == 0, [&]() -> void {
        dp.select_and_process(mpi);
    });

    magma_util::measure_duration("Assign Labels: ", mpi.rank == 0, [&]() -> void {
        dp.select_cores_and_process(mpi);
    });

    auto result = nextdbscan::result();
    magma_util::measure_duration("Collect Results: ", mpi.rank == 0, [&]() -> void {
        dp.get_result_meta(result.core_count, result.noise, result.clusters, result.n, mpi);
    });
    auto end_timestamp = std::chrono::high_resolution_clock::now();
    auto total_dur = std::chrono::duration_cast<std::chrono::milliseconds>(end_timestamp - start_timestamp).count();
    auto total_dur_no_io = std::chrono::duration_cast<std::chrono::milliseconds>(end_timestamp - start_timestamp_no_io).count();
    if (mpi.rank == 0) {
        std::cout << "Total Execution Time: " << total_dur << " milliseconds" << std::endl;
        std::cout << "Total Execution Time (without I/O): " << total_dur_no_io << " milliseconds" << std::endl;
    }
    return result;
}
