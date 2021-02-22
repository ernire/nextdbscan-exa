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
#include <stack>
#include "magma_util.h"
#include "data_process.h"
#ifdef OMP_ON
#include "magma_exa_omp.h"
#elif CUDA_ON
#include "magma_exa_cu.h"
#else
#include "magma_exa.h"
#endif

#ifdef CUDA_ON
void print_cuda_memory_usage() {
    size_t free_byte;
    size_t total_byte;
    auto cuda_status = cudaMemGetInfo( &free_byte, &total_byte );

    if ( cudaSuccess != cuda_status ) {
        printf("Error: cudaMemGetInfo fails, %s \n", cudaGetErrorString(cuda_status) );
        exit(1);
    }
    double free_db = (double)free_byte ;
    double total_db = (double)total_byte ;
    double used_db = total_db - free_db ;
    printf("GPU memory usage: used = %f, free = %f MB, total = %f MB\n",
            used_db/1024.0/1024.0, free_db/1024.0/1024.0, total_db/1024.0/1024.0);
}
#endif

void data_process::get_result_meta(int &cores, int &noise, int &clusters, int &n, magmaMPI mpi) noexcept {
    n = n_coord;
    auto const _m = m;
    cores = exa::count_if(v_coord_nn, 0, v_coord_nn.size(), [=]
#ifdef CUDA_ON
    __device__
#endif
    (int const &v) -> bool {
        return v >= _m;
    });

    noise = exa::count_if(v_coord_cluster_index, 0, v_coord_cluster_index.size(), []
#ifdef CUDA_ON
    __device__
#endif
    (int const &v) -> bool {
        return v == NO_CLUSTER;
    });

#ifdef MPI_ON
    d_vec<int> v_data(2);
    v_data[0] = cores;
    v_data[1] = noise;
    if (mpi.n_nodes > 1)
        mpi.allReduce(v_data, magmaMPI::sum);
    cores = v_data[0];
    noise = v_data[1];
#endif

    auto const it_cluster_label = v_cluster_label.begin();
    d_vec<int> v_cluster_iota(v_cluster_label.size());
    exa::iota(v_cluster_iota, 0, v_cluster_iota.size(), 0);
    clusters = exa::count_if(v_cluster_iota, 0, v_cluster_iota.size(), [=]
#ifdef CUDA_ON
        __device__
#endif
    (int const &i) -> bool {
        return it_cluster_label[i] == i;
    });

    // TODO write to file
}

void data_process::process_cores(d_vec<int> const &v_core_id, d_vec<float> const &v_core_data, magmaMPI mpi) noexcept {
    auto const it_core_id = v_core_id.begin();
    float const _max_float = std::numeric_limits<float>::max();
    auto const it_coord = v_coord.begin();
    auto const it_core = v_core_data.begin();
    auto const _n_cores = v_core_id.size();
    auto const it_coord_cluster_index = v_coord_cluster_index.begin();
    auto const it_coord_nn = v_coord_nn.begin();
    auto const _m = m;
    auto const _n_dim = n_dim;
    auto const _e2 = e2;

    d_vec<int> v_core_cluster_index(v_core_id.size(), NO_CLUSTER);
    auto const it_core_cluster_index = v_core_cluster_index.begin();

    d_vec<int> v_point_new_cluster_mark(v_core_id.size(), 0);
    d_vec<int> v_point_new_cluster_offset(v_core_id.size(), 0);
    auto const it_point_new_cluster_mark = v_point_new_cluster_mark.begin();

    exa::for_each(0, v_core_id.size(), [=]
#ifdef CUDA_ON
            __device__
#endif
    (auto const &jj) -> void {
        if (it_core_id[jj] >= 0) {
            auto j = it_core_id[jj];
            if (it_coord_cluster_index[j] == NO_CLUSTER) {
                it_point_new_cluster_mark[jj] = 1;
            } else {
                it_core_cluster_index[jj] = it_coord_cluster_index[j];
            }
        }
    });

#ifdef MPI_ON
    if (mpi.n_nodes > 1)
        mpi.allReduce(v_point_new_cluster_mark, magmaMPI::max);
#endif

    // count the new labels
    auto new_cluster_cores = exa::count_if(v_point_new_cluster_mark, 0, v_point_new_cluster_mark.size(), []
#ifdef CUDA_ON
            __device__
#endif
    (auto const &v) -> bool {
        return v == 1;
    });

    int cluster_index_begin = static_cast<int>(v_cluster_label.size());
    v_cluster_label.resize(cluster_index_begin + new_cluster_cores);

//#ifdef CUDA_ON
//    print_cuda_memory_usage();
//#endif

    auto const it_cluster_label = v_cluster_label.begin();
    if (new_cluster_cores > 0) {
        // Create new label ids
        exa::iota(v_cluster_label, cluster_index_begin, v_cluster_label.size(), cluster_index_begin);
        // the new label indexes
        exa::exclusive_scan(v_point_new_cluster_mark, 0, v_point_new_cluster_mark.size(), v_point_new_cluster_offset,
                0, cluster_index_begin);
        auto const it_point_new_cluster_offset = v_point_new_cluster_offset.begin();

        exa::for_each(0, v_core_id.size(), [=]
#ifdef CUDA_ON
            __device__
#endif
        (auto const &jj) -> void {
            if (it_point_new_cluster_mark[jj] == 1) {
                // mark the new cluster indexes
                it_core_cluster_index[jj] = it_point_new_cluster_offset[jj];

                if (it_core_id[jj] >= 0) {
                    it_coord_cluster_index[it_core_id[jj]] = it_core_cluster_index[jj];
                }
            }
        });
    }

#ifdef MPI_ON
    if (mpi.n_nodes > 1)
        mpi.allReduce(v_core_cluster_index, magmaMPI::min);
#endif
    int iter_cnt = 0;
    d_vec<int> v_running(1, 1);
    d_vec<int> v_running_cnt(1, 0);
    auto const it_running = v_running.begin();
    auto const it_running_cnt = v_running_cnt.begin();
    auto const _is_approximate = is_approximate;
    while (v_running[0] == 1) {
        it_running[0] = 0;
        ++iter_cnt;
        exa::for_each(0, n_coord, [=]
#ifdef CUDA_ON
        __device__
#endif
        (auto const &i) -> void {
            bool keep_running = false;
            auto const p1 = &it_coord[i * _n_dim];
            for (std::size_t jj = 0; jj < _n_cores; ++jj) {
                auto const p2 = &it_core[jj * _n_dim];
                // check for empty padding data
                if (p2[0] == _max_float) {
                    continue;
                }
                float length = 0;
                for (int d = 0; d < _n_dim; ++d) {
                    length += (p1[d] - p2[d]) * (p1[d] - p2[d]);
                }
                if (length <= _e2) {
                    if (it_coord_cluster_index[i] == NO_CLUSTER) {
                        exa::atomic_min(&it_coord_cluster_index[i], it_cluster_label[it_core_cluster_index[jj]]);
                    } else if (it_coord_nn[i] < _m) {
                        if (it_cluster_label[it_coord_cluster_index[i]] > it_cluster_label[it_core_cluster_index[jj]]) {
                            exa::atomic_min(&it_coord_cluster_index[i], it_cluster_label[it_core_cluster_index[jj]]);
                        }
                    } else {
                        if (it_cluster_label[it_coord_cluster_index[i]] > it_cluster_label[it_core_cluster_index[jj]]) {
                            exa::atomic_min(&it_cluster_label[it_coord_cluster_index[i]],
                                    it_cluster_label[it_core_cluster_index[jj]]);
                        } else if (it_cluster_label[it_coord_cluster_index[i]] < it_cluster_label[it_core_cluster_index[jj]]) {
                            if (!_is_approximate) {
                                keep_running = true;
                            }
                            exa::atomic_min(&it_cluster_label[it_core_cluster_index[jj]],
                                    it_cluster_label[it_coord_cluster_index[i]]);
                        }
                    }
                }
            }
            if (keep_running) {
                it_running[0] = 1;
                it_running_cnt[0] = it_running_cnt[0] + 1;
            }
        });

#ifdef MPI_ON
        if (mpi.n_nodes > 1) {
            mpi.allReduce(v_running, magmaMPI::max);
            mpi.allReduce(v_cluster_label, magmaMPI::min);
        }
#endif
        // flatten points
        exa::for_each(0, v_core_id.size(), [=]
#ifdef CUDA_ON
            __device__
#endif
        (auto const &jj) -> void {
            if (it_core_cluster_index[jj] == NO_CLUSTER)
                return;
            while (it_cluster_label[it_core_cluster_index[jj]] != it_core_cluster_index[jj]) {
                it_core_cluster_index[jj] = it_cluster_label[it_core_cluster_index[jj]];
            }
        });

        // flatten coords
        exa::for_each(0, n_coord, [=]
#ifdef CUDA_ON
            __device__
#endif
        (auto const &i) -> void {
            if (it_coord_cluster_index[i] == NO_CLUSTER)
                return;
            while (it_cluster_label[it_coord_cluster_index[i]] != it_coord_cluster_index[i]) {
                it_coord_cluster_index[i] = it_cluster_label[it_coord_cluster_index[i]];
            }
        });
    }
#ifdef DEBUG_ON
    if (mpi.rank == 0)
        std::cout << "FlatLabel required " << iter_cnt << " iterations" << std::endl;
#endif
}

void data_process::select_cores_and_process(magmaMPI mpi) noexcept {
    auto const it_coord_nn = v_coord_nn.begin();
    d_vec<int> v_point_iota(v_coord_id.size());
    d_vec<int> v_point_core_id(v_point_iota.size());
    auto const it_point_core_id = v_point_core_id.begin();
    auto const _m = m;
    auto const _n_dim = n_dim;
    auto const it_coord = v_coord.begin();
    exa::iota(v_point_iota, 0, v_point_iota.size(), 0);
    exa::copy_if(v_point_iota, 0, v_point_iota.size(), v_point_core_id, 0, [=]
#ifdef CUDA_ON
            __device__
#endif
    (auto const &i) -> bool {
        return it_coord_nn[i] >= _m;
    });

    std::size_t million = 1000000;
    int n_sample_size = 100 * million;
    if (n_total_coord < n_sample_size) {
        // A bit of padding is added at the end to counter rounding issues
        n_sample_size = static_cast<int>(n_total_coord) + (mpi.n_nodes * 2);
    }
    v_coord_cluster_index.resize(n_coord, NO_CLUSTER);

    d_vec<int> v_id_chunk(n_sample_size, -1);
    d_vec<float> v_data_chunk(n_sample_size * n_dim);
    auto const it_data_chunk = v_data_chunk.begin();
    d_vec<int> v_point_id(v_coord_id);
    exa::iota(v_point_id, 0, v_point_id.size(), 0);
    int node_transmit_size = magma_util::get_block_size(mpi.rank, n_sample_size, mpi.n_nodes);
    int node_transmit_offset = magma_util::get_block_offset(mpi.rank, n_sample_size, mpi.n_nodes);
    std::size_t transmit_cnt = 0;
    int n_iter = 0;
    while (transmit_cnt < n_coord) {
        exa::fill(v_id_chunk, 0, v_id_chunk.size(), -1);
        exa::fill(v_data_chunk, 0, v_data_chunk.size(), std::numeric_limits<float>::max());
        int size = node_transmit_size;
        if (transmit_cnt + node_transmit_size > v_point_core_id.size()) {
            size = v_point_core_id.size() - transmit_cnt;
        }
        if (size > 0) {
            exa::copy(v_point_core_id, transmit_cnt, transmit_cnt + size,
                    v_id_chunk, node_transmit_offset);
            exa::for_each(0, size, [=]
    #ifdef CUDA_ON
            __device__
    #endif
            (auto const ii) -> void {
                auto const i = it_point_core_id[ii + transmit_cnt];
                for (int d = 0; d < _n_dim; ++d) {
                    it_data_chunk[((node_transmit_offset + ii) * _n_dim) + d] = it_coord[i * _n_dim + d];
                }
            });
        }
#ifdef MPI_ON
        if (mpi.n_nodes > 1)
            mpi.allGather(v_data_chunk);
#endif
        ++n_iter;
        process_cores(v_id_chunk, v_data_chunk, mpi);
        transmit_cnt += node_transmit_size;
    }

}

void data_process::select_and_process(magmaMPI mpi) noexcept {
    std::size_t million = 1000000;
    vector_resize(v_coord_id, n_coord, 0);
    exa::iota(v_coord_id, 0, v_coord_id.size(), 0);
    std::size_t n_sample_size = 100 * million;
    if (n_total_coord < n_sample_size) {
        // A bit of padding is added at the end to counter rounding issues
        n_sample_size = static_cast<int>(n_total_coord) + 2;
    }
#ifdef DEBUG_ON
    if (mpi.rank == 0) {
        std::cout << "sample size: " << n_sample_size << std::endl;
    }
#endif
    d_vec<int> v_id_chunk(n_sample_size, -1);
    d_vec<float> v_data_chunk(n_sample_size * n_dim);
    int node_transmit_size = magma_util::get_block_size(mpi.rank, static_cast<int>(n_sample_size), mpi.n_nodes);
    int node_transmit_offset = magma_util::get_block_offset(mpi.rank, static_cast<int>(n_sample_size), mpi.n_nodes);
#ifdef DEBUG_ON
    std::cout << "node: " << mpi.rank << " transmit offset: " << node_transmit_offset << " size: " << node_transmit_size << " : " << n_coord << std::endl;
#endif
    d_vec<int> v_point_id(v_coord_id);
    exa::iota(v_point_id, 0, v_point_id.size(), 0);
    v_coord_nn.resize(n_coord, 0);
    d_vec<int> v_point_nn(n_sample_size, 0);
    std::size_t transmit_cnt = 0;
    int n_iter = 0;

    while (transmit_cnt < n_coord) {
        exa::fill(v_id_chunk, 0, v_id_chunk.size(), -1);
        exa::fill(v_data_chunk, 0, v_data_chunk.size(), std::numeric_limits<float>::max());
        exa::fill(v_point_nn, 0, v_point_nn.size(), 0);
        if (transmit_cnt + node_transmit_size <= n_coord) {
            exa::copy(v_point_id, transmit_cnt, transmit_cnt + node_transmit_size,
                    v_id_chunk, node_transmit_offset);
            exa::copy(v_coord, transmit_cnt * n_dim,
                    (transmit_cnt + node_transmit_size) * n_dim,
                    v_data_chunk, node_transmit_offset * n_dim);
        } else {
            std::size_t size = n_coord - transmit_cnt;
            exa::copy(v_point_id, transmit_cnt, transmit_cnt + size,
                    v_id_chunk, node_transmit_offset);
            exa::copy(v_coord, transmit_cnt * n_dim,
                    (transmit_cnt + size) * n_dim,
                    v_data_chunk, node_transmit_offset * n_dim);
        }
        transmit_cnt += node_transmit_size;
#ifdef DEBUG_ON
        if (mpi.rank == 0)
            std::cout << "transmit iter: " << n_iter << ", elems sent:" << transmit_cnt << std::endl;
#endif
#ifdef MPI_ON
        if (mpi.n_nodes > 1)
            mpi.allGather(v_data_chunk);
#endif
        process_points(v_id_chunk, v_data_chunk);
        ++n_iter;
    }
}


void data_process::process_points(d_vec<int> const &v_point_id, d_vec<float> const &v_point_data) noexcept {
    float _max_float = std::numeric_limits<float>::max();
    auto const it_point = v_point_data.begin();
    auto const it_coord = v_coord.begin();
    auto const _n_dim = n_dim;
    auto const _e2 = e2;
    auto const it_coord_nn = v_coord_nn.begin();
    auto const _n_coord = n_coord;
    auto const _n_point = v_point_id.size();

    /*
    // 55 sec, 27 sec, 15 sec
    exa::for_each(0, n_coord, [=]
#ifdef CUDA_ON
        __device__
#endif
    (auto const &i) -> void {
        int hits = it_coord_nn[i];
        auto const p1 = &it_coord[i * _n_dim];
        for (std::size_t j = 0; j < _n_points; ++j) {
            auto const p2 = &it_point[j * _n_dim];
            // check for empty padding data
            if (p2[0] == _max_float) {
                continue;
            }
            float length = 0;
            for (int d = 0; d < _n_dim; ++d) {
                length += (p1[d] - p2[d]) * (p1[d] - p2[d]);
            }
            if (length <= _e2) {
                ++hits;
            }
        }
        it_coord_nn[i] = hits;
    });
     */


    exa::for_each(0, _n_point, [=]
#ifdef CUDA_ON
            __device__
#endif
    (auto const &j) -> void {
        auto const p2 = &it_point[j * _n_dim];
        // check for empty padding data
        if (p2[0] != _max_float) {
            for (std::size_t i = 0; i < _n_coord; ++i) {
                auto const p1 = &it_coord[i * _n_dim];
                float length = 0;
                for (int d = 0; d < _n_dim; ++d) {
                    length += (p1[d] - p2[d]) * (p1[d] - p2[d]);
                }
                if (length <= _e2) {
//                    it_coord_nn[i] = it_coord_nn[i] + 1;
                    exa::atomic_add(&it_coord_nn[i], 1);
                }
            }
        }

    });



}


