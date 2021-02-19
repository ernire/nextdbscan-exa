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

#ifndef NEXTDBSCAN20_MAGMA_INPUT_H
#define NEXTDBSCAN20_MAGMA_INPUT_H

#include <string>
#include <iostream>
#include <sstream>
#include <fstream>
#include <iterator>
#ifdef HDF5_ON
#include <hdf5.h>
#endif

int get_block_size(int block_index, int number_of_samples, int number_of_blocks) noexcept {
    int block = (number_of_samples / number_of_blocks);
    int reserve = number_of_samples % number_of_blocks;
    //    Some processes will need one more sample if the data size does not fit completely
    if (reserve > 0 && block_index < reserve) {
        return block + 1;
    }
    return block;
}

int get_block_offset(int block_index, int number_of_samples, int number_of_blocks) noexcept {
    int offset = 0;
    for (int i = 0; i < block_index; i++) {
        offset += get_block_size(i, number_of_samples, number_of_blocks);
    }
    return offset;
}

namespace magma_input {

    void count_lines_and_dimensions(std::string const &in_file, int &lines, int &dim) noexcept {
        std::ifstream is(in_file);
        std::string line, buf;
        int cnt = 0;
        dim = 0;
        while (std::getline(is, line)) {
            if (dim == 0) {
                std::istringstream iss(line);
                std::vector<std::string> results(std::istream_iterator<std::string>{iss},
                        std::istream_iterator<std::string>());
                dim = results.size();
            }
            ++cnt;
        }
        lines = cnt;
        is.close();
    }

    inline bool is_equal(const std::string &in_file, const std::string &s_cmp) noexcept {
        return in_file.compare(in_file.size() - s_cmp.size(), s_cmp.size(), s_cmp) == 0;
    }

    void read_input_hdf5(const std::string &in_file, h_vec<float> &v_points, int &n_coord, int &n_dim,
            int const n_nodes, int i_node) noexcept {
#ifdef HDF5_ON
//        std::cout << "HDF5 Reading Start: " << std::endl;


        // TODO H5F_ACC_RDONLY ?
        hid_t file = H5Fopen(in_file.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
//        hid_t dset = H5Dopen1(file, "DBSCAN");
//        hid_t dset = H5Dopen1(file, i_node < (n_nodes / 2)? "xyz_1" : "xyz_2");
        hid_t dset = H5Dopen1(file, "xyz");
        hid_t fileSpace= H5Dget_space(dset);

        // Read dataset size and calculate chunk size
        hsize_t count[2];
        H5Sget_simple_extent_dims(fileSpace, count,NULL);
        n_coord = count[0];
        n_dim = count[1];
//        std::cout << "HDF5 total size: " << n_coord << std::endl;

//        hsize_t block_size =  get_block_size(i_node >= (n_nodes / 2)? i_node - (n_nodes / 2): i_node, n_coord, n_nodes / 2);
//        hsize_t block_offset =  get_block_offset(i_node >= (n_nodes / 2)? i_node - (n_nodes / 2): i_node, n_coord, n_nodes / 2);
        hsize_t block_size =  get_block_size(i_node, n_coord, n_nodes);
        hsize_t block_offset =  get_block_offset(i_node, n_coord, n_nodes);
//        std::cout << "i_node: " << i_node << " offset:size " << block_offset << " : " <<  block_size << std::endl;
        hsize_t offset[2] = {block_offset, 0};
        count[0] = block_size;
        v_points.resize(block_size * n_dim);

        hid_t memSpace = H5Screate_simple(2, count, NULL);
        H5Sselect_hyperslab(fileSpace,H5S_SELECT_SET,offset, NULL, count, NULL);
        H5Dread(dset, H5T_IEEE_F32LE, memSpace, fileSpace,H5P_DEFAULT, &v_points[0]);

        H5Dclose(dset);
        H5Fclose(file);

#endif
    }

    void read_input_bin(const std::string &in_file, h_vec<float> &v_points, int &n_coord, int &n_dim,
            int const n_nodes, int const i_node) noexcept {
        std::ifstream ifs(in_file, std::ios::in | std::ifstream::binary);
        ifs.read((char *) &n_coord, sizeof(int));
        ifs.read((char *) &n_dim, sizeof(int));
        auto size = get_block_size(i_node, n_coord, n_nodes);
        auto offset = get_block_offset(i_node, n_coord, n_nodes);
        auto feature_offset = 2 * sizeof(int) + (offset * n_dim * sizeof(float));
        v_points.resize(size * n_dim);
        ifs.seekg(feature_offset, std::istream::beg);
        ifs.read((char *) &v_points[0], size * n_dim * sizeof(float));
        ifs.close();
    }

    void read_input_csv(const std::string &in_file, h_vec<float> &v_points, long const n_dim) noexcept {
        std::ifstream is(in_file, std::ifstream::in);
        std::string line, buf;
        std::stringstream ss;
        int index = 0;
        while (std::getline(is, line)) {
            ss.str(std::string());
            ss.clear();
            ss << line;
            for (int j = 0; j < n_dim; j++) {
                ss >> buf;
                v_points[index++] = static_cast<float>(atof(buf.c_str()));
            }
        }
        is.close();
    }

    void read_input(const std::string &in_file, h_vec<float> &v_input, int &n, int &n_dim,
            int const n_nodes, int const i_node) noexcept {
        std::string s_cmp_bin = ".bin";
        std::string s_cmp_hdf5_1 = ".h5";
        std::string s_cmp_hdf5_2 = ".hdf5";
        std::string s_cmp_csv = ".csv";

        if (is_equal(in_file, s_cmp_bin)) {
            read_input_bin(in_file, v_input, n, n_dim, n_nodes, i_node);
        } else if (is_equal(in_file, s_cmp_hdf5_1) || is_equal(in_file, s_cmp_hdf5_2)) {
#ifdef HDF5_ON
            read_input_hdf5(in_file, v_input, n, n_dim, n_nodes, i_node);
#endif
#ifndef HDF5_ON
            std::cerr << "Error: HDF5 is not supported by this executable." << std::endl;
            exit(-1);
#endif
        } else if (is_equal(in_file, s_cmp_csv)) {
            count_lines_and_dimensions(in_file, n, n_dim);
            v_input.resize(n * n_dim);
            std::cout << "WARNING: USING SLOW CSV I/O." << std::endl;
            read_input_csv(in_file, v_input, n_dim);
        }
    }
}

#endif //NEXTDBSCAN20_MAGMA_INPUT_H
