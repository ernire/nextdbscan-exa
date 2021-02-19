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
#ifdef MPI_ON
#include <mpi.h>
#endif
#ifdef OMP_ON
#include <omp.h>
#endif
#include "nextdbscan.h"

void usage() {
    std::cout << "NextDBSCAN compiled for OpenMP";
#ifdef MPI_ON
    std::cout << ", MPI";
#endif
#ifdef HDF5_ON
    std::cout << ", HDF5";
#endif
#ifdef CUDA_ON
    std::cout << ", CUDA (V100)";
#endif
    std::cout << std::endl << std::endl;
    std::cout << "Usage: [executable] -m minPoints -e epsilon -t threads [input file]" << std::endl;
    std::cout << "    -m minPoints : DBSCAN parameter, minimum number of points required to form a cluster, postive integer, required" << std::endl;
    std::cout << "    -e epsilon   : DBSCAN parameter, maximum neighborhood search radius for cluster, positive floating point, required" << std::endl;
    std::cout << "    -t threads   : Processing parameter, the number of threads to use, positive integer, defaults to 1" << std::endl;
    std::cout << "    -a 0|1       : For faster results this option will produce an upper-bound approximation of the final number of clusters. Noise and cores, however, will be identical to the non-approximate case" << std::endl;
    std::cout << "    -o output    : Output file containing the cluster ids in the same order as the input" << std::endl;
    std::cout << "    -h help      : Show this help message" << std::endl << std::endl;
    std::cout << "Supported Input Types:" << std::endl;

    std::cout << ".csv: Text file with one sample/point per line and features/dimensions separated by a space delimiter, i.e. ' '" << std::endl;
    std::cout << ".bin: Custom binary format for faster file reads. Use cvs2bin executable to transform csv into bin files." << std::endl;
#ifdef HDF5_ON
    std::cout << ".hdf5: The HDF5 parallel file format" << std::endl;
#endif
}

int main(int argc, char** argv) {
    char *p;
    int m = -1;
    float e = -1;
    int t = 1;
    int a = 0;
    std::string input_file;
    std::string output_file;

    for (int i = 1; i < argc; i += 2) {
        std::string str(argv[i]);
        if (str == "-m") {
            m = std::stoi(argv[i+1]);
        } else if (str == "-e") {
            e = std::strtof(argv[i+1], &p);
        } else if (str == "-t") {
            t = std::stoi(argv[i+1]);
        } else if (str == "-a") {
            a = std::stoi(argv[i+1]);
        }
    }
    input_file = argv[argc-1];

    if (m == -1 || e == -1) {
        std::cout << "Input Error: Please specify the m and e parameters" << std::endl << std::endl;
        usage();
        std::exit(EXIT_FAILURE);
    }
#ifdef MPI_ON
    MPI_Init(&argc, &argv);
#endif
    auto mpi = magmaMPI::build();
#ifdef OMP_ON
    omp_set_num_threads(t);
#endif
    if (mpi.rank == 0) {
        std::cout << "Starting NextDBSCAN with file: " << input_file << " m: " << m << " e: " << e << " t: " << t
                  << " a: " << a << std::endl;
    }
    auto results = nextdbscan::start(m, e, t, input_file, mpi, a == 1);

    if (mpi.rank == 0) {
        std::cout << std::endl;
        if (a == 0) {
            std::cout << "Estimated Clusters: " << results.clusters << std::endl;
        } else if (a == 1) {
            std::cout << "Upper-bound Approximate Clusters: " << results.clusters << std::endl;
        }
        std::cout << "Core Points: " << results.core_count << std::endl;
        std::cout << "Noise Points: " << results.noise << std::endl;

        /*
        if (output_file.length() > 0) {
            std::cout << "Writing output to " << output_file << std::endl;
            std::ofstream os(output_file);
            // TODO
            for (int i = 0; i < results.n; ++i) {
                os << results.point_clusters[i] << std::endl;
            }
//            for (auto &c : results.point_clusters) {
//                os << c << '\n';
//            }
            os.flush();
            os.close();
            std::cout << "Done!" << std::endl;
        }
         */
    }
#ifdef MPI_ON
    MPI_Finalize();
#endif
    return 0;
}
