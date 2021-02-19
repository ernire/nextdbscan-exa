OUT=nextdbscan-exa

mpi:
	rm -f $(OUT)-mpi
	mpicxx -O3 -DMPI_ON=1 -DOMP_ON=1 -DHDF5_ON=1 -fopenmp -lhdf5 -std=c++14 main.cpp nextdbscan.cpp data_process.cpp -o $(OUT)-mpi
mpi-debug:
	rm -f $(OUT)-mpi-debug
	mpicxx -O3 -DMPI_ON=1 -DDEBUG_ON=1 -DOMP_ON=1 -DHDF5_ON=1 -fopenmp -lhdf5 -std=c++14 main.cpp nextdbscan.cpp data_process.cpp -o $(OUT)-mpi
omp:
	rm -f $(OUT)-omp
	icc -O3 -DOMP_ON=1 -fopenmp -std=c++14 main.cpp nextdbscan.cpp data_process.cpp -o $(OUT)-omp
omp-debug:
	rm -f $(OUT)-omp-debug
	icc -O3 -DOMP_ON=1 -DDEBUG_ON=1 -fopenmp -std=c++14 main.cpp nextdbscan.cpp data_process.cpp -o $(OUT)-omp
omp-bms:
	rm -f $(OUT)-bms
	icc -O3 -DOMP_ON=1 -DDEBUG_ON=1 -fopenmp -std=c++14 exa_bms.cpp -o $(OUT)-bms
cu:
	rm -f $(OUT)-cu
	nvcc -O3 -std=c++14 -x cu -Xcompiler -DDEBUG_ON=1 -DCUDA_ON=1 --expt-extended-lambda main.cpp data_process.cpp nextdbscan.cpp -o $(OUT)-cu
cu-mpi:
	rm -f $(OUT)-cu.o
	rm -f $(OUT)-cu-mpi
	rm -f ble-cu.o
	nvcc -O3 -std=c++14 -x cu -Xcompiler -DCUDA_ON=1 -DMPI_ON=1 --expt-extended-lambda -c data_process.cpp -o $(OUT)-cu.o
	nvcc -O3 -std=c++14 -x cu -Xcompiler -DCUDA_ON=1 -DMPI_ON=1 --expt-extended-lambda -c nextdbscan.cpp -o ble-cu.o
	mpicxx -O3 -DMPI_ON=1 -DCUDA_ON=1 -DHDF5_ON=1 -lhdf5 -std=c++14 -lcudart main.cpp $(OUT)-cu.o ble-cu.o -o $(OUT)-cu-mpi
cu-mpi-debug:
	rm -f $(OUT)-cu.o
	rm -f $(OUT)-cu-mpi
	nvcc -O3 -Xcompiler -std=c++14 -DDEBUG_ON=1 -DCUDA_ON=1 -DMPI_ON=1 --expt-extended-lambda -c nc_tree.cu -o $(OUT)-cu.o
	mpicxx -O3 -DDEBUG_ON=1 -DMPI_ON=1 -DCUDA_ON=1 -DHDF5_ON=1 -lhdf5 -std=c++14 -lcudart main.cpp nextdbscan.cpp data_process.cpp $(OUT)-cu.o -o $(OUT)-cu-mpi

	
