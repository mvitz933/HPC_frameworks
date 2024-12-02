# Compiler and Flags
MPICC = mpicc
MPICXX = mpic++
MPIF90 = mpif90
CFLAGS = -O3 -Wall -I./include
CXXFLAGS = -O3 -Wall -std=c++17 -I./include -Wno-unused-variable
LDFLAGS = -lm -ldl

# Directories
SRC_DIR = src
MAIN_DIR = main
BUILD_DIR = build
OBJ_DIR = $(BUILD_DIR)/obj
BIN_DIR = $(BUILD_DIR)/bin

# Object files for sources
SRC_OBJS = surfaces.o

# Binaries
BINARIES = mpihello-c mpihello-f send_mpi walkmpic walkmpic_reduction jacobi_mpi jacobi_mpi_better
BINARIES += sat1_mpi sat2_mpi sat3_mpi mpi_surface_maximum mpi_surface_max_array mpi_pi_montecarlo mpi_pi_trapezoid1 mpi_pi_trapezoid2

MPI_TARGET = $(BIN_DIR)/mpi_matrix_inversion 
MPI_SRC = $(SRC_DIR)/mpi_matrix_inversion.cpp


# Binary paths
BIN_PATHS = $(patsubst %, $(BIN_DIR)/%, $(BINARIES))

# Default target
all: $(BIN_PATHS) $(MPI_TARGET)

# Rule for creating directories
$(BIN_DIR) $(OBJ_DIR) $(SRC_DIR):
	mkdir -p $@

# Rules for C, C++, and Fortran programs
$(BIN_DIR)/mpihello-c: $(SRC_DIR)/mpi1.cpp | $(BIN_DIR)
	$(MPICXX) $(CXXFLAGS) -o $@ $<

$(BIN_DIR)/mpihello-f: $(SRC_DIR)/mpi1.f90 | $(BIN_DIR)
	$(MPIF90) $(CFLAGS) -o $@ $<

$(BIN_DIR)/send_mpi: $(SRC_DIR)/send_mpi.cpp | $(BIN_DIR)
	$(MPICXX) $(CXXFLAGS) -o $@ $<

$(BIN_DIR)/walkmpic: $(SRC_DIR)/walkmpi.cpp | $(BIN_DIR)
	$(MPICXX) $(CXXFLAGS) -o $@ $<

$(BIN_DIR)/walkmpic_reduction: $(SRC_DIR)/walkmpi_reduction.cpp | $(BIN_DIR)
	$(MPICXX) $(CXXFLAGS) -o $@ $<

$(BIN_DIR)/jacobi_mpi: $(SRC_DIR)/jacobi_mpi.cpp | $(BIN_DIR)
	$(MPICXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

$(BIN_DIR)/jacobi_mpi_better: $(SRC_DIR)/jacobi_mpi_better.cpp | $(BIN_DIR)
	$(MPICXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

$(BIN_DIR)/sat1_mpi: $(SRC_DIR)/sat1.cpp | $(BIN_DIR)
	$(MPICXX) $(CXXFLAGS) -o $@ $<

$(BIN_DIR)/sat2_mpi: $(SRC_DIR)/sat2.cpp | $(BIN_DIR)
	$(MPICXX) $(CXXFLAGS) -o $@ $<

$(BIN_DIR)/sat3_mpi: $(SRC_DIR)/sat3.cpp | $(BIN_DIR)
	$(MPICXX) $(CXXFLAGS) -o $@ $<

$(BIN_DIR)/mpi_surface_maximum: $(SRC_DIR)/mpi_surface_maximum.cpp | $(BIN_DIR)
	$(MPICXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

$(BIN_DIR)/mpi_surface_max_array: $(SRC_DIR)/mpi_surface_max_array.cpp | $(BIN_DIR)
	$(MPICXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

$(BIN_DIR)/mpi_pi_montecarlo: $(SRC_DIR)/mpi_pi_montecarlo.cpp | $(BIN_DIR)
	$(MPICXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

$(BIN_DIR)/mpi_pi_trapezoid1: $(SRC_DIR)/mpi_pi_trapezoid1.cpp | $(BIN_DIR)
	$(MPICXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

$(BIN_DIR)/mpi_pi_trapezoid2: $(SRC_DIR)/mpi_pi_trapezoid2.cpp | $(BIN_DIR)
	$(MPICXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

# Rule for surfaces object file
$(OBJ_DIR)/surfaces.o: $(SRC_DIR)/surfaces.cpp ./include/surfaces.hpp | $(OBJ_DIR)
	$(MPICXX) $(CXXFLAGS) -c $< -o $@

# Rule to build the MPI application
$(MPI_TARGET): $(MPI_SRC)
	$(MPICXX) $(CXXFLAGS) -o $(MPI_TARGET) $(MPI_SRC)

# Cleaning rules
.PHONY: clean deepclean

clean:
	rm -rf $(BUILD_DIR)

deepclean:
	rm -rf $(BUILD_DIR) 
	rm -rf $(LOG_DIR)
