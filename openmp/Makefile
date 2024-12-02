# Compiler and Flags
CXX = pgc++
CXXFLAGS = -fopenmp -O3 -Wall -std=c++17 -I./include
LDFLAGS = -L./build/lib

# Directories
BUILD_DIR = build
SRC_DIR = src
MAIN_DIR = main
LIB_DIR = $(BUILD_DIR)/lib
BIN_DIR = $(BUILD_DIR)/bin
OBJ_DIR = $(BUILD_DIR)/obj
LOG_DIR = $(BUILD_DIR)/log

# Executable categories
BIN_NAMES = matrix_multiply matrix_invert lu_decompose qr_decompose \
		cholesky_decompose linear_solve matrix_eigenvalues  \
		solve_poisson solve_laplace2d solve_heatedplate solve_poisson_tiled

# TODO: incorporate the Gauss Seidel Method

# Source files for methods are
# multiply.cpp 
# invert.cpp 
# lu.cpp 
# qr.cpp 
# cholesky.cpp 
# solve.cpp 
# eigenvalues.cpp
# jacobi.cpp
# gaussseidel.cpp
#

# Main source files for methods autodetect
METHOD_SRCS = $(wildcard $(SRC_DIR)/*.cpp)

# Main source files for executables autodetect
MAIN_SRCS = $(wildcard $(MAIN_DIR)/*.cpp)

# Object files for main source files
MAIN_OBJS = $(patsubst $(MAIN_DIR)/%.cpp, $(OBJ_DIR)/%.o, $(MAIN_SRCS))

# Object files for methods
METHOD_OBJS = $(patsubst $(SRC_DIR)/%.cpp, $(OBJ_DIR)/%.o, $(METHOD_SRCS))

# Shared libraries
SHARED_LIBS = $(patsubst $(SRC_DIR)/%.cpp, $(LIB_DIR)/lib%.so, $(METHOD_SRCS))

# Executable paths
BIN_PATHS = $(patsubst %, $(BIN_DIR)/%, $(BIN_NAMES))

# Default target
all: $(BIN_PATHS) 

# Rule for creating directories
$(BIN_DIR) $(LIB_DIR) $(OBJ_DIR):
	mkdir -p $@

# Rule to compile object files from source files
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -fPIC -c $< -o $@

# Rule to compile object files from main files
$(OBJ_DIR)/%.o: $(MAIN_DIR)/%.cpp | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -fPIC -c $< -o $@

$(LIB_DIR)/lib%.so: $(OBJ_DIR)/%.o $(METHOD_OBJS) | $(LIB_DIR)
	$(CXX) $(CXXFLAGS) -shared -fPIC -o $@ $^

# Rule to build binaries
$(BIN_DIR)/%: $(OBJ_DIR)/%.o $(METHOD_OBJS) $(SHARED_LIBS) | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $< $(METHOD_OBJS) -L$(LIB_DIR) $(LDFLAGS)

# Rule to build dependent executables
$(BIN_DIR)/linear_solve: $(OBJ_DIR)/linear_solve.o $(OBJ_DIR)/qr.o $(OBJ_DIR)/lu.o $(METHOD_OBJS) | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $< $(METHOD_OBJS)

$(BIN_DIR)/matrix_eigenvalues: $(OBJ_DIR)/matrix_eigenvalues.o $(OBJ_DIR)/qr.o $(OBJ_DIR)/lu.o $(METHOD_OBJS) | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $< $(METHOD_OBJS)

$(BIN_DIR)/solve_poisson: $(OBJ_DIR)/solve_poisson.o $(OBJ_DIR)/jacobi.o
	$(CXX) $(CXXFLAGS) -o $@ $< $(OBJ_DIR)/jacobi.o

$(BIN_DIR)/solve_poisson_tiled: $(OBJ_DIR)/solve_poisson_tiled.o $(OBJ_DIR)/jacobi.o
	$(CXX) $(CXXFLAGS) -o $@ $< $(OBJ_DIR)/jacobi.o

$(BIN_DIR)/solve_laplace2d: $(OBJ_DIR)/solve_laplace2d.o $(OBJ_DIR)/jacobi.o
	$(CXX) $(CXXFLAGS) -o $@ $< $(OBJ_DIR)/jacobi.o

$(BIN_DIR)/solve_heatedplate: $(OBJ_DIR)/solve_heatedplate.o $(OBJ_DIR)/jacobi.o
	$(CXX) $(CXXFLAGS) -o $@ $< $(OBJ_DIR)/jacobi.o

# TO DO: 
#  -- audit the code in these files and 
#  -- integrate them into SRC and TARGET

laplace2d_f90_omp: laplace2d.f90
	$(F90) -o $@ $(OMPFLAGS) $<

heatedplateF: heatedplate.f90
	$(F90) -o $@ $<

# Cleaning rules
.PHONY: clean deepclean

clean:
	rm -rf $(BUILD_DIR)

deepclean:
	rm -rf $(BUILD_DIR) 
	rm -rf $(LOG_DIR)