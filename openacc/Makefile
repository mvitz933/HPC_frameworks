# Compiler and Flags
CXX = pgc++
#CXXFLAGS = -acc=multicore -O3 -Wall -std=c++17 -Iinclude
CXXFLAGS = -acc -gpu=cc86,mem:managed -Kieee -O3 -Wall -std=c++17 -I./include
#CXX = /opt/homebrew/Cellar/gcc/14.2.0_1/bin/aarch64-apple-darwin23-g++-14
#CXXFLAGS = -fopenacc -foffload=disable -O3 -Wall -Wno-class-memaccess -Wno-uninitialized -std=c++17 -Iinclude -I/opt/homebrew/include/eigen3
#NVCCFLAGS = -gencode arch=compute_86,code=sm_86 -O3 -Xcompiler -fPIC
LDFLAGS = -L./build/lib

# Directories
BUILD_DIR = build
MAIN_DIR = main
SRC_DIR = src
LIB_DIR = $(BUILD_DIR)/lib
BIN_DIR = $(BUILD_DIR)/bin
OBJ_DIR = $(BUILD_DIR)/obj
LOG_DIR = outputs/logs

# Executable categories
BIN_NAMES = matrix_multiply matrix_invert lu_decompose qr_decompose \
		cholesky_decompose linear_solve matrix_eigenvalues  \
		solve_poisson solve_laplace2d solve_poisson_tiled solve_heatedplate 

# Source files for methods are in the header files and the files below are just for reference
# multiply.txt
# invert.txt
# lu.txt 
# qr.txt 
# cholesky.txt 
# solve.txt
# eigenvalues.txt
# jacobi.txt
# gaussseidel.txt
#

# Main source files for methods autodetect
METHOD_SRCS = $(wildcard $(SRC_DIR)/*.cpp)

# Main source files for executables autodetect
MAIN_SRCS = $(wildcard $(MAIN_DIR)/*.cpp)

# Object files for main source files
MAIN_OBJS = $(patsubst main/%.cpp, $(OBJ_DIR)/%.o, $(MAIN_SRCS))

# Object files for methods
METHOD_OBJS = $(patsubst src/%.cpp, $(OBJ_DIR)/%.o, $(METHOD_SRCS))

# Shared libraries
SHARED_LIBS = $(patsubst src/%.cpp, $(LIB_DIR)/lib%.so, $(METHOD_SRCS))

# Executable paths
BIN_PATHS = $(patsubst %, $(BIN_DIR)/%, $(BIN_NAMES))

# Default target
all: $(BIN_PATHS)

# Rule for creating directories
$(BIN_DIR) $(LIB_DIR) $(OBJ_DIR) $(MAIN_DIR):
	mkdir -p $@

# Rule to compile object files from source files
$(OBJ_DIR)/%.o: src/%.cpp | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -fPIC -c $< -o $@

# Rule to compile object files from main files
$(OBJ_DIR)/%.o: $(MAIN_DIR)/%.cpp | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -fPIC -c $< -o $@

$(LIB_DIR)/lib%.so: $(OBJ_DIR)/%.o $(METHOD_OBJS) | $(LIB_DIR)
	$(CXX) $(CXXFLAGS) -shared -fPIC -o $@ $^

# Rule to build independent executables
$(BIN_DIR)/%: $(OBJ_DIR)/%.o $(METHOD_OBJS) $(SHARED_LIBS) | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $< $(METHOD_OBJS) -L$(LIB_DIR) $(LDFLAGS)

# Rule to build dependent executables
$(BIN_DIR)/linear_solve: $(OBJ_DIR)/linear_solve.o $(METHOD_OBJS) | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $< $(METHOD_OBJS)

$(BIN_DIR)/matrix_eigenvalues: $(OBJ_DIR)/matrix_eigenvalues.o $(METHOD_OBJS) | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $< $(METHOD_OBJS)

$(BIN_DIR)/solve_poisson: $(OBJ_DIR)/solve_poisson.o 
	$(CXX) $(CXXFLAGS) -o $@ $< 

$(BIN_DIR)/solve_poisson_tiled: $(OBJ_DIR)/solve_poisson_tiled.o 
	$(CXX) $(CXXFLAGS) -o $@ $< 

$(BIN_DIR)/solve_laplace2d: $(OBJ_DIR)/solve_laplace2d.o 
	$(CXX) $(CXXFLAGS) -o $@ $< 

$(BIN_DIR)/solve_heatedplate: $(OBJ_DIR)/solve_heatedplate.o 
	$(CXX) $(CXXFLAGS) -o $@ $< 


# TO DO: 
#  -- audit the code in these files and 
#  -- integrate them into SRC and TARGET

laplace2d_f90_acc: laplace2d.f90
	$(F90) -o $@ $<

# Cleaning rules
.PHONY: clean deepclean

clean:
	rm -rf $(BUILD_DIR)

deepclean:
	rm -rf $(BUILD_DIR) 
	rm -rf $(LOG_DIR)
