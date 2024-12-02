# Compiler and Flags
CXX = /opt/homebrew/Cellar/gcc/14.2.0_1/bin/aarch64-apple-darwin23-g++-14
CXXFLAGS = -fopenmp -O3 -Wall -Wno-class-memaccess -Wno-uninitialized -std=c++17 -Iinclude -I/opt/homebrew/include/eigen3
LDFLAGS = -Lbuild/lib

# Directories
BUILD_DIR = build
MAIN_DIR = main
SRC_DIR = src
LIB_DIR = $(BUILD_DIR)/lib
BIN_DIR = $(BUILD_DIR)/bin
OBJ_DIR = $(BUILD_DIR)/obj

# Executable categories
BIN_NAMES = matrix_multiply matrix_invert lu_decompose qr_decompose cholesky_decompose \
			linear_solve matrix_eigenvalues

# Source files for methods are
# multiply.cpp invert.cpp lu.cpp qr.cpp cholesky.cpp solve.cpp eigenvalues.cpp

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
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -fPIC -c $< -o $@

# Rule to compile object files from main files
$(OBJ_DIR)/%.o: $(MAIN_DIR)/%.cpp | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -fPIC -c $< -o $@

# Rule to compile library files from .o files
$(LIB_DIR)/lib%.so: $(OBJ_DIR)/%.o $(METHOD_OBJS) | $(LIB_DIR)
	$(CXX) $(CXXFLAGS) -shared -fPIC -o $@ $^

# Rule to build independent executables
$(BIN_DIR)/%: $(OBJ_DIR)/%.o $(METHOD_OBJS) $(SHARED_LIBS) | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $< $(METHOD_OBJS) -L$(LIB_DIR) $(LDFLAGS)

# Rules to build dependent executables
$(BIN_DIR)/linear_solve: $(OBJ_DIR)/linear_solve.o $(OBJ_DIR)/qr.o $(OBJ_DIR)/lu.o $(METHOD_OBJS) | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $< $(METHOD_OBJS)

$(BIN_DIR)/matrix_eigenvalues: $(OBJ_DIR)/matrix_eigenvalues.o $(OBJ_DIR)/qr.o $(OBJ_DIR)/lu.o $(METHOD_OBJS) | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $< $(METHOD_OBJS)


# Clean rule
clean:
	rm -rf $(BUILD_DIR)
