# Compiler and Flags
CXX = nvcc
CXXFLAGS = -arch=sm_86 -O3 -std=c++17 -I./include

# Directories
BUILD_DIR = build
SRC_DIR = src
MAIN_DIR = main
BIN_DIR = $(BUILD_DIR)/bin
OBJ_DIR = $(BUILD_DIR)/obj
LOG_DIR = $(BUILD_DIR)/log

# Executable names
BIN_NAMES = add_vectors_bin multiply_matrices_bin sigmoid_activation_bin linear_layer_and_activation_bin \
multiple_inputs_bin multiple_layers_bin solve_jacobi_naive_bin solve_jacobi_shmem_bin \
bce_bin linear_layer_bin train_linear_bin train_relu_bin train_sigmoid_bin full_training_bin
#invertMatrix 

# Source files (automatically detect .cu files in SRC_DIR)
METHOD_SRCS = $(wildcard $(SRC_DIR)/*.cu)

# Main source files for executables autodetect
MAIN_SRCS = $(wildcard $(MAIN_DIR)/*.cu)

# Object files (based on source files)
METHOD_OBJS = $(patsubst $(SRC_DIR)/%.cu, $(OBJ_DIR)/%.o, $(METHOD_SRCS))

# Object files for main source files
MAIN_OBJS = $(patsubst $(MAIN_DIR)/%.cu, $(OBJ_DIR)/%.o, $(MAIN_SRCS))

# Executable paths
BIN_PATHS = $(patsubst %, $(BIN_DIR)/%, $(BIN_NAMES))

# Default target
all: $(BIN_PATHS)

# Rule for creating directories
$(BIN_DIR) $(OBJ_DIR):
	mkdir -p $@

# Rule to compile object files from .cu files
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Rule to compile object files from .cu files
$(OBJ_DIR)/%.o: $(MAIN_DIR)/%.cu | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Rule to build executables from object files
#$(BIN_DIR)/%: $(OBJ_DIR)/%.o | $(BIN_DIR)
#	$(CXX) $(CXXFLAGS) -o $@ $<

$(BIN_DIR)/add_vectors_bin: $(OBJ_DIR)/add_vectors_main.o $(OBJ_DIR)/add_vectors.o | $(BIN_DIR)
	$(CXX) -DN=25600 $(CXXFLAGS) -o $@ $^ 

$(BIN_DIR)/multiply_matrices_bin: $(OBJ_DIR)/multiply_matrices_main.o $(OBJ_DIR)/multiply_matrices.o | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $^ 

$(BIN_DIR)/sigmoid_activation_bin: $(OBJ_DIR)/sigmoid_activation_main.o $(OBJ_DIR)/sigmoid_activation.o | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $^ 

$(BIN_DIR)/linear_layer_and_activation_bin: $(OBJ_DIR)/linear_layer_and_activation_main.o $(OBJ_DIR)/linear_layer_and_activation.o | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $^ 

$(BIN_DIR)/multiple_inputs_bin: $(OBJ_DIR)/multiple_inputs_main.o $(OBJ_DIR)/multiple_inputs.o | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $^ 

$(BIN_DIR)/multiple_layers_bin: $(OBJ_DIR)/multiple_layers_main.o $(OBJ_DIR)/multiple_layers.o | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $^ 

$(BIN_DIR)/solve_jacobi_naive_bin: $(OBJ_DIR)/solve_jacobi_naive_main.o $(OBJ_DIR)/solve_jacobi_naive.o | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $^ 

$(BIN_DIR)/solve_jacobi_shmem_bin: $(OBJ_DIR)/solve_jacobi_shmem_main.o $(OBJ_DIR)/solve_jacobi_shmem.o | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $^ 


####  THESE ARE FOR THE FULL REPRODUCTION OF NEURAL NET TRAINING WITH CUDA
$(BIN_DIR)/bce_bin: $(OBJ_DIR)/bce_cost.o $(OBJ_DIR)/matrix.o $(OBJ_DIR)/shape.o | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $(MAIN_DIR)/bce_cost.cu $^

$(BIN_DIR)/linear_layer_bin: $(OBJ_DIR)/linear_layer_activate.o $(OBJ_DIR)/matrix.o $(OBJ_DIR)/shape.o $(OBJ_DIR)/bce_cost.o $(OBJ_DIR)/linear_layer.o | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $^

$(BIN_DIR)/train_linear_bin: $(OBJ_DIR)/training_with_linear.o $(OBJ_DIR)/matrix.o $(OBJ_DIR)/shape.o $(OBJ_DIR)/bce_cost.o $(OBJ_DIR)/linear_layer.o | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $^

$(BIN_DIR)/train_relu_bin: $(OBJ_DIR)/training_with_relu.o $(OBJ_DIR)/matrix.o $(OBJ_DIR)/shape.o $(OBJ_DIR)/bce_cost.o $(OBJ_DIR)/relu_activation.o | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $^

$(BIN_DIR)/train_sigmoid_bin: $(OBJ_DIR)/training_with_sigmoid.o $(OBJ_DIR)/matrix.o $(OBJ_DIR)/shape.o $(OBJ_DIR)/bce_cost.o $(OBJ_DIR)/sigmoid_activation_v2.o | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $^

$(BIN_DIR)/full_training_bin: $(OBJ_DIR)/full_training.o $(OBJ_DIR)/matrix.o $(OBJ_DIR)/shape.o $(OBJ_DIR)/bce_cost.o $(OBJ_DIR)/sigmoid_activation_v2.o $(OBJ_DIR)/coordinates_dataset.o $(OBJ_DIR)/neural_network.o $(OBJ_DIR)/linear_layer.o $(OBJ_DIR)/relu_activation.o | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $^

# Cleaning rules
.PHONY: clean deepclean

clean:
	rm -rf $(BUILD_DIR)

deepclean:
	rm -rf $(BUILD_DIR) 
	rm -rf $(LOG_DIR)


#nvcc -arch=sm_86 -DN=25601  -I./include main/add_vectors_main.cu src/add_vectors.cu -o build/bin/add_vectors_executable

#nvcc -arch=sm_86 -I./include main/bce_cost.cu  src/matrix.cu src/shape.cu src/bce_cost.cu -o build/bin/bce
#nvcc -arch=sm_86 -I./include main/linear_layer_activate.cu src/matrix.cu src/shape.cu src/bce_cost.cu src/linear_layer.cu -o build/bin/linear_layer
#nvcc -arch=sm_86 -I./include main/training_with_linear.cu src/matrix.cu src/shape.cu src/bce_cost.cu src/linear_layer.cu -o build/bin/train_linear
#nvcc -arch=sm_86 -I./include main/training_with_relu.cu src/matrix.cu src/shape.cu src/bce_cost.cu src/relu_activation.cu -o build/bin/train_relu
#nvcc -arch=sm_86 -I./include main/training_with_sigmoid.cu src/matrix.cu src/shape.cu src/bce_cost.cu src/sigmoid_activation_v2.cu -o build/bin/train_sigmoid
#nvcc -arch=sm_86 -I./include main/full_training.cu src/matrix.cu src/shape.cu src/bce_cost.cu src/sigmoid_activation_v2.cu src/coordinates_dataset.cu src/neural_network.cu src/linear_layer.cu src/relu_activation.cu -o build/bin/full_training
#nvcc -arch=sm_86 -I./include main/linear_layer_activate.cu src/matrix.cu src/shape.cu src/bce_cost.cu src/linear_layer.cu -o build/bin/linear_layer