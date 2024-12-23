#pragma once
#include "nn_layer.hpp"


class LinearLayer : public NNLayer {
private:
 const float weights_init_threshold = 0.01;

 Matrix W;
 Matrix b;

 Matrix Z;
 Matrix A;
 Matrix dA;

 void initializeBiasWithZeros();
 void initializeWeightsRandomly();

 void computeAndStoreBackpropError(Matrix& dZ);
 void computeAndStoreLayerOutput(Matrix& A);
 void updateWeights(Matrix& dZ, float learning_rate);
 void updateBias(Matrix& dZ, float learning_rate);

public:
 LinearLayer(std::string name, Shape W_shape);
 ~LinearLayer();

 Matrix& forward(Matrix& A);
 Matrix& backprop(Matrix& dZ, float learning_rate = 0.01);

 int getXDim() const;
 int getYDim() const;

 Matrix& getWeightsMatrix();
    Matrix& getBiasVector();
};
