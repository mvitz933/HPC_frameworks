#pragma once
#include "matrix.hpp"

class BCECost {
public:
 float cost(Matrix predictions, Matrix target);
 Matrix dCost(Matrix predictions, Matrix target, Matrix dY);
};
