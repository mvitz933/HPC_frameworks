#pragma once

#include <iostream>
#include <cmath>
#include <mpi.h>

constexpr double PI = 3.141592653589793;
constexpr double mu1 = 1.4142135623730951; // Precomputed sqrt(2.0)
constexpr double mu2 = 1.7724538509055159; // Precomputed sqrt(PI)
constexpr double sigma1 = 3.1;
constexpr double sigma2 = 1.4;


// Function to calculate `a`
double calculate_a(double x) {
    return std::pow(x - mu1, 2) / (2.0 * std::pow(sigma1, 2));
}

// Function to calculate `b`
double calculate_b(double y) {
    return std::pow(y - mu2, 2) / (2.0 * std::pow(sigma2, 2));
}

// Function to calculate `z1`
double calculate_z1(double x, double y) {
    return 0.1 * std::sin(x) * std::sin(x * y);
}

// Function to calculate `z2`
double calculate_z2(double a, double b) {
    return std::exp(-1 * (a + b)) / (sigma1 * sigma2 * std::sqrt(2 * PI));
}

