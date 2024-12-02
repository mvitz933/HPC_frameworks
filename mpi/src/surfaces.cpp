#include "surfaces.hpp"

int main(int argc, char** argv) {
    // Example usage
    double x = 1.0;
    double y = 2.0;

    double a = calculate_a(x);
    double b = calculate_b(y);
    double z1 = calculate_z1(x, y);
    double z2 = calculate_z2(a, b);

    std::cout << "a: " << a << "\n"
              << "b: " << b << "\n"
              << "z1: " << z1 << "\n"
              << "z2: " << z2 << "\n";

    return 0;
}