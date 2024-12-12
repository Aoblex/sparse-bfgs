#include "MatrixOps.h"
#include <algorithm>
#include <vector>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <random>

Eigen::VectorXd generateDiscreteDistribution(int N) {
    // Random number generator with fixed seed
    std::mt19937 gen(42); // Fixed seed for reproducibility
    std::uniform_real_distribution<> dis(0.0, 1.0);

    // Generate N random numbers
    Eigen::VectorXd randomVec(N);
    for (int i = 0; i < N; ++i) {
        randomVec(i) = dis(gen);
    }

    // Normalize the vector to sum to 1
    randomVec /= randomVec.sum();

    return randomVec;
}