#include "MatrixOps.h"
#include <algorithm>
#include <vector>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <random>


Eigen::SparseMatrix<double> sparsifyMatrix(const Eigen::MatrixXd& matrix) {
    std::vector<double> elements;
    for (int i = 0; i < matrix.rows(); ++i) {
        for (int j = 0; j < matrix.cols(); ++j) {
            elements.push_back(matrix(i, j));
        }
    }

    std::nth_element(elements.begin(), elements.begin() + elements.size() * 0.1, elements.end());
    double threshold = elements[elements.size() * 0.1];

    Eigen::SparseMatrix<double> sparseMatrix(matrix.rows(), matrix.cols());
    for (int i = 0; i < matrix.rows(); ++i) {
        for (int j = 0; j < matrix.cols(); ++j) {
            if (matrix(i, j) >= threshold) {
                sparseMatrix.insert(i, j) = matrix(i, j);
            }
        }
    }
    sparseMatrix.makeCompressed();
    return sparseMatrix;
}

Eigen::VectorXd generateDiscreteDistribution(int N) {
    // Random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
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