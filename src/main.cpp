#include "BFGS.h"
#include "MatrixOps.h"
#include <Eigen/Dense>
#include <iostream>
#include <cmath>

int main() {
    int n = 100, m = 150;
    double eta = 0.1;
    Eigen::VectorXd x1_dis = generateDiscreteDistribution(n); // distribution of x1
    Eigen::VectorXd x2_dis = generateDiscreteDistribution(m); // distribution of x2
    Eigen::VectorXd x1 = Eigen::VectorXd::Random(n).array() * 5;
    Eigen::VectorXd x2 = Eigen::VectorXd::Random(m).array() * 5;
    Eigen::MatrixXd M(n, m);
    Eigen::VectorXd initialPoint = Eigen::VectorXd::Zero(n + m);

    // Compute the cost matrix
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            M(i, j) = (x1(i) - x2(j)) * (x1(i) - x2(j));
        }
    }

    auto entropic_regularized_OT = [x1_dis, x2_dis, eta, M](const Eigen::VectorXd& input) -> double {
        int n = x1_dis.size(), m = x2_dis.size();
        Eigen::VectorXd alpha = input.head(n), beta = input.tail(m);
        double term1 = alpha.dot(x1_dis) + beta.dot(x2_dis);
        double term2 = 0.0;

        for (int i = 0; i < n; ++i)
            for (int j = 0; j < m; ++j)
                term2 += std::exp((alpha(i) + beta(j) - M(i, j)) / eta);

        return eta * term2 - term1;
    };

    auto entropic_regularized_OT_grad = [x1_dis, x2_dis, eta, M](const Eigen::VectorXd& input) -> Eigen::VectorXd {
        int n = x1_dis.size(), m = x2_dis.size();
        Eigen::VectorXd alpha = input.head(n), beta = input.tail(m);
        Eigen::VectorXd grad(n + m);

        grad.setZero();
        grad.head(n) = -x1_dis;
        grad.tail(m) = -x2_dis;

        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < m; ++j) {
                double exp_term = std::exp((alpha(i) + beta(j) - M(i, j)) / eta);
                grad(i) += exp_term;
                grad(j + n) += exp_term;
            }
        }
        return grad;
    };

    BFGS optimizer(entropic_regularized_OT, entropic_regularized_OT_grad);
    Eigen::VectorXd result;
    std::cout << "---------- B_optimize ----------" << std::endl;
    result = optimizer.B_optimize(initialPoint);
    std::cout << "Optimized parameters: " << result.transpose() << std::endl;
    std::cout << "Minimum value: " << entropic_regularized_OT(result) << std::endl;
    std::cout << "--------------------------------\n\n" << std::endl;

    std::cout << "---------- H_optimize ----------" << std::endl;
    result = optimizer.H_optimize(initialPoint);
    std::cout << "Optimized parameters: " << result.transpose() << std::endl;
    std::cout << "Minimum value: " << entropic_regularized_OT(result) << std::endl;
    std::cout << "--------------------------------\n\n" << std::endl;

    std::cout << "---------- sparseB_optimize ----------" << std::endl;
    result = optimizer.sparseB_optimize(
        initialPoint, 
        x1_dis, x2_dis,
        eta, M);
    std::cout << "Optimized parameters: " << result.transpose() << std::endl;
    std::cout << "Minimum value: " << entropic_regularized_OT(result) << std::endl;
    std::cout << "--------------------------------\n\n" << std::endl;
    
    return 0;
}

