#include "BFGS.h"
#include "MatrixOps.h"
#include <Eigen/Dense>
#include <iostream>
#include <cmath>

int main() {
    std::srand(42); // Set random seed for reproducibility

    int n = 100, m = 80;
    double eta = 0.1;
    Eigen::VectorXd a = generateDiscreteDistribution(n);
    Eigen::VectorXd b = generateDiscreteDistribution(m);
    Eigen::VectorXd x1 = Eigen::VectorXd::Random(n).array() * 5;
    Eigen::VectorXd x2 = Eigen::VectorXd::Random(m).array() * 5;
    Eigen::VectorXd initialPoint = Eigen::VectorXd::Random(n + m);

    auto entropic_regularized_OT = [n, m, a, b, x1, x2, eta](const Eigen::VectorXd& input) -> double {
        Eigen::VectorXd alpha = input.head(n), beta = input.tail(m);
        double result = 0.0;

        result -= alpha.dot(a) + beta.dot(b);

        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < m; ++j) {
                double exp_term = std::exp((alpha(i) + beta(j) - pow(x1(i) - x2(j), 2)) / eta);
                result += eta * exp_term;
            }
        }
        return result;
    };

    auto entropic_regularized_OT_grad = [n, m, a, b, x1, x2, eta](const Eigen::VectorXd& input) -> Eigen::VectorXd {
        Eigen::VectorXd alpha = input.head(n), beta = input.tail(m);
        Eigen::VectorXd grad(n + m);
        grad.setZero();

        grad.head(n) = -a;
        grad.tail(m) = -b;

        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < m; ++j) {
                double exp_term = std::exp((alpha(i) + beta(j) - pow(x1(i) - x2(j), 2)) / eta);
                grad(i) += exp_term;
                grad(j + n) += exp_term;
            }
        }
        return grad;
    };

    BFGS optimizer(entropic_regularized_OT, entropic_regularized_OT_grad);
    Eigen::VectorXd result = optimizer.B_optimize(initialPoint);

    std::cout << "Optimized parameters: " << result.transpose() << std::endl;
    std::cout << "Minimum value: " << entropic_regularized_OT(result) << std::endl;

    return 0;
}

