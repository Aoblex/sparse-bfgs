#include "BFGS.h"
#include <Eigen/Dense>
#include <iostream>

int main() {
    auto rosenbrock = [](const Eigen::VectorXd& x) -> double {
        double a = 1.0, b = 100.0;
        double sum = 0.0;
        for (int i = 0; i < x.size() - 1; ++i) {
            double x1 = x(i), x2 = x(i + 1);
            sum += std::pow(a - x1, 2) + b * std::pow(x2 - x1 * x1, 2);
        }
        return sum;
    };

    auto rosenbrockGrad = [](const Eigen::VectorXd& x) -> Eigen::VectorXd {
        double a = 1.0, b = 100.0;
        Eigen::VectorXd grad(x.size());
        grad.setZero();
        for (int i = 0; i < x.size() - 1; ++i) {
            double x1 = x(i), x2 = x(i + 1);
            grad(i) += -2 * (a - x1) - 4 * b * (x2 - x1 * x1) * x1;
            grad(i + 1) += 2 * b * (x2 - x1 * x1);
        }
        return grad;
    };

    int dimension = 200;

    Eigen::VectorXd initialPoint(dimension);
    std::srand(42); // Set random seed for reproducibility
    initialPoint.setRandom(); // Initialize with random values

    BFGS optimizer(rosenbrock, rosenbrockGrad);
    Eigen::VectorXd result = optimizer.optimize(initialPoint);

    std::cout << "Optimized parameters: " << result.transpose() << std::endl;
    std::cout << "Minimum value: " << rosenbrock(result) << std::endl;

    return 0;
}

