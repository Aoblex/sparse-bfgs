#include "BFGS.h"
#include <Eigen/Dense>
#include <iostream>

int main() {
    auto rosenbrock = [](const Eigen::VectorXd& x) -> double {
        double a = 1.0, b = 100.0;
        double x1 = x(0), x2 = x(1);
        return std::pow(a - x1, 2) + b * std::pow(x2 - x1 * x1, 2);
    };

    auto rosenbrockGrad = [](const Eigen::VectorXd& x) -> Eigen::VectorXd {
        double a = 1.0, b = 100.0;
        double x1 = x(0), x2 = x(1);
        Eigen::VectorXd grad(2);
        grad(0) = -2 * (a - x1) - 4 * b * (x2 - x1 * x1) * x1;
        grad(1) = 2 * b * (x2 - x1 * x1);
        return grad;
    };

    Eigen::VectorXd initialPoint(2);
    initialPoint << -1.2, 1.0;

    BFGS optimizer(rosenbrock, rosenbrockGrad);
    Eigen::VectorXd result = optimizer.optimize(initialPoint);

    std::cout << "Optimized parameters: " << result.transpose() << std::endl;
    std::cout << "Minimum value: " << rosenbrock(result) << std::endl;

    return 0;
}

