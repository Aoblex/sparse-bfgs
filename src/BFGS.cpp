#include "BFGS.h"
#include "LineSearch.h"
#include <iostream>
#include <cmath>

// BFGS constructor
BFGS::BFGS(
        std::function<double(const Eigen::VectorXd&)> objectiveFunction,
        std::function<Eigen::VectorXd(const Eigen::VectorXd&)> gradientFunction,
        double tolerance, int maxIterations)
    : objectiveFunction(objectiveFunction), gradientFunction(gradientFunction),
    tolerance(tolerance), maxIterations(maxIterations) {}

// BFGS::optimize function
Eigen::VectorXd BFGS::optimize(const Eigen::VectorXd& initialPoint) {
    Eigen::VectorXd x = initialPoint;
    int n = x.size();
    Eigen::MatrixXd H = Eigen::MatrixXd::Identity(n, n);  // Initialize the Inverse of Hessian as identity matrix
    Eigen::VectorXd grad = gradientFunction(x); // Gradient function
    Eigen::MatrixXd I = Eigen::MatrixXd::Identity(n, n); // Identity matrix

    int iter = 0;
    while (grad.norm() > tolerance && iter < maxIterations) {
        Eigen::VectorXd p = -H * grad; // Calculate the search direction

        double alpha = simpleLineSearch(x, p, objectiveFunction, gradientFunction);

        Eigen::VectorXd s = alpha * p; // Calculate the step
        Eigen::VectorXd x_new = x + s;
        Eigen::VectorXd grad_new = gradientFunction(x_new);
        Eigen::VectorXd y = grad_new - grad;

        double rho = 1.0 / y.dot(s);
        if (std::isfinite(rho)) { // Update the inverse Hessian
            Eigen::MatrixXd W = I - rho * y * s.transpose();
            H = W.transpose() * H * W + rho * (s * s.transpose());
        }

        x = x_new;
        grad = grad_new;
        iter++;

        std::cout << "Iteration: " << iter
            << ", Objective: " << objectiveFunction(x)
            << ", Gradient norm: " << grad.norm() << std::endl;

    }

    if (grad.norm() <= tolerance)
        std::cout << "Converged after " << iter << " iterations." << std::endl;
    else
        std::cout << "Maximum iterations reached without convergence." << std::endl;

    return x;
}

