#include "LineSearch.h"

double simpleLineSearch(
        const Eigen::VectorXd& x, // Current point
        const Eigen::VectorXd& p, // Search direction
        std::function<double(const Eigen::VectorXd&)> objectiveFunction, // Objective function
        std::function<Eigen::VectorXd(const Eigen::VectorXd&)> gradientFunction, // Gradient function
        // alpha is the initial step size
        // c is the Armijo condition parameter
        // tau is the factor by which alpha is multiplied when the Armijo condition is not satisfied
        double alpha, double c, double tau) {
    double fx = objectiveFunction(x);
    Eigen::VectorXd grad = gradientFunction(x);
    // Armijo condition
    while (objectiveFunction(x + alpha * p) > fx + c * alpha * grad.dot(p)) {
        alpha *= tau;
    }
    return alpha;

}

