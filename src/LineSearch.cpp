#include "LineSearch.h"

double ArmijoLineSearch(
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

    // doesn't converge if alpha is too small
    return alpha >= 1e-2 ? alpha : 1e-2;
}

double WolfeLineSearch(
        const Eigen::VectorXd& x, const Eigen::VectorXd& p,
        std::function<double(const Eigen::VectorXd&)> objectiveFunction,
        std::function<Eigen::VectorXd(const Eigen::VectorXd&)> gradientFunction,
        double alpha, double c1, double c2, int maxIterations) {

    double alphaLow = 0.0;
    double alphaHigh = std::numeric_limits<double>::infinity();

    double fx = objectiveFunction(x);
    Eigen::VectorXd grad = gradientFunction(x);
    double gradDotP = grad.dot(p);

    for (int iter = 0; iter < maxIterations; ++iter) {
        Eigen::VectorXd xNew = x + alpha * p;
        double fxNew = objectiveFunction(xNew);
        Eigen::VectorXd gradNew = gradientFunction(xNew);
        double gradNewDotP = gradNew.dot(p);

        // Check Armijo condition (sufficient decrease)
        if (fxNew > fx + c1 * alpha * gradDotP) {
            alphaHigh = alpha;

        } 
        // Check Curvature condition
        else if (gradNewDotP < c2 * gradDotP) {
            alphaLow = alpha;

        } else {
            // Both conditions satisfied
            return alpha;

        }

        // Update alpha using bisection or interpolation
        if (std::isfinite(alphaHigh)) {
            alpha = 0.5 * (alphaLow + alphaHigh); // Bisection

        } else {
            alpha *= 2.0; // Increase step size

        }

    }

    std::cerr << "Wolfe line search did not converge within the maximum iterations." << std::endl;
    return alpha;
}
