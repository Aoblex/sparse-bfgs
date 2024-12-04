#ifndef LINESEARCH_H
#define LINESEARCH_H

#include <Eigen/Dense>
#include <functional>

double simpleLineSearch(
        const Eigen::VectorXd& x, const Eigen::VectorXd& p,
        std::function<double(const Eigen::VectorXd&)> objectiveFunction,
        std::function<Eigen::VectorXd(const Eigen::VectorXd&)> gradientFunction,
        double alpha = 1.0, double c = 1e-4, double tau = 0.5);

#endif // LINESEARCH_H
       //
