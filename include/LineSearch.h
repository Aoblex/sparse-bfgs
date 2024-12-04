#ifndef LINESEARCH_H
#define LINESEARCH_H

#include <Eigen/Dense>
#include <functional>
#include <iostream>

double ArmijoLineSearch(
        const Eigen::VectorXd& x, const Eigen::VectorXd& p,
        std::function<double(const Eigen::VectorXd&)> objectiveFunction,
        std::function<Eigen::VectorXd(const Eigen::VectorXd&)> gradientFunction,
        double alpha = 1.0, double c = 1e-4, double tau = 0.5);

double WolfeLineSearch(
        const Eigen::VectorXd& x, const Eigen::VectorXd& p,
        std::function<double(const Eigen::VectorXd&)> objectiveFunction,
        std::function<Eigen::VectorXd(const Eigen::VectorXd&)> gradientFunction,
        double alpha = 1.0, double c1 = 1e-4, double c2 = 0.9, int maxIterations = 1000);

#endif // LINESEARCH_H
       //
