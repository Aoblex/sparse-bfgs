#ifndef BFGS_H
#define BFGS_H

#include <Eigen/Dense>
#include <functional>

class BFGS {
    public:
        BFGS(std::function<double(const Eigen::VectorXd&)> objectiveFunction,
                std::function<Eigen::VectorXd(const Eigen::VectorXd&)> gradientFunction,
                double tolerance = 1e-6, int maxIterations = 1000);

        Eigen::VectorXd H_optimize(const Eigen::VectorXd& initialPoint);
        Eigen::VectorXd B_optimize(const Eigen::VectorXd& initialPoint);
        Eigen::VectorXd sparseB_optimize(const Eigen::VectorXd& initialPoint);

    private:
        std::function<double(const Eigen::VectorXd&)> objectiveFunction;
        std::function<Eigen::VectorXd(const Eigen::VectorXd&)> gradientFunction;
        double tolerance;
        int maxIterations;

        double lineSearch(const Eigen::VectorXd& x, const Eigen::VectorXd& p, double alpha = 1.0,
                double c = 1e-4, double tau = 0.5);

};

#endif // BFGS_H
       //
