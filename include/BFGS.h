#ifndef BFGS_H
#define BFGS_H

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>
#include <functional>

class BFGS {
    public:
        BFGS(std::function<double(const Eigen::VectorXd&)> objectiveFunction,
                std::function<Eigen::VectorXd(const Eigen::VectorXd&)> gradientFunction,
                double tolerance = 1e-6, int maxIterations = 10000);

        Eigen::VectorXd H_optimize(const Eigen::VectorXd& initialPoint);
        Eigen::VectorXd B_optimize(const Eigen::VectorXd& initialPoint);
        Eigen::VectorXd sparseB_optimize(
            const Eigen::VectorXd& initialPoint,
            const Eigen::VectorXd& x1_dis,
            const Eigen::VectorXd& x2_di,
            double eta,
            const Eigen::MatrixXd& M
        );

    private:
        std::function<double(const Eigen::VectorXd&)> objectiveFunction;
        std::function<Eigen::VectorXd(const Eigen::VectorXd&)> gradientFunction;
        double tolerance;
        int maxIterations;

};

#endif // BFGS_H
       //
