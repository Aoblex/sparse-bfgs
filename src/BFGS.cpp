#include "BFGS.h"
#include "MatrixOps.h"
#include "LineSearch.h"
#include <iostream>
#include <cmath>
#include <chrono>

// BFGS constructor
BFGS::BFGS(
        std::function<double(const Eigen::VectorXd&)> objectiveFunction,
        std::function<Eigen::VectorXd(const Eigen::VectorXd&)> gradientFunction,
        double tolerance, int maxIterations)
    : objectiveFunction(objectiveFunction), gradientFunction(gradientFunction),
    tolerance(tolerance), maxIterations(maxIterations) {}

// BFGS::optimize function
Eigen::VectorXd BFGS::H_optimize(const Eigen::VectorXd& initialPoint) {
    // time start
    auto start = std::chrono::high_resolution_clock::now();

    // declaration
    Eigen::VectorXd x, grad, x_new, grad_new, p;
    Eigen::VectorXd y, s;
    Eigen::MatrixXd Hessian_inv_hat, W;
    int n, iter;
    double step_size, rho;

    // initialization
    x = initialPoint;
    grad = gradientFunction(x);
    n = x.size(), iter = 0;
    Hessian_inv_hat = Eigen::MatrixXd::Identity(n, n);

    // start the optimization
    while(grad.norm() > tolerance && iter < maxIterations) {
        p = -Hessian_inv_hat * grad;
        step_size = WolfeLineSearch(x, p, objectiveFunction, gradientFunction);
        s = step_size * p;

        x_new = x + s;
        grad_new = gradientFunction(x_new);
        y = grad_new - grad;

        // update Hessian_hat
        rho = 1.0 / y.dot(s);
        if (std::isfinite(rho)) { // Update the inverse Hessian
            W = Eigen::MatrixXd::Identity(n, n) - rho * y * s.transpose();
            Hessian_inv_hat = W.transpose() * Hessian_inv_hat * W + rho * (s * s.transpose());
        }

        // update x and grad
        x = x_new;
        grad = grad_new;

        // show the iteration
        iter++;
        if (iter % 50 == 0) {
            std::cout << "Iteration: " << iter
                << ", Objective: " << objectiveFunction(x)
                << ", Gradient norm: " << grad.norm() << std::endl;
        }
    }

    // check convergence
    if (grad.norm() <= tolerance)
        std::cout << "Converged after " << iter << " iterations." << std::endl;
    else
        std::cout << "Maximum iterations reached without convergence." << std::endl;
    
    // time end
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Elapsed time: " << elapsed.count() << " s" << std::endl;

    return x;
}

Eigen::VectorXd BFGS::B_optimize(const Eigen::VectorXd& initialPoint) {
    // time start
    auto start = std::chrono::high_resolution_clock::now();

    // declaration
    Eigen::VectorXd x, grad, x_new, grad_new, p;
    Eigen::VectorXd y, s, W;
    Eigen::MatrixXd Hessian_hat;
    Eigen::LLT<Eigen::MatrixXd> solver;
    double step_size, rho;
    int n, iter;

    // initialization
    x = initialPoint;
    grad = gradientFunction(x);
    n = x.size(), iter = 0;
    Hessian_hat = Eigen::MatrixXd::Identity(n, n);

    // start the optimization
    while (grad.norm() > tolerance && iter < maxIterations) {
        // calculate the search direction
        solver.compute(Hessian_hat);
        if (solver.info() != Eigen::Success) {
            std::cerr << "Decomposition failed!" << std::endl;
            break;
        }
        p = -solver.solve(grad);

        // perform line search
        step_size = WolfeLineSearch(x, p, objectiveFunction, gradientFunction);

        // update Hessian_hat
        s = step_size * p; // Calculate the step
        x_new = x + s;
        grad_new = gradientFunction(x_new);
        y = grad_new - grad;
        W = Hessian_hat * s;
        Hessian_hat += (y * y.transpose()) / y.dot(s) - (W * W.transpose()) / s.dot(W);

        // update x and grad
        x = x_new;
        grad = grad_new;

        // show the iteration
        iter++;
        if (iter % 50 == 0) {
            std::cout << "Iteration: " << iter
                << ", Objective: " << objectiveFunction(x)
                << ", Gradient norm: " << grad.norm() << std::endl;
        }
    }

    if (grad.norm() <= tolerance)
        std::cout << "Converged after " << iter << " iterations." << std::endl;
    else
        std::cout << "Maximum iterations reached without convergence." << std::endl;

    // time end
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Elapsed time: " << elapsed.count() << " s" << std::endl;

    return x;
}

Eigen::VectorXd BFGS::sparseB_optimize(
    const Eigen::VectorXd& initialPoint,
    const Eigen::VectorXd& x1_dis,
    const Eigen::VectorXd& x2_dis,
    double eta,
    const Eigen::MatrixXd& M
) { 
    /* This method does not converge with these parameters */

    //time start
    auto start = std::chrono::high_resolution_clock::now();

    // declaration
    int n = x1_dis.size(), m = x2_dis.size();
    int l = initialPoint.size(), iter = 0;
    Eigen::VectorXd x, grad, alpha, beta;
    Eigen::SparseMatrix<double> Hessian(l, l);
    Eigen::MatrixXd T(n, m);
    Eigen::VectorXd T_colsum(m), T_rowsum(n), p;
    double step_size, eps = 1e-6; // how much to add to diagonal elements?
    typedef Eigen::Triplet<double> Triplet;
    std::vector<Triplet> TripletList;
    Eigen::SimplicialLLT<Eigen::SparseMatrix<double>> solver;

    x = initialPoint, x(l - 1) = 0;
    do {
        // initialize
        grad = gradientFunction(x);
        alpha = x.head(n), beta = x.tail(m);
        for(int i = 0; i < n; ++i) {
            for(int j = 0; j < m; ++j) {
                T(i, j) = std::exp((alpha(i) + beta(j) - M(i, j)) / eta);
            }
        }

        T_colsum = T.rowwise().sum(); // shape = (n, 1)
        T_rowsum = T.colwise().sum(); // shape = (1, m)

        // top left
        for (int i = 0; i < n; ++i)
            TripletList.emplace_back(Triplet(i, i, T_colsum(i) + eps));

        // bottom right
        for (int j = 0; j < m; ++j)
            TripletList.emplace_back(Triplet(j + n, j + n, T_rowsum(j) + eps));
        
        // top right and bottom left
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < m; ++j) {
                double val = T(i, j);
                if (val < 1e-6) continue; // how to make it sparse?
                TripletList.emplace_back(Triplet(i, j + n, val));
                TripletList.emplace_back(Triplet(j + n, i, val));
            }
        
        // Construct the triplets
        Hessian.setFromTriplets(TripletList.begin(), TripletList.end());
        solver.compute(Hessian);

        // Check if the decomposition failed
        if (solver.info() != Eigen::Success) {
            std::cerr << "Decomposition failed!" << std::endl;
            break;
        }

        // update x
        p = -solver.solve(grad);
        p(l - 1) = 0; // set the last element to 0
        // step_size = WolfeLineSearch(x, p, objectiveFunction, gradientFunction);
        step_size = ArmijoLineSearch(x, p, objectiveFunction, gradientFunction);
        x = x + step_size * p;

        iter++;

        if (iter % 50 == 0) {
            std::cout << "Iteration: " << iter
                << ", Objective: " << objectiveFunction(x)
                << ", Gradient norm: " << grad.norm() << std::endl;
        }
    } while(grad.norm() > tolerance && iter < maxIterations);

    if (grad.norm() <= tolerance)
        std::cout << "Converged after " << iter << " iterations." << std::endl;
    else
        std::cout << "Maximum iterations reached without convergence." << std::endl;

    // time end
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Elapsed time: " << elapsed.count() << " s" << std::endl;
    
    return x;
}