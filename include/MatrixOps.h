#include <vector>
#include <Eigen/Dense>
#include <Eigen/Sparse>

// Sparsify a matrix by keeping only the top 10% of the elements
Eigen::SparseMatrix<double> sparsifyMatrix(const Eigen::MatrixXd& matrix);

// Generate a random discrete distribution of size N
Eigen::VectorXd generateDiscreteDistribution(int N);
