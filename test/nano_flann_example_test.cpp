#include <nanoFlannExample/nanoFlannExample.h>

#include <catch2/catch.hpp>
#include <fstream>
#include <iostream>
#include <map>
#include <nanoflann.hpp>

#include <cstdlib>
#include <ctime>
#include <Eigen/Dense>

using namespace Eigen;
using namespace std;
using namespace nanoflann;

template <typename Der>
void generateRandomPointCloud(Eigen::MatrixBase<Der> &mat, const size_t N,
                              const size_t dim,
                              const typename Der::Scalar max_range = 10) {
  std::cout << "Generating " << N << " random points...";
  mat.resize(N, dim);
  for (size_t i = 0; i < N; i++)
    for (size_t d = 0; d < dim; d++)
      mat(i, d) = max_range * (rand() % 1000) / typename Der::Scalar(1000);
  std::cout << "done\n";
}

class Fixture
{


};

const int SAMPLES_DIM = 15;


TEST_CASE_METHOD(Fixture, "test running nano flann", "[nanoFlannExample.cpp]")
{
  auto nSamples = 1000;
  auto dim = 3;

  Eigen::Matrix<double, Dynamic, Dynamic> mat(nSamples, dim);

  const double max_range = 20;

  // Generate points:
  generateRandomPointCloud(mat, nSamples, dim, max_range);

  // Query point:
  std::vector<double> query_pt(dim);
  for (size_t d = 0; d < dim; d++)
    query_pt[d] = max_range * (rand() % 1000) / double(1000);

  typedef KDTreeEigenMatrixAdaptor<Eigen::Matrix<double, Dynamic, Dynamic>>
  my_kd_tree_t;

  my_kd_tree_t mat_index(dim, std::cref(mat), 10 /* max leaf */);
  mat_index.index->buildIndex();

  // do a knn search
  const size_t num_results = 3;
  vector<size_t> ret_indexes(num_results);
  vector<double> out_dists_sqr(num_results);

  nanoflann::KNNResultSet<double> resultSet(num_results);

  resultSet.init(&ret_indexes[0], &out_dists_sqr[0]);
  mat_index.index->findNeighbors(resultSet, &query_pt[0],
                                 nanoflann::SearchParams(10));

  std::cout << "knnSearch(nn=" << num_results << "): \n";
  for (size_t i = 0; i < num_results; i++)
    std::cout << "ret_index[" << i << "]=" << ret_indexes[i]
              << " out_dist_sqr=" << out_dists_sqr[i] << endl;
}