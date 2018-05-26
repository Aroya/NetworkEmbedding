#ifndef AROYA_M_NMF_H
#define AROYA_M_NMF_H

#include<vector>
#include<cmath>
#include<stack>
using namespace std;

#include"eigen3/Eigen/Dense"
using namespace Eigen;

#include"KMeans.h"

class M_NMF
{
    public:
      M_NMF(const int &communities = 3, const int &dimension = 3, const double &alpha = 1.0, const double &beta = 1.0);
      void setSDataByS1(double **dataArray, const int &rows, const int &cols);
      void run();
	  void setLearnRate(const double&);
      vector<int> getAns();
      void setCommunities(const int&);
      AroyaKMeans kmeans;
    private:
      MatrixXd S, M, U, H, C, B;
      int rows, cols;
	  double learnrate;
      double alpha, beta;
      int communities;
      int dimension;
};

#endif
#include"M-NMF.cpp"