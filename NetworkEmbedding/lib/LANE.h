#ifndef AROYA_LANE_H
#define AROYA_LANE_H

#include<vector>
#include<cmath>
#include<algorithm>
using namespace std;

#include"KMeans.h"

#include "armadillo/include/armadillo"
using namespace arma;

#include"KMeans.h"

class LANE
{
    public:
      LANE(const int &dimension = 2);
      // void setDataByAdj(double **, const int &rows, const int &cols);
      void setAdjData(const vector<vector<double>>&);
      void setAttrData(const vector<vector<double>>&);
      //Run setLabel after setAttr
      void setLabelData(const vector<vector<double>>&);
      AroyaKMeans kmeans;
      void run();
    private:
      mat L_g, L_a, L_y;
      mat U_g, U_a, U_y, H;
      int rows, cols_attr, dimensions, labels;
      double alpha1, alpha2;
};

#endif
#include"LANE.cpp"