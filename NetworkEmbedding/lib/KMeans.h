#ifndef AROYA_KMEANS_H
#define AROYA_KMEANS_H

#include<iostream>
#include<vector>
#include<cmath>
#include<fstream>
using namespace std;

//#define AROYA_DEBUG

class AroyaKMeans {
public:
	AroyaKMeans();
	void setClusters(const int&clusters);					//设置聚类点数
	void setData(const vector<vector<double>>&yourData);	//从reader读入信息
	void run();												//进行聚类
	void setBord(const double&newBord);						//设置收敛条件
	void writeFile(const char*fileName,const bool&withData=false);	//聚类情况写出文件
	vector<int>getFlag();
private:
	vector<vector<double>>data;		//data
	int*cluster;					//指明data所在的centre
	vector<vector<double>>centre;	//质心
	double **distance;				//data到质心的距离
	int rows, columns, clusters;
	double bord;					//默认5%以下的变化时已收敛
};


#endif
#include"KMeans.cpp"