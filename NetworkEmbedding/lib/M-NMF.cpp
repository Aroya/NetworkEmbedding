#ifndef AROYA_M_NMF_C
#define AROYA_M_NMF_C

#include"M-NMF.h"

M_NMF::M_NMF(const int &communities_, const int &dimension_, const double &alpha_, const double &beta_)
{
    alpha = alpha_;
    beta = beta_;
    communities = communities_;
    dimension = dimension_;
	learnrate = 0.1;
}
void M_NMF::setLearnRate(const double&t) { learnrate = t; }
void M_NMF::setSDataByS1(double **dataArray, const int &rows_, const int &cols_)
{
    int i, j;
    rows = rows_;
    cols = cols_;
    MatrixXd S1;
    S1.resize(rows, cols);
    double *double_p;
    for (i = 0; i < rows;i++)
    {
        double_p = dataArray[i];
        for (j = 0; j < cols;j++)
        {
            S1(i, j) = double_p[j];
        }
    }
	//cout << S1 << endl;
	//system("pause");
    MatrixXd S2;
    S2.resize(rows, cols);
    vector<MatrixXd>vector_b;
    double *vector_n1 = (double *)malloc(sizeof(double) * rows);
    double *vector_sum = (double *)malloc(sizeof(double) * rows);
    for (i = 0; i < rows;i++)
    {
		vector_b.push_back(S1.block(i, 0, 1, cols));
		//cout << "Vector:"<<vector_b[i] << endl;
		//cout << "absSum:" << vector_b[i].cwiseAbs().sum() << endl;
		//cout << "Sum:" << vector_b[i].sum() << endl;

        vector_n1[i] = vector_b[i].cwiseAbs().sum();
        vector_sum[i] = vector_b[i].sum();
    }
    double scalar_2e = 0.0;
    for (i = 0; i < rows;i++)
    {
        scalar_2e += vector_sum[i];
    }

    for (i = 0; i < rows; i++)
    {
        for (j = i; j < cols; j++)
        {
			//cout << "upper:"<<vector_b[i].cwiseProduct(vector_b[j]).sum() << endl;
			//cout << "lower:"<<vector_n1[i] * vector_n1[j] << endl;
            S2(i, j) = vector_b[i].cwiseProduct(vector_b[j]).sum() / (vector_n1[i] * vector_n1[j]);
			S2(j, i) = S2(i, j);
        }
    }
    B.resize(rows, cols);
    for (i = 0; i < rows; i++)
    {
        for (j = 0; j < cols;j++)
        {
            B(i, j) = S1(i, j) - vector_sum[i] * vector_sum[j] / scalar_2e;
        }
    }
    free(vector_n1);
    free(vector_sum);
    S = S1 + 5 * S2;
}

void M_NMF::run(){
    int i, j;

    MatrixXd Mupper, Mlower, Uupper, Ulower, Cupper, Clower, Hupper, Hlower;
    MatrixXd Mnext, Unext, Cnext, Hnext;
    //S→NMF→M, U
    Mnext.resize(rows, dimension);
    Unext.resize(cols, dimension);
    Hnext.resize(rows, communities);
    Cnext.resize(communities, dimension);
	//initial H is set by the nmf of B

	//find k minimum eigenvectors to dimension reduction
	{
		EigenSolver<MatrixXd> Bsolver(B);
		MatrixXcd eigenVectors = Bsolver.eigenvectors();
		MatrixXcd eigenValues = Bsolver.eigenvalues();
		stack<int>indexBase;
		stack<int>tempBase;
		int rows = eigenVectors.rows();
		int cols = eigenVectors.cols();
		indexBase.push(0);
		for (i = 1; i < rows; i++) {
			while (!indexBase.empty() && eigenValues(i, 0).real() != 0 &&
				eigenValues(indexBase.top(), 0).real() > eigenValues(i, 0).real())
			{
				tempBase.push(indexBase.top());
				indexBase.pop();
			}
			indexBase.push(i);
			while (!tempBase.empty()) {
				indexBase.push(tempBase.top());
				tempBase.pop();
			}
			while (indexBase.size()>communities) {
				indexBase.pop();
			}
		}
		//select the vectors
		int k;
		for (i = 0; i < communities; i++) {
			k = indexBase.top();
			indexBase.pop();
			for (j = 0; j < rows; j++) {
				Hnext(j, i) = eigenVectors(j, k).real();
			}
		}
	}

    Mnext.setRandom();
    Unext.setRandom();
    Cnext.setRandom();
    //Minimize M&U
   	//**********************************************//
	//part 1 is always nan							//
	//add a nmf to reduce error on part 1 first		//
	//**********************************************//

	double LossBefore = DBL_MAX;
	//cout << S << endl;
	//cout << (S - Mnext * Unext.transpose()).cwiseAbs().sum() << '\t' << alpha * (Hnext - Unext * Cnext.transpose()).norm() << '\t' << beta * (Hnext.transpose() * B * Hnext).trace() << endl;
	double LossNow = (S - Mnext * Unext.transpose()).norm() + alpha * (Hnext - Unext * Cnext.transpose()).norm() - beta * (Hnext.transpose() * B * Hnext).trace();
	//int n = 100;
	//while (n--)
    while(LossBefore>LossNow)
    {
        //update
        M = Mnext;
        U = Unext;
        H = Hnext;
        C = Cnext;
        //Mnext
        Mupper = S * U;
        Mlower = M * U.transpose() * U;
        //Unext
		Uupper = S.transpose() * M + alpha*H*C;
		Ulower = U * (M.transpose()*M + alpha*C.transpose()*C);
        //Hnext
        //Lagrange multiplier not add
        Hupper = -2 * beta * B * H;
        Hlower = 8 * H * H.transpose() * H;
        //Cnext
        Cupper = H.transpose() * U;
        Clower = C * U.transpose() * U;

        //update next
        for (i = 0; i < Hnext.rows();i++)
        {
            for (j = 0; j < Hnext.cols();j++)
            {  
                Hnext(i, j) = H(i, j) * sqrt(Hupper(i, j) / Hlower(i, j));
            }
        }
		for (i = 0; i < Mnext.rows(); i++)
		{
			for (j = 0; j < Mnext.cols(); j++)
			{
				Mnext(i, j) = M(i, j) * Mupper(i, j) / Mlower(i, j);
			}
		}
		for (i = 0; i < Unext.rows(); i++)
		{
			for (j = 0; j < Unext.cols(); j++)
			{
				Unext(i, j) = U(i, j) * Uupper(i, j) / Ulower(i, j);
			}
		}
		for (i = 0; i < Cnext.rows(); i++)
		{
			for (j = 0; j < Cnext.cols(); j++)
			{
				Cnext(i, j) = C(i, j) * Cupper(i, j) / Clower(i, j);
			}
		}
		Hnext = H - Hnext*learnrate;
		Mnext = M - Mnext*learnrate;
		Unext = U - Unext*learnrate;
		Cnext = C - Cnext*learnrate;
        //update loss
        LossBefore = LossNow;
		//cout << (S - Mnext * Unext.transpose()).cwiseAbs().sum() << '\t' << alpha * (Hnext - Unext * Cnext.transpose()).norm() << '\t'<< beta * (Hnext.transpose() * B * Hnext).trace() << endl;
        LossNow = (S - Mnext * Unext.transpose()).norm() + alpha * (Hnext - Unext * Cnext.transpose()).norm() - beta * (Hnext.transpose() * B * Hnext).trace();
        printf("LossBefore:\t%f\t\tLossNew:\t%f\t\n", LossBefore, LossNow);
    }
    vector<vector<double>> toKmeans;
    for (i = 0; i < cols;i++)
    {
        vector<double> empty;
        for (j = 0; j < dimension;j++)
        {
            empty.push_back(U(i, j));
        }
        toKmeans.push_back(empty);
    }
    kmeans.setData(toKmeans);
    kmeans.setClusters(communities);
    kmeans.run();
}
vector<int> M_NMF::getAns()
{
	return kmeans.getFlag();
}
void M_NMF::setCommunities(const int&a)
{
	communities = a;
}

#endif