#ifndef AROYA_LANE_C
#define AROYA_LANE_C

#include"LANE.h"

static struct box{
    int i;
    double val;
};
static bool sort_box(const box&a,const box&b)
{
    if(a.val>b.val)
        return true;
    return false;
}
LANE::LANE(const int &dimension_)
{
    dimensions = dimension_;
    rows = 0;
    cols_attr = 0;
    alpha1 = 1;
    alpha2 = 1;
}

void LANE::setAdjData(const vector<vector<double>> &t)
{
    rows = t.size();
    assert(t.size() == t[0].size());
    mat S, D;
    vector<vec> G;
    S.resize(rows, rows);
    D.resize(rows, rows);
    D.zeros();
    int i, j;

    for (i = 0; i < rows;i++)
    {
        vec thisG(rows);
        for (j = 0; j < rows;j++)
        {
            thisG(j) = t[i][j];
        }
        G.push_back(thisG);
    }

    for (i = 0; i < rows;i++)
    {
        for (j = i; j < rows;j++)
        {
            S(i, j) = dot(G[i], G[j]) / (sum(abs(G[i])) * sum(abs(G[j])));
            S(j, i) = S(i, j);
        }
		D(i, i) = sum(G[i]);
    }
    L_g = inv(sqrt(D)) * S * inv(sqrt(D));
    //caculate U_g
    cx_mat eigvec;
    cx_vec eigval;
    eig_gen(eigval, eigvec, L_g);

    //if rows ordered
    int eignum = eigval.size();
    box *p = (box *)malloc(sizeof(box) * eignum);
    for (i = 0; i < eignum;i++)
    {
        p[i].i = i;
        p[i].val = eigval(i).real();
    }
    sort(p, p + eignum, sort_box);
    eignum = eignum > dimensions ? dimensions : eignum;
    int thisI;
    U_g.resize(rows, eignum);
    for (i = 0; i < eignum;i++)
    {
        thisI = p[i].i;
        for (j = 0; j < rows;j++)
        {
            U_g(j, i) = eigvec(thisI, j).real();
        }
    }
    free(p);
}


void LANE::setAttrData(const vector<vector<double>>&t){
    cols_attr = t[0].size();
    mat S, D;
    vector<vec> G;
    S.resize(rows, rows);
    D.resize(rows, rows);
	D.zeros();
    int i, j;

    for (i = 0; i < rows;i++)
    {
        vec thisG;
        thisG.resize(cols_attr);
        for (j = 0; j < cols_attr;j++)
        {
            thisG(j) = t[i][j];
        }
        G.push_back(thisG);
    }

    for (i = 0; i < rows;i++)
    {
        for (j = i; j < rows;j++)
        {
            S(i, j) = dot(G[i], G[j]) / (sum(abs(G[i])) * sum(abs(G[j])));
            S(j, i) = S(i, j);
        }
		D(i, i) = sum(G[i]);
    }
    L_a = inv(sqrt(D)) * S * inv(sqrt(D));
    //caculate U_a

    //copy the code of U_g
    cx_mat eigvec;
    cx_vec eigval;
    eig_gen(eigval, eigvec, L_a);

    //if rows ordered
    int eignum = eigval.size();
    box *p = (box *)malloc(sizeof(box) * eignum);
    for (i = 0; i < eignum;i++)
    {
        p[i].i = i;
        p[i].val = eigval(i).real();
    }
    sort(p, p + eignum, sort_box);
    eignum = eignum > dimensions ? dimensions : eignum;
    int thisI;
    U_a.resize(rows, eignum);
    for (i = 0; i < eignum;i++)
    {
        thisI = p[i].i;
        for (j = 0; j < rows;j++)
        {
            U_a(j, i) = eigvec(thisI, j).real();
        }
    }
    free(p);
}
void LANE::setLabelData(const vector<vector<double>> &t)
{
    int i, j;
    labels = INT_MIN;
    //generate matrix Y]
    for (i = 0; i < rows;i++)
    {
        if(t[i][0]>labels)
            labels = t[i][0];
    }
    mat Y(rows, labels, fill::zeros);
    Y.resize(rows, labels);
    for (i = 0; i < rows; i++)
    {
        Y(i, t[i][0] - 1) = 1;
    }
    mat YY = Y * Y.t();
    mat S, D;
	S.resize(rows, rows);
	D.resize(rows, rows);
    vector<vec> G;
    for (i = 0; i < rows;i++)
    {
        vec thisG;
        thisG.resize(rows);
        for (j = 0; j < rows;j++)
        {
			thisG(j) = YY(i, j);
        }
        G.push_back(thisG);
    }

    for (i = 0; i < rows;i++)
    {
        for (j = i; j < rows;j++)
        {
            S(i, j) = dot(G[i], G[j]) / (sum(abs(G[i])) * sum(abs(G[j])));
            S(j, i) = S(i, j);
        }
		D(i, i) = sum(G[i]);
    }
    L_y = inv(sqrt(D)) * S * inv(sqrt(D));
    mat L = L_y + U_g * U_g.t();

    //caculate U_y

    //copy the code of U_g
    cx_mat eigvec;
    cx_vec eigval;
    eig_gen(eigval, eigvec, L);

    //if rows ordered
    int eignum = eigval.size();
    box *p = (box *)malloc(sizeof(box) * eignum);
    for (i = 0; i < eignum;i++)
    {
        p[i].i = i;
        p[i].val = eigval(i).real();
    }
    sort(p, p + eignum, sort_box);
    eignum = eignum > dimensions ? dimensions : eignum;
    int thisI;
    U_y.resize(rows, eignum);
    for (i = 0; i < eignum;i++)
    {
        thisI = p[i].i;
        for (j = 0; j < rows;j++)
        {
            U_y(j, i) = eigvec(thisI, j).real();
        }
    }
    free(p);
}
void LANE::run()
{
	int i, j;
    H.resize(rows, dimensions);
    H.randu();
    double lossBefore = DBL_MAX;
    double LossNow, Jcorr, J_g, J_a, J_y, r1;
    mat Un_g = U_g;
    mat Un_a = U_a;
    mat Un_y = U_y;
    mat Hn = H;
    Jcorr = trace(Un_g.t() * H * H.t() * Un_g)   //r2
            + trace(Un_a.t() * H * H.t() * Un_a) //r3
            + trace(Un_y.t() * H * H.t() * Un_y);//r4
    r1 = trace(Un_a.t() * Un_g * Un_g.t() * Un_a);
    J_g = trace(Un_g.t() * L_g * Un_g);
    J_a = trace(Un_a.t() * L_a * Un_a);
    J_y = trace(Un_y.t() * (L_y + Un_g * Un_g.t()) * Un_y);
    LossNow = J_g + alpha1 * J_a + alpha1 * r1 + alpha2 * J_y + Jcorr;
    printf("Initial Loss=%f\n", LossNow);
    mat L;
    cx_mat eigvec;
    cx_vec eigval;
    box *p;
    int thisI;
    int eignum;
    while (LossNow < lossBefore)
    {
        U_g = Un_g;
        U_a = Un_a;
        U_y = Un_y;
        H = Hn;

        //update U_g by eq13
        L = L_g + alpha1 * U_a * U_a.t() + alpha2 * U_y * U_y.t() + H * H.t();

        eig_gen(eigval, eigvec, L);

        //if rows ordered
        eignum = eigval.size();
        p = (box *)malloc(sizeof(box) * eignum);
        for (i = 0; i < eignum;i++)
        {
            p[i].i = i;
            p[i].val = eigval(i).real();
        }
        sort(p, p + eignum, sort_box);
        eignum = eignum > dimensions ? dimensions : eignum;
        
        for (i = 0; i < eignum;i++)
        {
            thisI = p[i].i;
            for (j = 0; j < rows;j++)
            {
                Un_g(j, i) = eigvec(thisI, j).real();
            }
        }
        free(p);

        //update U_a by eq14
        L = alpha1 * L_a + alpha1 * U_g * U_g.t() + H * H.t();

                eig_gen(eigval, eigvec, L);

        //if rows ordered
        eignum = eigval.size();
        p = (box *)malloc(sizeof(box) * eignum);
        for (i = 0; i < eignum;i++)
        {
            p[i].i = i;
            p[i].val = eigval(i).real();
        }
        sort(p, p + eignum, sort_box);
        eignum = eignum > dimensions ? dimensions : eignum;
        
        for (i = 0; i < eignum;i++)
        {
            thisI = p[i].i;
            for (j = 0; j < rows;j++)
            {
                Un_a(j, i) = eigvec(thisI, j).real();
            }
        }
        free(p);
        //update U_y by eq15
        L = alpha2 * L_y + alpha2 * U_g * U_g.t() + H * H.t();

        eig_gen(eigval, eigvec, L);

        //if rows ordered
        eignum = eigval.size();
        p = (box *)malloc(sizeof(box) * eignum);
        for (i = 0; i < eignum;i++)
        {
            p[i].i = i;
            p[i].val = eigval(i).real();
        }
        sort(p, p + eignum, sort_box);
        eignum = eignum > dimensions ? dimensions : eignum;
        
        for (i = 0; i < eignum;i++)
        {
            thisI = p[i].i;
            for (j = 0; j < rows;j++)
            {
                Un_y(j, i) = eigvec(thisI, j).real();
            }
        }
        free(p);
        //update H   by eq16
        L = U_g * U_g.t() + H * H.t() + U_a * U_a.t();

        eig_gen(eigval, eigvec, L);

        //if rows ordered
        eignum = eigval.size();
        p = (box *)malloc(sizeof(box) * eignum);
        for (i = 0; i < eignum;i++)
        {
            p[i].i = i;
            p[i].val = eigval(i).real();
        }
        sort(p, p + eignum, sort_box);
        eignum = eignum > dimensions ? dimensions : eignum;
        
        for (i = 0; i < eignum;i++)
        {
            thisI = p[i].i;
            for (j = 0; j < rows;j++)
            {
                Hn(j, i) = eigvec(thisI, j).real();
            }
        }
        free(p);

        //new Loss
        Jcorr = trace(Un_g.t() * H * H.t() * Un_g)   //r2
            + trace(Un_a.t() * H * H.t() * Un_a) //r3
            + trace(Un_y.t() * H * H.t() * Un_y);//r4
        r1 = trace(Un_a.t() * Un_g * Un_g.t() * Un_a);
        J_g = trace(Un_g.t() * L_g * Un_g);
        J_a = trace(Un_a.t() * L_a * Un_a);
        J_y = trace(Un_y.t() * (L_y + Un_g * Un_g.t()) * Un_y);
		lossBefore = LossNow;
        LossNow = J_g + alpha1 * J_a + alpha1 * r1 + alpha2 * J_y + Jcorr;
        printf("nowLoss:%f\n", LossNow);
    }
    vector<vector<double>> tt;
    for (i = 0; i < rows;i++)
    {
        vector<double> ttt;
        for (j = 0; j < eignum;j++)
        {
            ttt.push_back(H(i, j));
        }
        tt.push_back(ttt);
    }
    kmeans.setData(tt);
    kmeans.setClusters(labels);
    kmeans.run();
}
#endif