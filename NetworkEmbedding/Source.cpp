#include<iostream>
#include<string>
using namespace std;

//need intel MKL libraries
#include"lib/Reader.h"
#include"lib/ReaderHelper.h"
#include"lib/M-NMF.h"
#include"lib/LANE.h"
#include"lib/Writer.h"

int main() {
	string adj;
	string fea;
	string lab;
	string outputNMF;
	string outputLANE;
	//cornell
	adj = "dataset/cornell/cornell_adj.txt";
	fea = "dataset/cornell/cornell_feature.txt";
	lab = "dataset/cornell/cornell_label.txt";
	outputNMF = "output/cornell_M-NMF.txt";
	outputLANE = "output/cornell_LANE.txt";
	
	////texas
	//adj = "dataset/texas/texas_adj.txt";
	//fea = "dataset/texas/texas_feature.txt";
	//lab = "dataset/texas/texas_label.txt";
	//outputNMF = "output/texas_M-NMF.txt";
	//outputLANE = "output/texas_LANE.txt";

	////washington
	//adj = "dataset/washington/washington_adj.txt";
	//fea = "dataset/washington/washington_feature.txt";
	//lab = "dataset/washington/washington_label.txt";
	//outputNMF = "output/washington_M-NMF.txt";
	//outputLANE = "output/washington_LANE.txt";

	////wisconsin
	//adj = "dataset/wisconsin/wisconsin_adj.txt";
	//fea = "dataset/wisconsin/wisconsin_feature.txt";
	//lab = "dataset/wisconsin/wisconsin_label.txt";
	//outputNMF = "output/wisconsin_M-NMF.txt";
	//outputLANE = "output/wisconsin_LANE.txt";

	AroyaReader reader;
	reader.setSplit('\t');
	reader.read(adj.c_str());
	AroyaReaderHelper readerHelper;
	readerHelper.insertAll(reader, false);

	AroyaReader reader_fea;
	reader_fea.setSplit('\t');
	reader_fea.read(fea.c_str());
	AroyaReaderHelper readerHelper_fea;
	readerHelper_fea.insertAll(reader_fea, false);

	AroyaReader reader_lab;
	reader_lab.setSplit('\t');
	reader_lab.read(lab.c_str());
	AroyaReaderHelper readerHelper_lab;
	readerHelper_lab.insertAll(reader_lab, false);
	
	double **p = readerHelper.getDataPointer();

	M_NMF m_nmf(5);
	m_nmf.setSDataByS1(p, readerHelper.getRows(), readerHelper.getColumns());
	printf("M-NMF:\n");
	m_nmf.run();
	writeFile(m_nmf.getAns(), outputNMF.c_str());
	printf("**************\n");
	LANE lane;
	lane.setAdjData(readerHelper.getData());
	lane.setAttrData(readerHelper_fea.getData());
	lane.setLabelData(readerHelper_lab.getData());
	printf("LANE:\n");
	lane.run();
	writeFile(lane.kmeans.getFlag(), outputLANE.c_str());
	printf("**************\n");
	system("pause");
	return 0;
}