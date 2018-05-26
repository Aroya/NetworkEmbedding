#ifndef Aroya_READER_HELPER_H
#define Aroya_READER_HELPER_H
#include<vector>
#include<fstream>
using namespace std;
#include"Reader.h"

class AroyaReaderHelper {
public:
	AroyaReaderHelper();
	
	//if not input myTableName, will set it as same as tableName
	void insert(AroyaReader&reader, const char*tableName, const char*myTableName=nullptr);
	//insert by column position
	void insert(AroyaReader&reader, const int&col, const bool&exist_table=true);

	void insertAll(AroyaReader&reader, const bool&exist_table = true);

	//写出数据
	void writeFile(const char*fileName);

	//get
	int getRows();
	int getColumns();

	//for AroyaKMeans to load
	vector<vector<double>> getData();
	//for bpnn to load
	double** getDataPointer();

	//记录table
	void saveTable(const char*fileName);
	//更变为记录的table顺序
	void transformTable(const char*fileName);
private:
	int findTable(const char*tableName);
	vector<string>table;
	vector<vector<double>>buffer;
	vector<double>empty;
};

#endif
#include"ReaderHelper.cpp"