#pragma once

#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <iterator>
#include <sstream>
#include <cstring>

#define VERBOSE

using Idx_Type = unsigned int;
using Dst_Type = float;
using Dat_Type = float;
using graph_t = std::vector<std::vector<Idx_Type>>;

using namespace std;

class IO
{

public:
    template <typename T>
    static void saveBinVec(string destPath, vector<vector<T>> &data)
    {

        size_t size_n = data.size();
        unsigned int dim = data[0].size();

        auto outStrm = std::fstream(destPath, ios::out | ios::binary);
        if (!outStrm.is_open())
        {
            std::cerr << "File '" << destPath << "' cannot open for write!" << std::endl;
            exit(0);
        }

        for (size_t i = 0; i < size_n; ++i)
        {
            dim = data[i].size();
            outStrm.write((char *)&dim, sizeof(unsigned int));
            for (size_t j = 0; j < dim; ++j)
            {
                outStrm.write((char *)&data[i][j], sizeof(unsigned int));
            }
        }
        outStrm.close();
#ifdef VERBOSE
        cout << destPath << " has been saved!" << endl;
#endif
    }

    template <typename T>
    static void saveBinVecPtr(string destPath, const T *data, size_t data_size, size_t data_dim)
    {

        size_t size_n = data_size;
        unsigned int dim = data_dim;

        auto outStrm = std::fstream(destPath, ios::out | ios::binary);
        if (!outStrm.is_open())
        {
            std::cerr << "File '" << destPath << "' cannot open for write!" << std::endl;
            exit(0);
        }

        for (size_t i = 0; i < size_n; ++i)
        {
            outStrm.write((char *)&dim, sizeof(unsigned int));
            outStrm.write((char *)(data + i * data_dim), data_dim * sizeof(T));
        }
        outStrm.close();
#ifdef VERBOSE
        cout << destPath << " has been saved!" << endl;
#endif
    }

    template <typename T>
    static vector<vector<T>> LoadBinVec(string in_pth)
    {
        ifstream data(in_pth, ios::binary);

        if (data.fail())
        {
            std::cout << "File :" << in_pth << "<-- cannot open\n";
            exit(0);
        }

        unsigned int dim = 0;

        data.seekg(0, ios::beg);

        vector<vector<T>> matrix;

        while (data.read((char *)&dim, sizeof(unsigned int)))
        {
            vector<T> row;
            row.resize(dim);
            data.read((char *)row.data(), dim * sizeof(T));
            matrix.push_back(row);
        }
#ifdef VERBOSE
        std::cout << "Read " << in_pth << " Succeed!" << std::endl;
        std::cout << "\t>> Elem typesize = " << sizeof(T) << std::endl;
        std::cout << "\t>> Its shape :: " << matrix.size() << " X " << matrix[0].size() << std::endl;
#endif
        return matrix;
    }

    template <typename T>
    static T *LoadBinVecPtr(string in_pth, size_t &nRow, size_t &nDim)
    {

        streampos fileSize;
        ifstream data(in_pth, ios::binary);

        if (data.fail())
        {
            std::cout << "File :" << in_pth << "<-- cannot open\n";
            exit(0);
        }

        // // get size :: row :: dim.
        unsigned int dim = 0;
        data.seekg(0, ios::end);
        fileSize = data.tellg();
        data.seekg(0, ios::beg);
        data.read((char *)&dim, sizeof(unsigned int));
        size_t size_n = (double)fileSize / (sizeof(unsigned int) + dim * sizeof(T));

        nRow = size_n;
        nDim = dim;
        T *matrix = new T[nRow * nDim];

        data.seekg(0, ios::beg);
        for (size_t i = 0; i < size_n; ++i)
        {
            data.read((char *)&dim, sizeof(unsigned int));
            data.read((char *)(matrix + i * nDim), nDim * sizeof(T));
        }
#ifdef VERBOSE
        std::cout << "Read " << in_pth << " Succeed!" << std::endl;
        std::cout << "\t>> Elem typesize = " << sizeof(T) << std::endl;
        std::cout << "\t>> Its shape :: " << nRow << " X " << nDim << std::endl;
#endif
        return matrix;
    }

    static bool endsWith(string const &str, string const &suffix)
    {
        if (str.length() < suffix.length())
        {
            return false;
        }
        return str.compare(str.length() - suffix.length(), suffix.length(), suffix) == 0;
    }
};
