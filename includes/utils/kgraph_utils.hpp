//Source: https://github.com/JieFengWang/mini_rnn/blob/main/utils/kgraph_utils.hpp


#pragma once

/*
 * store the data
 * calculate the distance
 */

#include <cmath>
#include <cstring>
#include <malloc.h>
#include <vector>
#include <fstream>
#include <stdexcept>
#include <iostream>
#include <stdlib.h>
#ifdef __GNUC__
#ifdef __AVX__
#define KGRAPH_MATRIX_ALIGN 32
#else
#ifdef __SSE2__
#define KGRAPH_MATRIX_ALIGN 16
#else
#define KGRAPH_MATRIX_ALIGN 4
#endif
#endif
#else
#define KGRAPH_MATRIX_ALIGN 4
#endif

#include "basis.h"

namespace rnndescent
{

    /// namespace for various distance metrics.
    namespace metric
    {
        /// L2 square distance.
        struct l2sqr
        {
            template <typename T>
            /// L2 square distance.
            static float apply(T const *t1, T const *t2, unsigned dim)
            {
                return cal_L2sqr(t1,t2,dim);
                float r = 0;
                for (unsigned i = 0; i < dim; ++i)
                {
                    float v = float(t1[i]) - float(t2[i]);
                    v *= v;
                    r += v;
                }
                return r;
            }

            /// inner product.
            template <typename T>
            static float dot(T const *t1, T const *t2, unsigned dim)
            {
                return cal_inner_product(t1,t2,dim);

                float r = 0;
                for (unsigned i = 0; i < dim; ++i)
                {
                    r += float(t1[i]) * float(t2[i]);
                }
                return r;
            }

            /// L2 norm.
            template <typename T>
            static float norm2(T const *t1, unsigned dim)
            {
                return (cal_inner_product(t1,t1,dim));

                float r = 0;
                for (unsigned i = 0; i < dim; ++i)
                {
                    float v = float(t1[i]);
                    v *= v;
                    r += v;
                }
                return r;
            }
        };

        struct l2
        {
            template <typename T>
            static float apply(T const *t1, T const *t2, unsigned dim)
            {
                return sqrt(l2sqr::apply<T>(t1, t2, dim));
            }
        };
    }

    /// Matrix data.
    template <typename T, unsigned A = KGRAPH_MATRIX_ALIGN>
    class Matrix
    {
        unsigned cur_size;
        unsigned col;
        unsigned row;
        size_t stride;
        char *data;

        void reset(unsigned r, unsigned c)
        {
            cur_size = 0;
            row = r;
            col = c;
            stride = (sizeof(T) * c + A - 1) / A * A;
            /*
            data.resize(row * stride);
            */
            if (data)
                free(data);

#ifdef __GNUC__
            data = (char*)memalign(A, row * stride); // SSE instruction needs data to be aligned
#else
            data = (char*)_aligned_malloc(row * stride, A);
#endif
            

            // if (!data) throw runtime_error("memalign");
        }

    public:
        Matrix() : col(0), row(0), stride(0), data(0) {}
        Matrix(unsigned r, unsigned c) : data(0)
        {
            reset(r, c);
        }
        ~Matrix()
        {
            if (data) {
#ifdef __GNUC__
                free(data);
#else
                _aligned_free(data);
#endif
            }
                
        }
        unsigned size() const
        {
            return row;
        }
        unsigned dim() const
        {
            return col;
        }
        size_t step() const
        {
            return stride;
        }
        void resize(unsigned r, unsigned c)
        {
            reset(r, c);
        }
        T const *operator[](unsigned i) const
        {
            return reinterpret_cast<T const *>(&data[stride * i]);
        }
        T *operator[](unsigned i)
        {
            return reinterpret_cast<T *>(&data[stride * i]);
        }
        void zero()
        {
            memset(data, 0, row * stride);
        }

        void normalize2()
        {
#pragma omp parallel for
            for (unsigned i = 0; i < row; ++i)
            {
                T *p = operator[](i);
                double sum = metric::l2sqr::norm2(p, col);
                sum = std::sqrt(sum);
                for (unsigned j = 0; j < col; ++j)
                {
                    p[j] /= sum;
                }
            }
        }

        void load(const std::string &path, unsigned dim, unsigned skip = 0, unsigned gap = 0)
        {
            std::ifstream is(path.c_str(), std::ios::binary);
            // if (!is) throw io_error(path);
            is.seekg(0, std::ios::end);
            size_t size = is.tellg();
            size -= skip;
            is.seekg(0, std::ios::beg);
            is.read((char *)&dim, sizeof(unsigned int));
            unsigned line = sizeof(T) * dim + gap;
            unsigned N = size / line;
            // {
            //     std::cout << "N = " << N << std::endl;
            // }
            reset(N, dim);
            zero();
            is.seekg(skip, std::ios::beg);
            for (unsigned i = 0; i < N; ++i)
            {
                is.seekg(gap, std::ios::cur);
                is.read(&data[stride * i], sizeof(T) * dim);
            }
            // if (!is) throw io_error(path);
        }


        void load(const T *base_vector, size_t data_size, size_t data_dim)
        {
            reset(data_size, data_dim);
            zero();
            for (size_t i = 0; i < data_size; ++i)
            {
                memcpy(&data[stride * i], base_vector + i * data_dim, sizeof(T) * data_dim);
            }
        }

        void add_test(const T *new_vector)
        {
            /// 一定要添加的size == row
            if (cur_size >= row)
                return;
            memcpy(&data[stride * cur_size], new_vector, sizeof(T) * col);
            ++cur_size;
        }

        void batch_add_test(const T *vector, size_t n)
        {
            /// 一定要添加的size == row
            // if (cur_size >= row) return;
            // memcpy(&data[stride * cur_size], new_vector, sizeof(T) * col);
            // ++cur_size;
#pragma omp parallel for
            for (unsigned i = 0; i < n; ++i)
            {
                memcpy(&data[stride * i], (char *)((float *)vector + i * col), sizeof(T) * col);
            }

            cur_size = n;
        }
    };

    /// Matrix proxy to interface with 3rd party libraries (FLANN, OpenCV, NumPy).
    template <typename DATA_TYPE, unsigned A = KGRAPH_MATRIX_ALIGN>
    class MatrixProxy
    {
        unsigned rows;
        unsigned cols; // # elements, not bytes, in a row,
        size_t stride; // # bytes in a row, >= cols * sizeof(element)
        uint8_t const *data;

    public:
        MatrixProxy(Matrix<DATA_TYPE> const &m)
            : rows(m.size()), cols(m.dim()), stride(m.step()), data(reinterpret_cast<uint8_t const *>(m[0]))
        {
        }

#ifndef __AVX__
#ifdef FLANN_DATASET_H_
        /// Construct from FLANN matrix.
        MatrixProxy(flann::Matrix<DATA_TYPE> const &m)
            : rows(m.rows), cols(m.cols), stride(m.stride), data(m.data)
        {
            if (stride % A)
                throw invalid_argument("bad alignment");
        }
#endif
#ifdef CV_MAJOR_VERSION
        /// Construct from OpenCV matrix.
        MatrixProxy(cv::Mat const &m)
            : rows(m.rows), cols(m.cols), stride(m.step), data(m.data)
        {
            if (stride % A)
                throw invalid_argument("bad alignment");
        }
#endif
#ifdef NPY_NDARRAYOBJECT_H
        /// Construct from NumPy matrix.
        MatrixProxy(PyArrayObject *obj)
        {
            if (!obj || (obj->nd != 2))
                throw invalid_argument("bad array shape");
            rows = obj->dimensions[0];
            cols = obj->dimensions[1];
            stride = obj->strides[0];
            data = reinterpret_cast<uint8_t const *>(obj->data);
            if (obj->descr->elsize != sizeof(DATA_TYPE))
                throw invalid_argument("bad data type size");
            if (stride % A)
                throw invalid_argument("bad alignment");
            if (!(stride >= cols * sizeof(DATA_TYPE)))
                throw invalid_argument("bad stride");
        }
#endif
#endif
        unsigned size() const
        {
            return rows;
        }
        unsigned dim() const
        {
            return cols;
        }
        DATA_TYPE const *operator[](unsigned i) const
        {
            return reinterpret_cast<DATA_TYPE const *>(data + stride * i);
        }
        DATA_TYPE *operator[](unsigned i)
        {
            return const_cast<DATA_TYPE *>(reinterpret_cast<DATA_TYPE const *>(data + stride * i));
        }
    };

    class IndexOracle
    {
    public:
        /// Returns the size of the dataset.
        virtual unsigned size() const = 0;
        /// Computes similarity
        /**
         * 0 <= i, j < size() are the index of two objects in the dataset.
         * This method return the distance between objects i and j.
         */
        virtual float operator()(unsigned i, unsigned j) const = 0;
        virtual float get_dist(unsigned, float *) const = 0;
        //virtual void *operator[](unsigned i) const = 0;
    };

    template <typename DATA_TYPE, typename DIST_TYPE>
    class MatrixOracle : public IndexOracle
    {
        MatrixProxy<DATA_TYPE> proxy;

    public:
        template <typename MATRIX_TYPE>
        MatrixOracle(MATRIX_TYPE const &m) : proxy(m)
        {
        }
        virtual unsigned size() const
        {
            return proxy.size();
        }
        virtual float operator()(unsigned i, unsigned j) const
        {
            return DIST_TYPE::apply(proxy[i], proxy[j], proxy.dim());
        }
        virtual float get_dist(unsigned i, float *vec) const
        {
            return DIST_TYPE::apply((float *)proxy[i], vec, proxy.dim());
        }
        virtual DATA_TYPE *operator[](unsigned i) const
        {
            return const_cast<DATA_TYPE *>(proxy[i]);
        }
    };

}
