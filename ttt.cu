#include <iostream>
#include <vector>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cusolverDn.h> 
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/gather.h>
#include <thrust/reduce.h>
#include <thrust/transform_reduce.h>
#include <thrust/random.h>


namespace mt{
template<typename T> class device_vector;
};

cublasHandle_t g_handle;
cusolverDnHandle_t g_handle2;

enum class VectorType
{ 
  Row,
  Col
};

struct transpose_index : public thrust::unary_function<size_t,size_t>
{
    size_t m, n;

    __host__ __device__
    transpose_index(size_t _m, size_t _n) : m(_m), n(_n) {}

    __host__ __device__
    size_t operator()(size_t linear_index)
    {
        size_t i = linear_index / n;
        size_t j = linear_index % n;

        return m * j + i;
    }
};

template<typename T>
thrust::host_vector<T> line(thrust::host_vector<thrust::host_vector<T>> hvv){
    thrust::host_vector<T> ret;
    for(auto &hv: hvv){
        for(auto u :hv){
            ret.push_back(u);
        }
    }

    return ret;
}

template <typename T>
auto transpose(thrust::host_vector<T>&& src, size_t m, size_t n)
{
    thrust::host_vector<T> dst(m *n);
    
    thrust::counting_iterator<size_t> indices(0);

    thrust::gather
    (thrust::make_transform_iterator(indices, transpose_index(n, m)),
    thrust::make_transform_iterator(indices, transpose_index(n, m)) + dst.size(),
    src.begin(),dst.begin());

    return dst;
}


template <typename T>
mt::device_vector<T> transpose(mt::device_vector<T>& src, size_t m, size_t n)
{
    mt::device_vector<T> dst(m *n);
    
    thrust::counting_iterator<size_t> indices(0);

    thrust::gather
    (thrust::make_transform_iterator(indices, transpose_index(n, m)),
    thrust::make_transform_iterator(indices, transpose_index(n, m)) + dst.size(),
    src.begin(),dst.begin());

    return dst;
}

namespace mt{
template<typename T = float>
class device_vector : public thrust::device_vector<T>{

public: 

    using thrust::device_vector<T, thrust::THRUST_200500_500_NS::device_allocator<T>>::iterator;
    using thrust::device_vector<T>::data;
    using thrust::device_vector<T>::begin;
    using thrust::device_vector<T>::end;
    using thrust::device_vector<T>::push_back;
    using thrust::device_vector<T>::operator=;
    using thrust::device_vector<T>::operator[];
    
    using thrust::device_vector<T>::device_vector;
    using thrust::device_vector<T>::size;
    using thrust::device_vector<T>::reserve;
    using thrust::device_vector<T>::insert;


    /* Scalor */
    device_vector()
    : thrust::device_vector<T>::device_vector(1) 
    {
        row_ = 1;
        col_ = 1;
    }

    /* Vector */
    device_vector(int sz, VectorType type)
    : thrust::device_vector<T>::device_vector(sz) 
    {
        if(type == VectorType::Row){
            row_ = sz;
            col_ = 1;    
        }else if(type == VectorType::Col){
            row_ = 1;
            col_ = sz;
        }
    }

    /* Matrix */
    device_vector(int r, int c)
    : thrust::device_vector<T>::device_vector(r*c) 
    {
        row_ = r;
        col_ = c;
    }

    /* device vector */
    device_vector(thrust::device_vector<T> &d, VectorType type)
    :thrust::device_vector<T>::device_vector(d)
    {
        if(type == VectorType::Row){
            row_ = d.size();
            col_ = 1;
        }else if(type == VectorType::Col){
            row_ = 1;
            col_ = d.size();
        }
    }

    /* copy constructor */
    device_vector(mt::device_vector<T> &a)
    : thrust::device_vector<T>::device_vector(a)
    {
       // type_ = a.type_;
        row_ = a.row();
        col_ = a.col();
    }

    /* host vector */
    device_vector(thrust::host_vector<T> &h, VectorType type)
    : thrust::device_vector<T>::device_vector(h)
    {
        if(type == VectorType::Row){
            row_ = h.size();
            col_ = 1;
        }else if(type == VectorType::Col){
            row_ = 1;
            col_ = h.size();
        }        
    }

    /* Matrix */
    device_vector(thrust::device_vector<T> &d, int row, int col)
    : thrust::device_vector<T>::device_vector(d)
    {
        row_ = row;
        col_ = col;
    }

    /* host vector Matrix 1 */
    device_vector(thrust::host_vector<T> &h, int r, int c)
    :thrust::device_vector<T>::device_vector(h)
    {
        row_ = r; 
        col_ = c;
    }

    /* host vector Matrix 2 */
    device_vector(thrust::host_vector<thrust::host_vector<T>> &hvv)
    :  thrust::device_vector<T>::device_vector( transpose(line(hvv), hvv.size(), hvv[0].size()))
    {
        row_ = hvv.size();
        col_ = hvv[0].size();

    }

    int row(){
        return row_;
    }

    int col(){
        return col_;
    }

    void setRow(int r){
        row_ = r;
    }

    void setCol(int c){
        col_ = c;
    }

    
private:
    int row_;
    int col_;
};

template<typename T>
class device_queue{
public:
    size_t size(){
        return vec.size();
    }

    void push(T in){
        vec.push_back(in);
    }

    auto operator[](int i){
        T p = vec[i];     // address
        return *p;
    }

private:
    thrust::device_vector<T> vec;

};

};

template <typename T>
void print(mt::device_vector<T>&& d)
{
    std::cout << "------" << std::endl;    
    thrust::host_vector<T> out(transpose(d, d.col(), d.row()));   
    for(int i=0; i<d.row(); ++i){
        thrust::copy(out.begin() + d.col() * i, out.begin() + d.col() * (i + 1) , std::ostream_iterator<T>(std::cout, " ")); 
        std::cout << std::endl;
    }
}

template <typename T>
void print(mt::device_vector<T>& d)
{
    print(std::move(d));
}

template <typename T>
auto operator+(mt::device_vector<T> &&a, mt::device_vector<T> &&b){
    assert(a.size() == b.size());

    mt::device_vector<T> ret(b);
    thrust::transform(a.begin(), a.end(), b.begin(), ret.begin(), thrust::plus<T>());

    return ret;
}

template <typename T>
auto operator+(mt::device_vector<T> &a, mt::device_vector<T> &b){
    mt::device_vector<T> ret = std::move(a) + std::move(b);
    
    return ret;
}

template <typename T>
auto operator+(mt::device_vector<T> &&a, mt::device_vector<T> &b){
    mt::device_vector<T> ret = std::move(a) + std::move(b);
    
    return ret;
}

template <typename T>
auto operator+(mt::device_vector<T> &a, mt::device_vector<T> &&b){
    mt::device_vector<T> ret = std::move(a) + std::move(b);
    
    return ret;
}

template<typename T, typename U>
struct add_s
{   
    const U val_;
        
    add_s(U p)
        : val_{p} {}

    __host__ __device__
    inline T operator()(T& a) const
    {      
        return a + val_;
    }
};

template <typename T, typename U>
auto operator+(mt::device_vector<T> &&a, U b){
    mt::device_vector<T> ret(a);
    //auto add = add_s<T>(b);
    thrust::transform(a.begin(), a.end(), ret.begin(), add_s<T, U>(b));
    
    return ret;
}

template <typename T, typename U>
auto operator+(mt::device_vector<T> &a, U b){
    mt::device_vector<T> ret = std::move(a) + b;
    return ret;
}

template <typename T, typename U>
auto operator+(U a, mt::device_vector<T> &&b){
    mt::device_vector<T> ret(b);
    //auto add = add_s<T, U>(a);
    thrust::transform(b.begin(), b.end(), ret.begin(), add_s<T, U>(a));
    
    return ret;
}

template <typename T, typename U>
auto operator+(U a, mt::device_vector<T> &b){
    mt::device_vector<T> ret = a + std::move(b);
    return ret;
}

/* sub */
template <typename T>
auto operator-(mt::device_vector<T> &&a, mt::device_vector<T> &&b){

    assert(a.size() == b.size());

    mt::device_vector<T> ret(b);
    thrust::transform(a.begin(), a.end(), b.begin(), ret.begin(), thrust::minus<T>());

    return ret;
}

template <typename T>
auto operator-(mt::device_vector<T> &a, mt::device_vector<T> &b){
    mt::device_vector<T> ret = std::move(a) + std::move(b);
    
    return ret;
}

template <typename T>
auto operator-(mt::device_vector<T> &&a, mt::device_vector<T> &b){
    mt::device_vector<T> ret = std::move(a) - std::move(b);
    
    return ret;
}

template <typename T>
auto operator-(mt::device_vector<T> &a, mt::device_vector<T> &&b){
    mt::device_vector<T> ret = std::move(a) - std::move(b);
    
    return ret;
}

template<typename T, typename U>
struct sub_s
{   
    const U val_;
        
    sub_s(U p)
        : val_{p} {}

    __host__ __device__
    inline T operator()(T& a) const
    {      
        return a - val_;
    }
};

template <typename T, typename U>
auto operator-(mt::device_vector<T> &&a, U b){
    mt::device_vector<T> ret(a);
    //auto sub = sub_s<T>(b);
    thrust::transform(a.begin(), a.end(), ret.begin(), sub_s<T, U>(b));
    
    return ret;
}

template <typename T, typename U>
auto operator-(mt::device_vector<T> &a, U b){
    mt::device_vector<T> ret = std::move(a) - b;
    return ret;
}


template<typename T, typename U>
struct sub_s_2
{   
    const U val_;
        
    sub_s_2(U p)
        : val_{p} {}

    __host__ __device__
    inline T operator()(T& a) const
    {      
        return val_ - a;
    }
};


template <typename T, typename U>
auto operator-(U a, mt::device_vector<T> &&b){
    mt::device_vector<T> ret(b);
    //auto sub = sub_s_2<T>(a);
    thrust::transform(b.begin(), b.end(), ret.begin(), sub_s_2<T, U>(a));
    
    return ret;
}

template <typename T, typename U>
auto operator-(U a, mt::device_vector<T> &b){
    mt::device_vector<T> ret = a - std::move(b);
    return ret;
}

/* mul */
template <typename T>
auto operator*(mt::device_vector<T> &&a, mt::device_vector<T> &&b){

    assert(a.size() == b.size());
    mt::device_vector<T> ret(b);

    thrust::transform(a.begin(), a.end(), b.begin(), ret.begin(), thrust::multiplies<T>());

    return ret;
}

template <typename T>
auto operator*(mt::device_vector<T> &a, mt::device_vector<T> &b){

    mt::device_vector<T> ret = std::move(a) * std::move(b);

    return ret;
}

template <typename T>
auto operator*(mt::device_vector<T> &&a, mt::device_vector<T> &b){

    mt::device_vector<T> ret = std::move(a) * std::move(b);

    return ret;

}

template <typename T>
auto operator*(mt::device_vector<T> &a, mt::device_vector<T> &&b){

    mt::device_vector<T> ret = std::move(a) * std::move(b);

    return ret;
}

template<typename T, typename U>
struct mul_s
{   
    const U val_;
        
    mul_s(U p)
        : val_{p} {}

    __host__ __device__
    inline T operator()(T& a) const
    {      
        return val_ * a;
    }
};

template <typename T, typename U>
auto operator*(mt::device_vector<T> &&a, U b){
    mt::device_vector<T> ret(a);
    //auto mul = mul_s<T>(b);
    thrust::transform(a.begin(), a.end(), ret.begin(), mul_s<T, U>(b));
    
    return ret;
}

template <typename T, typename U>
auto operator*(mt::device_vector<T> &a, U b){
    mt::device_vector<T> ret = std::move(a) * b;
    return ret;
}


template <typename T , typename U>
auto operator*(U a, mt::device_vector<T> &&b){
    mt::device_vector<T> ret(b);
    //auto mul = mul_s<T>(a);
    thrust::transform(b.begin(), b.end(), ret.begin(), mul_s<T, U>(a));
    
    return ret;
}

template <typename T, typename U>
auto operator*(U a, mt::device_vector<T> &b){
    mt::device_vector<T> ret = a * std::move(b);
    return ret;
}


/* A -> 1/A */
template<typename T>
struct opp_s
{   
    __host__ __device__
    inline T operator()(T& a) const
    {      
        return 1/a;
    }
};

template<typename T>
auto OPP(mt::device_vector<T> &&a){
    mt::device_vector<T> ret(a);    
    auto opp = opp_s<T>();
    thrust::transform(a.begin(), a.end(), ret.begin(), opp);

    return ret;

}

template<typename T>
auto OPP(mt::device_vector<T> &a){
    mt::device_vector<T> ret = OPP(std::move(a));
    return ret;
}

/* div */
template <typename T>
auto operator/(mt::device_vector<T> &&a, mt::device_vector<T> &&b){
    assert(a.size() == b.size());

    mt::device_vector<T> ret(b);
    thrust::transform(a.begin(), a.end(), b.begin(), ret.begin(), thrust::divides<T>());
    return ret;
}

template <typename T>
auto operator/(mt::device_vector<T> &&a, mt::device_vector<T> &b){
    
    mt::device_vector<T> ret = std::move(a) / std::move(b);
    return ret;
}

template <typename T>
auto operator/(mt::device_vector<T> &a, mt::device_vector<T> &&b){


    mt::device_vector<T> ret = std::move(a) / std::move(b);
    return ret;
}

template <typename T>
auto operator/(mt::device_vector<T> &a, mt::device_vector<T> &b){
    
    mt::device_vector<T> ret = std::move(a) / std::move(b);
    return ret;
}

template<typename T, typename U>
struct div_s
{   
    const U val_;
        
    div_s(U p)
        : val_{p} {}

    __host__ __device__
    inline T operator()(T& a) const
    {      
        return a / val_;
    }
};

template <typename T, typename U>
auto operator/(mt::device_vector<T> &&a, U b){
    mt::device_vector<T> ret(a);
    //auto div = div_s<T>(b);
    thrust::transform(a.begin(), a.end(), ret.begin(), div_s<T, U>(b));

    return ret;
}

template <typename T, typename U>
auto operator/(mt::device_vector<T> &a, U b){
    mt::device_vector<T> ret = std::move(a) / b;
    return ret;
}

template<typename T, typename U>
struct div_s_2
{   
    const U val_;
        
    div_s_2(U p)
        : val_{p} {}

    __host__ __device__
    inline T operator()(T& a) const
    {      
        return val_ / a;
    }
};


template <typename T, typename U>
auto operator/(U a, mt::device_vector<T> &&b){
    mt::device_vector<T> ret(b);
    //auto div = div_s_2<T>(a);
    thrust::transform(b.begin(), b.end(), ret.begin(), div_s_2<T, U>(a));
    
    return ret;
}

template <typename T, typename U>
auto operator/(U a, mt::device_vector<T> &b){
    mt::device_vector<T> ret = a / std::move(b);
    return ret;
}


/*  Matrix dot */
template <typename T>
struct Dotto_s
{
    T *A, *B;
    int arow, acol, bcol;

    Dotto_s(T*_A, T*_B, int _arow, int _acol, int _bcol): A(_A), B(_B), arow(_arow), acol(_acol), bcol(_bcol) {}

    __host__ __device__
    T operator()(size_t index){
        T sum = 0;
        int col = index / arow;         /* B target col */
        //int row = index % arow;       /* A target row */ 
        int row = index -(col * arow);

        for (int i = 0; i < acol; ++i)
           sum += A[row + acol * i] * B[col * arow + i];
        return sum;
    }
};

template<typename T>
auto mDot(mt::device_vector<T> &&a, mt::device_vector<T> &&b){
    
    assert(a.col() == b.row());

    int sz = a.row() * b.col();
    mt::device_vector<T> ret(a.row(), b.col());

    thrust::transform(
        thrust::counting_iterator<int>(0), 
        thrust::counting_iterator<int>(sz), 
        ret.begin(), 
        Dotto_s(a.data().get(), b.data().get(), a.row(), a.col(), b.col())
    );

#if 0
    if constexpr (std::is_same<T, float>{}) {
        float alpha = 1.0;
        float beta = 0.0;
        cublasSgemm(    
            g_handle,
            CUBLAS_OP_N,                            //行列A 転置有無
            CUBLAS_OP_N,                            //行列B 転置有無
            a.row(),                                // 行列Aの行数
            b.row(),                                // 行列Bの列数
            a.col(),                                // 行列Aの列数(=行列Ｂの行数)
            &alpha,                                 // 行列の積に掛ける値(なければ1)
            //a.data().get(),                       // 行列A
            thrust::raw_pointer_cast(a.data()),
            a.row(),                                // 行列Aの行数
            //b.data().get(),                       // 行列B
            thrust::raw_pointer_cast(b.data()),
            b.row(),                                // 行列Bの行数
            &beta,                                  // 行列Cに掛けるスカラ値(なければ0)
            //ret.data().get(),                     // 行列Cの初期値 兼 出力先
            thrust::raw_pointer_cast(ret.data()),
            ret.row()                               // 行列Cの行数
            
        );

    }else if constexpr(std::is_same<T, double>{}){
        double alpha = 1.0;
        double beta = 0.0;
        cublasDgemm(    
            g_handle,
            CUBLAS_OP_N,        //行列A 転置有無
            CUBLAS_OP_N,        //行列B 転置有無
            a.row(),            // 行列Aの行数
            b.row(),            // 行列Bの列数
            a.col(),            // 行列Aの列数(=行列Ｂの行数)
            &alpha,             // 行列の積に掛ける値(なければ1)
            a.data().get(),     // 行列A
            a.row(),            // 行列Aの行数
            b.data().get(),     // 行列B
            b.row(),            // 行列Bの行数
            &beta,              // 行列Cに掛けるスカラ値(なければ0)
            ret.data().get(),   // 行列Cの初期値 兼 出力先
            ret.row()           // 行列Cの行数
        );
    }
#endif
    return ret;
}


template<typename T>
auto mDot(mt::device_vector<T> &a, mt::device_vector<T> &b){
    mt::device_vector<T> ret = mDot(std::move(a), std::move(b));

    return ret;
}

template<typename T>
auto mDot(mt::device_vector<T> &&a, mt::device_vector<T> &b){
    mt::device_vector<T> ret = mDot(std::move(a), std::move(b));
    return ret;
}

template<typename T>
auto mDot(mt::device_vector<T> &a, mt::device_vector<T> &&b){
    mt::device_vector<T> ret = mDot(std::move(a), std::move(b)); 
    return ret;
}

/* vector inner product */

template<typename T>
T vDot(mt::device_vector<T> &&a, mt::device_vector<T> &&b){
    assert(a.size() == b.size());

    auto c = a * b;
    T ret = thrust::reduce(c.begin(), c.end(), 0.0, thrust::plus<T>());

    return ret;

#if 0
    cublasStatus_t status;
    //mt::device_vector<T> s;  /* Scalor */
    thrust::device_vector<T> s(1);

    if constexpr (std::is_same<T, float>{}) {    
        status = cublasSdot(
            g_handle,
            a.size(),
            a.data().get(),    // ベクトルA
            1,
            b.data().get(),    // ベクトルB
            1,
            s.data().get()
        );

    }else if constexpr(std::is_same<T, double>{}){
        status = cublasDdot(
            g_handle,
            a.size(),
            a.data().get(),    // ベクトルA
            1,
            b.data().get(),    // ベクトルB
            1,
            s.data().get()
        );

    }

    assert( status == CUBLAS_STATUS_SUCCESS );

    return s.data()[0];
#endif

}


template<typename T>
auto vDot(mt::device_vector<T> &a, mt::device_vector<T> &&b){
    auto ret = vDot(std::move(a), std::move(b));
    return ret;
}

template<typename T>
auto vDot(mt::device_vector<T> &&a, mt::device_vector<T> &b){
    auto ret = vDot(std::move(a), std::move(b));
    return ret;
}

template<typename T>
auto vDot(mt::device_vector<T> &a, mt::device_vector<T> &b){
    auto ret = vDot(std::move(a), std::move(b));
    return ret;
}

#if 0
/* Matrix dot */
template<typename T>
auto dot(mt::device_vector<T> &&a, bool T1 , mt::device_vector<T> &&b, bool T2 ){
    
    cublasOperation_t t1, t2;
    int a_row, a_col, b_row, b_col;

    if(T1 == false && T2 == false){
        assert(a.col() == b.row());
        t1 = CUBLAS_OP_N; 
        t2 = CUBLAS_OP_N; 
        a_row = a.row();
        a_col = a.col();
        b_row = b.row();
        b_col =  b.col();

    }else if(T1 == true && T2 == false){
        assert(a.row() == b.row());
        t1 = CUBLAS_OP_T; 
        t2 = CUBLAS_OP_N;
        a_row = a.col();
        a_col = a.row();
        b_row = b.row();
        b_col =  b.col();

    }else if(T1 == false && T2 == true){
        assert(a.col() == b.col());
        t1 = CUBLAS_OP_N;
        t2 = CUBLAS_OP_T;
        a_row = a.row();
        a_col = a.col();
        b_row = b.col();
        b_col =  b.row(); 
    }else if(T1 == true && T2 == true){
        assert(a.row() == b.col());
        t1 = CUBLAS_OP_T;
        t2 = CUBLAS_OP_T;
        a_row = a.col();
        a_col = a.row();
        b_row = b.col();
        b_col =  b.row(); 
    }    

    mt::device_vector<T> ret(a_row, b_col);

    if constexpr (std::is_same<T, float>{}) {
        float alpha = 1.0;
        float beta = 0.0;
        cublasSgemm(    
            g_handle,
            t1,                 // 行列A 転置有無
            t2,                 // 行列B 転置有無
            a_row,              // 行列Aの行数
            b_row,              // 行列Bの列数
            a_col,              // 行列Aの列数(=行列Ｂの行数)
            &alpha,             // 行列の積に掛ける値(なければ1)
            a.data().get(),     // 行列A
            a_row,              // 行列Aの行数
            b.data().get(),     // 行列B
            b_row,              // 行列Bの行数
            &beta,              // 行列Cに掛けるスカラ値(なければ0)
            ret.data().get(),   // 行列Cの初期値 兼 出力先
            ret.row()           // 行列Cの行数
        );

    }else if constexpr(std::is_same<T, double>{}){
        double alpha = 1.0;
        double beta = 0.0;
        cublasDgemm(    
            g_handle,
            CUBLAS_OP_N,        // 行列A 転置有無
            CUBLAS_OP_N,        // 行列B 転置有無
            a_row,              // 行列Aの行数
            b_row,              // 行列Bの列数
            a_col,              // 行列Aの列数(=行列Ｂの行数)
            &alpha,             // 行列の積に掛ける値(なければ1)
            a.data().get(),     // 行列A
            a_row,              // 行列Aの行数
            b.data().get(),     // 行列B
            b_row,              // 行列Bの行数
            &beta,              // 行列Cに掛けるスカラ値(なければ0)
            ret.data().get(),   // 行列Cの初期値 兼 出力先
            ret.row()           // 行列Cの行数
        );
    }

    return ret;
}

template<typename T>
auto dot(mt::device_vector<T> &&a, bool T1 , mt::device_vector<T> &b, bool T2 ){
    mt::device_vector<T> ret = dot(std::move(a), T1, std::move(b), T2); 
    return ret;

}

template<typename T>
auto dot(mt::device_vector<T> &a, bool T1 , mt::device_vector<T> &&b, bool T2 ){
    mt::device_vector<T> ret = dot(std::move(a), T1, std::move(b), T2); 
    return ret;

}


template<typename T>
auto dot(mt::device_vector<T> &a, bool T1 , mt::device_vector<T> &b, bool T2 ){
    mt::device_vector<T> ret = dot(std::move(a), T1, std::move(b), T2); 
    return ret;

}
#endif


/* Cos */
template<typename T>
struct Cos_s
{   

    __host__ __device__
    inline T operator()(T& a) const
    {      
        if constexpr (std::is_same<T, float>{}) {
            return cosf(a);
        }
        else  if constexpr (std::is_same<T, double>{}) {
            return cos(a);
        }
    }
};

template <typename T>
auto cos(mt::device_vector<T> &&a){
    mt::device_vector<T> ret(a);
    //auto Cos = Cos_s<T>();
    thrust::transform(a.begin(), a.end(), ret.begin(), Cos_s<T>());
    
    return ret;
}

template <typename T>
auto cos(mt::device_vector<T> &a){
    mt::device_vector<T> ret = cos(std::move(a));
    
    return ret;
}

/* Sin */
template<typename T>
struct Sin_s
{   
    __host__ __device__
    inline T operator()(T& a) const
    {      
        if constexpr (std::is_same<T, float>{}) {
            return sinf(a);
        }
        else  if constexpr (std::is_same<T, double>{}) {
            return sin(a);
        }
    }
};

template <typename T>
auto sin(mt::device_vector<T> &&a){
    mt::device_vector<T> ret(a);
    //auto Sin = Sin_s<T>();
    thrust::transform(a.begin(), a.end(), ret.begin(), Sin_s<T>());
    
    return ret;
}

template <typename T>
auto sin(mt::device_vector<T> &a){
    mt::device_vector<T> ret = sin(std::move(a));
    
    return ret;
}

/* Tan */
template<typename T>
struct Tan_s
{   
    __host__ __device__
    inline T operator()(T& a) const
    {      
        if constexpr (std::is_same<T, float>{}) {
            return tanf(a);
        }
        else  if constexpr (std::is_same<T, double>{}) {
            return tan(a);
        }
    }
};

template <typename T>
auto tan(mt::device_vector<T> &&a){
    mt::device_vector<T> ret(a);
    //auto Tan = Tan_s<T>();
    thrust::transform(a.begin(), a.end(), ret.begin(), Tan_s<T>());
    
    return ret;
}

template <typename T>
auto tan(mt::device_vector<T> &a){
    mt::device_vector<T> ret = tan(std::move(a));
    
    return ret;
}


/* Exp */
template<typename T>
struct Exp_s
{   
    __host__ __device__
    inline T operator()(T& a) const
    {      
        if constexpr (std::is_same<T, float>{}) {
            return expf(a);
        }
        else  if constexpr (std::is_same<T, double>{}) {
            return exp(a);
        }
    }
};

template <typename T>
auto exp(mt::device_vector<T> &&a){
    mt::device_vector<T> ret(a);
    //auto Exp = Exp_s<T>();
    thrust::transform(a.begin(), a.end(), ret.begin(), Exp_s<T>());
    
    return ret;
}

template <typename T>
auto exp(mt::device_vector<T> &a){
    mt::device_vector<T> ret = exp(std::move(a));
    
    return ret;
}

/* log */
template<typename T>
struct Log_s
{   
    Log_s(){};

    __host__ __device__
    inline T operator()(T& a) const
    {   
        if constexpr (std::is_same<T, float>{}) {
            return logf(a);
        }
        else  if constexpr (std::is_same<T, double>{}) {
            return log(a);
        } 
    }
};

template <typename T>
auto log(mt::device_vector<T> &&a){
    mt::device_vector<T> ret(a);
    //auto Log = Log_s<T>();
    thrust::transform(a.begin(), a.end(), ret.begin(), Log_s<T>());
    
    return ret;
}

template <typename T>
auto log(mt::device_vector<T> &a){
    mt::device_vector<T> ret = log(std::move(a));
    
    return ret;
}

/* sqrt */
template<typename T>
struct Sqrt_s
{   
    Sqrt_s(){}

    __host__ __device__
    inline T operator()(T& a) const
    {      
        if constexpr (std::is_same<T, float>{}) {
            return sqrtf(a);
        }
        else  if constexpr (std::is_same<T, double>{}) {
            return sqrt(a);
        }
    }
};

template <typename T>
auto sqrt(mt::device_vector<T> &&a){
    mt::device_vector<T> ret(a);
    //auto Sqrt = Sqrt_s<T>();
    thrust::transform(a.begin(), a.end(), ret.begin(), Sqrt_s<T>());
    
    return ret;
}

template <typename T>
auto sqrt(mt::device_vector<T> &a){
    mt::device_vector<T> ret = sqrt(std::move(a));
    
    return ret;
}

/* pow */
template<typename T, typename U>
struct Pow_s
{   
    const U val_;
        
    Pow_s(U p)
        : val_{p} {}
  
    __host__ __device__
    inline T operator()(T& a) const
    {      
        if constexpr (std::is_same<T, float>{}) {
            return powf(a);
        }
        else  if constexpr (std::is_same<T, double>{}) {
            return pow(a);
        }
    }
};

template <typename T, typename U>
auto pow(mt::device_vector<T> &&a, U b){
    mt::device_vector<T> ret(a);
    //auto Pow = Pow_s<T,U>(b);
    thrust::transform(a.begin(), a.end(), ret.begin(), Pow_s<T,U>(b));
    
    return ret;
}

template <typename T, typename U>
auto pow(mt::device_vector<T> &a, U b){
    mt::device_vector<T> ret = pow(std::move(a), b);
    
    return ret;
}

/* resize */
template<typename T>
auto resize(mt::device_vector<T> &&a, int r, int c){
    assert(a.size() == r*c);
    mt::device_vector<T> ret = a;
    ret.setRow = r;
    ret.setCol = c;
    return ret;

}

template<typename T>
auto resize(mt::device_vector<T> &a, int r, int c){
    assert(a.size() == r*c);
    mt::device_vector<T> ret = resize(std::move(a), r, c);
    
    return ret;
}

struct rm2cm_idx_functor : public thrust::unary_function<int, int>
{
  int r;
  int c;

  rm2cm_idx_functor(int _r, int _c) : r(_r), c(_c) {};

  __host__ __device__
  int operator() (int idx)  {
    unsigned mt_r = idx/c;
    unsigned mt_c = idx%c;
    return (mt_c * r) + mt_r;
  }
};

template<typename T>
void copyMat(thrust::device_ptr<T> src, thrust::device_ptr<T> dst, unsigned src_rows, unsigned src_cols, unsigned dst_rows, unsigned offset){
   thrust::copy_n(
                thrust::make_permutation_iterator(src, 
                thrust::make_transform_iterator(thrust::counting_iterator<int>(0), 
                rm2cm_idx_functor(src_rows, src_cols))), 
                src_rows*src_cols, 
                thrust::make_permutation_iterator(dst, 
                thrust::make_transform_iterator(thrust::counting_iterator<int>(offset), 
                rm2cm_idx_functor(dst_rows, src_cols)))
                );
}


/* Concat Horizon */
template<typename T>
auto concatH(mt::device_vector<T> &&a, mt::device_vector<T> &&b){
    assert(a.row() == b.row());
    
    mt::device_vector<T> ret(a);

    ret.reserve(a.size() + b.size());
    ret.setCol(a.col() + b.col());
    //ret.insert(b);
    ret.insert(ret.end(), b.begin(), b.end());

    return ret;
}

template<typename T>
auto concatH(mt::device_vector<T> &&a, mt::device_vector<T> &b){
    
    mt::device_vector<T> ret = concatH(std::move(a), std::move(b));
    return ret;
}

template<typename T>
auto concatH(mt::device_vector<T> &a, mt::device_vector<T> &&b){
    
    mt::device_vector<T> ret = concatH(std::move(a), std::move(b));
    return ret;
}


template<typename T>
auto concatH(mt::device_vector<T> &a, mt::device_vector<T> &b){
    
    mt::device_vector<T> ret = concatH(std::move(a), std::move(b));
    return ret;
}


/* Concat Vertical */
template<typename T>
auto concatV(mt::device_vector<T> &&a, mt::device_vector<T> &&b){
    assert(a.col() == b.col());

    mt::device_vector<T> ret(a.row() + b.row(), a.col());

    int offset = 0;
    copyMat(a.data(), ret.data(), a.row(), a.col(), ret.row(), offset);
    offset = a.col()*a.row();
    copyMat(b.data(), ret.data(), b.row(), b.col(), ret.row(), offset);

    return ret;
}


template<typename T>
auto concatV(mt::device_vector<T> &a, mt::device_vector<T> &b){
   
    mt::device_vector<T> ret = concatV(std::move(a), std::move(b));
    
    return ret;
}


template<typename T>
auto concatV(mt::device_vector<T> &&a, mt::device_vector<T> &b){
   
    mt::device_vector<T> ret = concatV(std::move(a), std::move(b));
    
    return ret;
}

template<typename T>
auto concatV(mt::device_vector<T> &a, mt::device_vector<T> &&b){
   
    mt::device_vector<T> ret = concatV(std::move(a), std::move(b));
    
    return ret;
}

/* TransPosition  */
template<typename T>
auto TP(mt::device_vector<T> &&a){
    
    mt::device_vector<T> ret(a);
    ret.trans();

    return ret;
}

template<typename T>
auto TP(mt::device_vector<T> &a){
    mt::device_vector<T> ret = TP(std::move(a));
    
    return ret;
}


/* reLU */
template<typename T>
struct Relu_s
{   
    __host__ __device__
    inline T operator()(T& a) const
    {      
        return (a > 0) ? a : (T)0;
    }
};

template <typename T>
auto relu(mt::device_vector<T> &&a){
    mt::device_vector<T> ret(a);
    //auto Relu = Relu_s<T>();
    thrust::transform(a.begin(), a.end(), ret.begin(), Relu_s<T>());
    
    return ret;
}

template <typename T>
auto relu(mt::device_vector<T> &a){
    mt::device_vector<T> ret = relu(std::move(a));
    
    return ret;
}

/* Sigmoid */
template<typename T>
struct Sig_s
{   
    __host__ __device__
    inline T operator()(T& a) const
    {      
        if constexpr (std::is_same<T, float>{}) {
            return 1/(1 + expf(-a));
        }
        else  if constexpr (std::is_same<T, double>{}) {
            return 1/(1 + exp(-a));
        }
    }
};

template <typename T>
auto sigmoid(mt::device_vector<T> &&a){
    mt::device_vector<T> ret(a);
    //auto Sig = Sig_s<T>();
    thrust::transform(a.begin(), a.end(), ret.begin(), Sig_s<T>());
    
    return ret;
}

template <typename T>
auto sigmoid(mt::device_vector<T> &a){
    mt::device_vector<T> ret = sigmoid(std::move(a));
    
    return ret;
}

/* Summation Vertical direction */

template <typename T>
struct SumV_s
{
    T *A;
    int arow, acol;

    SumV_s(T*_A, int _arow, int _acol): A(_A), arow(_arow), acol(_acol) {}

    __host__ __device__
    T operator()(size_t index){
        T sum = 0;

        for (int i = 0; i < arow; ++i)
           sum += A[index * arow + i];
        return sum;
    }
};


template <typename T>
auto sumV(mt::device_vector<T> &&a){

    mt::device_vector<T> ret(1, a.col());
    //thrust::device_vector<T> d_ones(a.row(), 1.0);
    //mt::device_vector<T> ret(a.row(), b.col());

    thrust::transform(
        thrust::counting_iterator<int>(0), 
        thrust::counting_iterator<int>(a.col()), 
        ret.begin(), 
        SumV_s(a.data().get(), a.row(), a.col())
    );

#if 0
    T alpha = 1.0;
    T beta  = 0.0;  
    
    if constexpr (std::is_same<T, float>{}) {
          cublasSgemv(g_handle, 
                        CUBLAS_OP_T, 
                        a.row(), 
                        a.col(), 
                        &alpha, 
                        a.data().get(), 
                        a.row(), 
                        d_ones.data().get(), 
                        1, 
                        &beta, 
                        ret.data().get(), 
                        1
                    );

    }
    else  if constexpr (std::is_same<T, double>{}) {
          cublasDgemv(g_handle, 
                        CUBLAS_OP_T, 
                        a.row(), 
                        a.col(), 
                        &alpha, 
                        a.data().get(), 
                        a.row(), 
                        d_ones.data().get(), 
                        1, 
                        &beta, 
                        ret.data().get(), 
                        1
                    );

    }
#endif
    return ret;

}

template <typename T>
auto sumV(mt::device_vector<T> &a){
    
    return sumV(std::move(a));

}

/* Summation Horizontal direction */
template <typename T>
struct SumH_s
{
    T *A;
    int arow, acol;

    SumH_s(T*_A, int _arow, int _acol): A(_A), arow(_arow), acol(_acol) {}

    __host__ __device__
    T operator()(size_t index){
        T sum = 0;

        for (int i = 0; i < acol; ++i)
           sum += A[index + arow * i];
        return sum;
    }
};

template <typename T>
auto sumH(mt::device_vector<T> &&a){

    mt::device_vector<T> ret(a.row(), 1);
    //thrust::device_vector<T> d_ones(a.col(), 1.0);

    thrust::transform(
        thrust::counting_iterator<int>(0), 
        thrust::counting_iterator<int>(a.row()), 
        ret.begin(), 
        SumH_s(a.data().get(), a.row(), a.col())
    );

#if 0
    T alpha = 1.0;
    T beta  = 0.0;  
          
    if constexpr (std::is_same<T, float>{}) {
          cublasSgemv(g_handle, 
                        CUBLAS_OP_N, 
                        a.row(), 
                        a.col(), 
                        &alpha, 
                        a.data().get(), 
                        a.row(), 
                        thrust::raw_pointer_cast(d_ones.data()), 
                        1, 
                        &beta, 
                        ret.data().get(), 
                        1
                    );

    }
    else if constexpr (std::is_same<T, double>{}) {
          cublasDgemv(g_handle, 
                        CUBLAS_OP_N, 
                        a.row(), 
                        a.col(), 
                        &alpha, 
                        a.data().get(), 
                        a.row(), 
                        thrust::raw_pointer_cast(d_ones.data()), 
                        1, 
                        &beta, 
                        ret.data().get(), 
                        1
                    );

    }

#endif

    return ret;
}

template <typename T>
auto sumH(mt::device_vector<T> &a){
    return sumH(std::move(a));
}

/* |a1| + |a2| + |a3| +   */
template<typename T>
struct Abs_s
{   

    __host__ __device__
    inline T operator()(T a) const
    {      
        if constexpr (std::is_same<T, float>{}) {
            return fabsf(a);
        }
        else  if constexpr (std::is_same<T, double>{}) {
            return fabs(a);
        }
    }
};

template<typename T>
T abs(mt::device_vector<T> &&a){

    //mt::device_vector<T> s;  /* Scalor */

    T ret = thrust::transform_reduce(a.begin(), a.end(), Abs_s<T>(), 0.0, thrust::plus<T>());

#if 0    
    thrust::device_vector<T> s(1);
    cublasStatus_t status;

    if constexpr (std::is_same<T, float>{}) {
        status = cublasSasum(
            g_handle,
            a.size(),
            a.data().get(),
            1,
            s.data().get()
        );
    }else if constexpr (std::is_same<T, double>{}) {
        status = cublasDasum(
            g_handle,
            a.size(),
            a.data().get(),
            1,
            s.data().get()
        );
    }

    assert( status == CUBLAS_STATUS_SUCCESS );
    return s.data()[0];

#endif

    return ret;
    
}

template<typename T>
T abs(mt::device_vector<T> &a){
    return abs(std::move(a));
}

/* L1 */
template<typename T>
T L1(mt::device_vector<T> &&a){

    if constexpr (std::is_same<T, float>{}) {
        return sqrtf(abs(a));
    }else if constexpr (std::is_same<T, double>{}) {
        return sqrt(abs(a));
    }

}

template<typename T>
T L1(mt::device_vector<T> &a){
    return L1(std::move(a));
}

/* |a1|^2 + |a2|^2 + |a3|^2 +   */
template<typename T>
struct Square_s
{   

    __host__ __device__
    inline T operator()(T a) const
    {   
        return a * a;   
    }
};

template<typename T>
T abs2(mt::device_vector<T> &&a){
    //mt::device_vector<T> s;  /* Scalor */
    T ret = thrust::transform_reduce(a.begin(), a.end(), Square_s<T>(), 0.0, thrust::plus<T>());
    return ret;
}

template<typename T>
T abs2(mt::device_vector<T> &a){
    return abs2(std::move(a));
}

/* L2 */
template<typename T>
T L2(mt::device_vector<T> &&a){

    if constexpr (std::is_same<T, float>{}) {
        return sqrtf(abs2(a));
    }else if constexpr (std::is_same<T, double>{}) {
        return sqrt(abs2(a));
    }

#if 0
    //mt::device_vector<T> s;  /* Scalor */
    thrust::device_vector<T> s(1);
    cublasStatus_t status;

    if constexpr (std::is_same<T, float>{}) {
        status = cublasSnrm2(
            g_handle,
            a.size(),
            a.data().get(),
            1,
            s.data().get()
        );
    }else if constexpr (std::is_same<T, double>{}) {
        status = cublasDnrm2(
            g_handle,
            a.size(),
            a.data().get(),
            1,
            s.data().get()
        );
    }

    assert( status == CUBLAS_STATUS_SUCCESS );

    return s.data()[0];
#endif

}

template<typename T>
T L2(mt::device_vector<T> &a){
    return L2(std::move(a));
}


/* inverse matrix　*/
template<typename T>
struct One_s
{   
    size_t sz;

    One_s(int n): 
    sz(n)
    {}

    __host__ __device__
    inline T operator()(int index) const
    {   
        return (index % sz == 0) ? (T)1 : (T)0;
    }
};

template<typename T>
auto inv(mt::device_vector<T> &a){
    assert( a.col() == a.row() );

    int n = a.col();
    mt::device_vector<T> A(a);

    /* create identity matrix */
    //auto One = One_s<T>(n+1);

    mt::device_vector<T> B(
    thrust::make_transform_iterator(
        thrust::make_counting_iterator<int>(0), One_s<T>(n+1)),
    thrust::make_transform_iterator(
        thrust::make_counting_iterator<int>(n*n), One_s<T>(n+1))
    );

    B.setRow(n);
    B.setCol(n);

    cusolverStatus_t status;
    int worksize;

    if constexpr (std::is_same<T, float>{}) {
        status = cusolverDnSgetrf_bufferSize(
              g_handle2,
              n,                 // 行
              n,                 // 列
              A.data().get(),    // A
              n,                 // Aのヨコハバ
              &worksize
              );

        //std::cout << worksize << std::endl;
        assert( status == CUSOLVER_STATUS_SUCCESS );
        thrust::device_vector<T> workspace(worksize);
        thrust::device_vector<int> info(1);
        thrust::device_vector<int> pivot(n);
        //std::cout << info[0]<< std::endl;

        status = cusolverDnSgetrf(
              g_handle2,
              n,                         // 行
              n,                         // 列
              A.data().get(),            // A
              n,                         // Aのヨコハバ
              workspace.data().get(),
              pivot.data().get(),
              info.data().get()
              );
        //std::cout << info[0]<< std::endl;

        assert( status == CUSOLVER_STATUS_SUCCESS );
        status = cusolverDnSgetrs(
               g_handle2,
               CUBLAS_OP_N,
               n,                     // 行(=列)
               n,                     // 問題数
               A.data().get(),        // A
               n,                     // Aのヨコハバ
               pivot.data().get(),    // LU分解で得られたピボット
               B.data().get(),        // B
               n,                     // Bのヨコハバ
               info.data().get());

        assert( status == CUSOLVER_STATUS_SUCCESS );

    }else if constexpr (std::is_same<T, double>{}) {

        status = cusolverDnDgetrf_bufferSize(
              g_handle2,
              n,                 // 行
              n,                 // 列
              A.data().get(),    // A
              n,                 // Aのヨコハバ
              &worksize
              );

        //std::cout << worksize << std::endl;
        assert( status == CUSOLVER_STATUS_SUCCESS );
        thrust::device_vector<T> workspace(worksize);
        thrust::device_vector<int> info(1);
        thrust::device_vector<int> pivot(n);
        //std::cout << info[0]<< std::endl;

        status = cusolverDnDgetrf(
              g_handle2,
              n,                         // 行
              n,                         // 列
              A.data().get(),            // A
              n,                         // Aのヨコハバ
              workspace.data().get(),
              pivot.data().get(),
              info.data().get()
              );
        //std::cout << info[0]<< std::endl;

        assert( status == CUSOLVER_STATUS_SUCCESS );
        status = cusolverDnDgetrs(
               g_handle2,
               CUBLAS_OP_N,
               n,                     // 行(=列)
               n,                     // 問題数
               A.data().get(),        // A
               n,                     // Aのヨコハバ
               pivot.data().get(),    // LU分解で得られたピボット
               B.data().get(),        // B
               n,                     // Bのヨコハバ
               info.data().get());

        assert( status == CUSOLVER_STATUS_SUCCESS );
    }
    return B;
}


/* RANDOM */
template <typename T>
struct Rand_s
{
    __host__ __device__
    inline T operator ()(int index) const
    {
        thrust::default_random_engine rng;
        thrust::uniform_real_distribution<T> uniDist;
        rng.discard(index);
        return uniDist(rng);
    }
};


template <typename T>
auto rand(mt::device_vector<T> &&a){
    mt::device_vector<T> ret(a);
    thrust::counting_iterator<int> index_sequence_begin(0);
    
    //auto Rand = Rand_s<T>();

    thrust::transform(
        index_sequence_begin,
        index_sequence_begin + a.size(),
        ret.begin(),
        Rand_s<T>()
    );

    return ret;
}

template <typename T>
auto rand(mt::device_vector<T> &a){
    return rand(std::move(a));
}

template <typename T = float>
auto rand(int r, int c){
    int sz = r * c;
    mt::device_vector<T> ret(sz);
    ret.setRow(r);
    ret.setCol(c);

    thrust::counting_iterator<int> index_sequence_begin(0);
    
    //auto Rand = Rand_s<T>();

    thrust::transform(
        index_sequence_begin,
        index_sequence_begin + sz,
        ret.begin(),
        Rand_s<T>()
    );

    return ret;
}

int main(){


    cublasStatus_t status = cublasCreate(&g_handle);
    assert( status == CUBLAS_STATUS_SUCCESS );

    cusolverStatus_t status2 = cusolverDnCreate(&g_handle2);
    assert( status2 == CUSOLVER_STATUS_SUCCESS );

    cublasSetPointerMode(g_handle, CUBLAS_POINTER_MODE_DEVICE); 

    thrust::host_vector<float> y1 = {5,-6,1,2,-3,4};
    thrust::host_vector<float> y2 = {7,4,3,3,2,2};      

    thrust::host_vector<thrust::host_vector<float>> h1 = {
                                                             {1,4},
                                                             {9,16},
                                                            {19,116}
                                                        };


    thrust::host_vector<thrust::host_vector<float>> h2 = {
                                                             {1,4,5},
                                                             {9,16,1},
                                                             {1,4,5},
                                                             {9,16,1},
                                                             {1,4,5},
                                                             {9,16,1},
                                                             {1,4,5},
                                                             {9,16,1},
                                                        };



    mt::device_vector t3(y1,VectorType::Row);
    mt::device_vector t4(y2,VectorType::Row);

    mt::device_vector t1(h1);
    mt::device_vector t2(h2);

    std::cout << L1(t3) << std::endl;;
    std::cout << L2(t3) << std::endl;;

    print(cos(t2));

    std::cout << vDot(t3,t4) << std::endl;

    //print(rand(p[0]));
    //print(concatV(  p[0]*p[1]/2.0f -110.0f, p[0]*p[1]/2.0f - 100.0f));

    //auto a = p[0];
    //print(rand(p[0]*p[0]/2.0f -100.0f));
   
    //mt::device_vector<float> d1(z);
    
    //ThrustQueue<mt::device_vector<float>> q1;
    //q1.push(d1);
    
    //q1.push(&d1);
    
    //q1.push(&d1);

    //print(q1[0]);

    //print(*q1[0]);

    //ThrustQueue<ThrustQueue<mt::device_vector<float>*>*> q2;

    //q2.push(&q1); 
    
    //for(auto &a : q1){
    //    print(a);
    //}

    //mt::device_vector<float> d2(x);

    //auto p1 = cos(d2 / 100.0f); 

    cusolverDnDestroy(g_handle2);
    cublasDestroy(g_handle);

}