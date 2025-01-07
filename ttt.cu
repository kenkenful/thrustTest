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
#include <thrust/random.h>


namespace mt{
template<typename T> class device_vector;
};

cublasHandle_t g_handle;

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
    device_vector(device_vector<T> &a)
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
        thrust::copy(out.begin() + d.col() * i, out.begin() + d.col() * (i + 1) , std::ostream_iterator<float>(std::cout, " ")); 
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

template<typename T>
struct add_s
{   
    const T val_;
        
    add_s(T p)
        : val_{p} {}

    __host__ __device__
    inline T operator()(T& a) const
    {      
        return a + val_;
    }
};

template <typename T>
auto operator+(mt::device_vector<T> &&a, T b){
    mt::device_vector<T> ret(a);
    auto add = add_s<T>(b);
    thrust::transform(a.begin(), a.end(), ret.begin(), add);
    
    return ret;
}

template <typename T>
auto operator+(mt::device_vector<T> &a, T b){
    mt::device_vector<T> ret = std::move(a) + b;
    return ret;
}

template <typename T>
auto operator+(T a, mt::device_vector<T> &&b){
    mt::device_vector<T> ret(b);
    auto add = add_s<T>(a);
    thrust::transform(b.begin(), b.end(), ret.begin(), add);
    
    return ret;
}

template <typename T>
auto operator+(T a, mt::device_vector<T> &b){
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

template<typename T>
struct sub_s
{   
    const T val_;
        
    sub_s(T p)
        : val_{p} {}

    __host__ __device__
    inline T operator()(T& a) const
    {      
        return a - val_;
    }
};

template <typename T>
auto operator-(mt::device_vector<T> &&a, T b){
    mt::device_vector<T> ret(a);
    auto sub = sub_s<T>(b);
    thrust::transform(a.begin(), a.end(), ret.begin(), sub);
    
    return ret;
}

template <typename T>
auto operator-(mt::device_vector<T> &a, T b){
    mt::device_vector<T> ret = std::move(a) - b;
    return ret;
}



template<typename T>
struct sub_s_2
{   
    const T val_;
        
    sub_s_2(T p)
        : val_{p} {}

    __host__ __device__
    inline T operator()(T& a) const
    {      
        return val_ -a;
    }
};


template <typename T>
auto operator-(T a, mt::device_vector<T> &&b){
    mt::device_vector<T> ret(b);
    auto sub = sub_s_2<T>(a);
    thrust::transform(b.begin(), b.end(), ret.begin(), sub);
    
    return ret;
}

template <typename T>
auto operator-(T a, mt::device_vector<T> &b){
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

template<typename T>
struct mul_s
{   
    const T val_;
        
    mul_s(T p)
        : val_{p} {}

    __host__ __device__
    inline T operator()(T& a) const
    {      
        return val_*a;
    }
};

template <typename T>
auto operator*(mt::device_vector<T> &&a, T b){
    mt::device_vector<T> ret(a);
    auto mul = mul_s<T>(b);
    thrust::transform(a.begin(), a.end(), ret.begin(), mul);
    
    return ret;
}

template <typename T>
auto operator*(mt::device_vector<T> &a, T b){
    mt::device_vector<T> ret = std::move(a) * b;
    return ret;
}


template <typename T>
auto operator*(T a, mt::device_vector<T> &&b){
    mt::device_vector<T> ret(b);
    auto mul = mul_s<T>(a);
    thrust::transform(b.begin(), b.end(), ret.begin(), mul);
    
    return ret;
}

template <typename T>
auto operator*(T a, mt::device_vector<T> &b){
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

template<typename T>
struct div_s
{   
    const T val_;
        
    div_s(T p)
        : val_{p} {}

    __host__ __device__
    inline T operator()(T& a) const
    {      
        return a / val_;
    }
};

template <typename T>
auto operator/(mt::device_vector<T> &&a, T b){
    mt::device_vector<T> ret(a);
    auto div = div_s<T>(b);
    thrust::transform(a.begin(), a.end(), ret.begin(), div);

    return ret;
}

template <typename T>
auto operator/(mt::device_vector<T> &a, T b){
    mt::device_vector<T> ret = std::move(a) / b;
    return ret;
}

template<typename T>
struct div_s_2
{   
    const T val_;
        
    div_s_2(T p)
        : val_{p} {}

    __host__ __device__
    inline T operator()(T& a) const
    {      
        return val_ / a;
    }
};


template <typename T>
auto operator/(T a, mt::device_vector<T> &&b){
    mt::device_vector<T> ret(b);
    auto div = div_s_2<T>(a);
    thrust::transform(b.begin(), b.end(), ret.begin(), div);
    
    return ret;
}

template <typename T>
auto operator/(T a, mt::device_vector<T> &b){
    mt::device_vector<T> ret = a / std::move(b);
    return ret;
}


/*  Matrix dot */

template<typename T>
auto mDot(mt::device_vector<T> &&a, mt::device_vector<T> &&b){
    
    assert(a.col() == b.row());



    mt::device_vector<T> ret(a.row(), b.col());
    //thrust::device_vector<T> ret(a.row()*b.col());

        std::cout << ret.row() << std::endl;
    std::cout << ret.col() << std::endl;

    if constexpr (std::is_same<T, float>{}) {
        float alpha = 1.0;
        float beta = 0.0;
        cublasSgemm(    
            g_handle,
            CUBLAS_OP_N,       //行列A 転置有無
            CUBLAS_OP_N,       //行列B 転置有無
            a.row(),           // 行列Aの行数
            b.row(),           // 行列Bの列数
            a.col(),           // 行列Aの列数(=行列Ｂの行数)
            &alpha,            // 行列の積に掛ける値(なければ1)
            a.data().get(),    // 行列A
            a.row(),           // 行列Aの行数
            b.data().get(),    // 行列B
            b.row(),           // 行列Bの行数
            &beta,             // 行列Cに掛けるスカラ値(なければ0)
            ret.data().get(),  // 行列Cの初期値 兼 出力先
            ret.row()          // 行列Cの行数
            
        );

    }else if constexpr(std::is_same<T, double>{}){
        double alpha = 1.0;
        double beta = 0.0;
        cublasDgemm(    
            g_handle,
            CUBLAS_OP_N, //行列A 転置有無
            CUBLAS_OP_N, //行列B 転置有無
            a.row(),     // 行列Aの行数
            b.row(),     // 行列Bの列数
            a.col(),     // 行列Aの列数(=行列Ｂの行数)
            &alpha,      // 行列の積に掛ける値(なければ1)
            a.data().get(),     // 行列A
            a.row(),     // 行列Aの行数
            b.data().get(),     // 行列B
            b.row(),     // 行列Bの行数
            &beta,       // 行列Cに掛けるスカラ値(なければ0)
            ret.data().get(),   // 行列Cの初期値 兼 出力先
            ret.row()    // 行列Cの行数
        );
    }

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
    
    //mt::device_vector<T> s;  /* Scalor */
    thrust::device_vector<T> s(1);

    if constexpr (std::is_same<T, float>{}) {    
        cublasSdot(
            g_handle,
            a.size(),
            a.data().get(),    // ベクトルA
            1,
            b.data().get(),    // ベクトルB
            1,
            s.data().get()
        );

    }else if constexpr(std::is_same<T, double>{}){
        cublasDdot(
            g_handle,
            a.size(),
            a.data().get(),    // ベクトルA
            1,
            b.data().get(),    // ベクトルB
            1,
            s.data().get()
        );

    }

    return s.data()[0];

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
            t1,             // 行列A 転置有無
            t2,             // 行列B 転置有無
            a_row,          // 行列Aの行数
            b_row,          // 行列Bの列数
            a_col,          // 行列Aの列数(=行列Ｂの行数)
            &alpha,         // 行列の積に掛ける値(なければ1)
            a.data().get(),        // 行列A
            a_row,          // 行列Aの行数
            b.data().get(),        // 行列B
            b_row,          // 行列Bの行数
            &beta,          // 行列Cに掛けるスカラ値(なければ0)
            ret.data().get(),      // 行列Cの初期値 兼 出力先
            ret.row()       // 行列Cの行数
        );

    }else if constexpr(std::is_same<T, double>{}){
            double alpha = 1.0;
            double beta = 0.0;
            cublasDgemm(    
                g_handle,
                CUBLAS_OP_N, // 行列A 転置有無
                CUBLAS_OP_N, // 行列B 転置有無
                a_row,       // 行列Aの行数
                b_row,       // 行列Bの列数
                a_col,       // 行列Aの列数(=行列Ｂの行数)
                &alpha,      // 行列の積に掛ける値(なければ1)
                a.data().get(),     // 行列A
                a_row,       // 行列Aの行数
                b.data().get(),     // 行列B
                b_row,       // 行列Bの行数
                &beta,       // 行列Cに掛けるスカラ値(なければ0)
                ret.data().get(),   // 行列Cの初期値 兼 出力先
                ret.row()    // 行列Cの行数
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


/*  Cos */
template<typename T>
struct Cos_s
{   

    __host__ __device__
    inline T operator()(T& a) const
    {      
        return cosf(a);
    }
};

template <typename T>
auto cos(mt::device_vector<T> &&a){
    mt::device_vector<T> ret(a);
    auto Cos = Cos_s<T>();
    thrust::transform(a.begin(), a.end(), ret.begin(), Cos);
    
    return ret;
}

template <typename T>
auto cos(mt::device_vector<T> &a){
    mt::device_vector<T> ret = cos(std::move(a));
    
    return ret;
}

/*  Sin */
template<typename T>
struct Sin_s
{   
    __host__ __device__
    inline T operator()(T& a) const
    {      
        return sinf(a);
    }
};

template <typename T>
auto sin(mt::device_vector<T> &&a){
    mt::device_vector<T> ret(a);
    auto Sin = Sin_s<T>();
    thrust::transform(a.begin(), a.end(), ret.begin(), Sin);
    
    return ret;
}

template <typename T>
auto sin(mt::device_vector<T> &a){
    mt::device_vector<T> ret = sin(std::move(a));
    
    return ret;
}


/*  Tan */
template<typename T>
struct Tan_s
{   
    __host__ __device__
    inline T operator()(T& a) const
    {      
        return tanf(a);
    }
};

template <typename T>
auto tan(mt::device_vector<T> &&a){
    mt::device_vector<T> ret(a);
    auto Tan = Tan_s<T>();
    thrust::transform(a.begin(), a.end(), ret.begin(), Tan);
    
    return ret;
}

template <typename T>
auto tan(mt::device_vector<T> &a){
    mt::device_vector<T> ret = tan(std::move(a));
    
    return ret;
}


/*  Exp */
template<typename T>
struct Exp_s
{   
    __host__ __device__
    inline T operator()(T& a) const
    {      
        return expf(a);
    }
};

template <typename T>
auto exp(mt::device_vector<T> &&a){
    mt::device_vector<T> ret(a);
    auto Exp = Exp_s<T>();
    thrust::transform(a.begin(), a.end(), ret.begin(), Exp);
    
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
    __host__ __device__
    inline T operator()(T& a) const
    {      
        return logf(a);
    }
};

template <typename T>
auto log(mt::device_vector<T> &&a){
    mt::device_vector<T> ret(a);
    auto Log = Log_s<T>();
    thrust::transform(a.begin(), a.end(), ret.begin(), Log);
    
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
    __host__ __device__
    inline T operator()(T& a) const
    {      
        return sqrtf(a);
    }
};

template <typename T>
auto sqrt(mt::device_vector<T> &&a){
    mt::device_vector<T> ret(a);
    auto Sqrt = Sqrt_s<T>();
    thrust::transform(a.begin(), a.end(), ret.begin(), Sqrt);
    
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
        return powf(a, val_);
    }
};

template <typename T, typename U>
auto pow(mt::device_vector<T> &&a, U b){
    mt::device_vector<T> ret(a);
    auto Pow = Pow_s<T,U>(b);
    thrust::transform(a.begin(), a.end(), ret.begin(), Pow);
    
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
    auto Relu = Relu_s<T>();
    thrust::transform(a.begin(), a.end(), ret.begin(), Relu);
    
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
        return 1/(1 + expf(-a));
    }
};

template <typename T>
auto sigmoid(mt::device_vector<T> &&a){
    mt::device_vector<T> ret(a);
    auto Sig = Sig_s<T>();
    thrust::transform(a.begin(), a.end(), ret.begin(), Sig);
    
    return ret;
}

template <typename T>
auto sigmoid(mt::device_vector<T> &a){
    mt::device_vector<T> ret = sigmoid(std::move(a));
    
    return ret;
}

/* Summation Vertical direction */
template <typename T>
auto sumV(mt::device_vector<T> &&a){

    mt::device_vector<T> ret(1, a.col());
    thrust::device_vector<T> d_ones(a.row(), 1.0);

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
                        thrust::raw_pointer_cast(d_ones.data()), 
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
                        thrust::raw_pointer_cast(d_ones.data()), 
                        1, 
                        &beta, 
                        ret.data().get(), 
                        1
                    );

    }

    return ret;

}

template <typename T>
auto sumV(mt::device_vector<T> &a){
    
    return sumV(std::move(a));

}

/* Summation Horizontal direction */
template <typename T>
auto sumH(mt::device_vector<T> &&a){

    mt::device_vector<T> ret(a.row(), 1);
    thrust::device_vector<T> d_ones(a.col(), 1.0);

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

    return ret;
}

template <typename T>
auto sumH(mt::device_vector<T> &a){
    return sumH(std::move(a));
}

/* |a1| + |a2| + |a3| +   */
template<typename T>
T abs(mt::device_vector<T> &&a){

    //mt::device_vector<T> s;  /* Scalor */
    thrust::device_vector<T> s(1);

    if constexpr (std::is_same<T, float>{}) {
        cublasSasum(
            g_handle,
            a.size(),
            a.data().get(),
            1,
            s.data().get()
        );
    }else if constexpr (std::is_same<T, double>{}) {
        cublasDasum(
            g_handle,
            a.size(),
            a.data().get(),
            1,
            s.data().get()
        );
    }

    return s.data()[0];

}

template<typename T>
T abs(mt::device_vector<T> &a){
    return abs(std::move(a));
}

/* L1 */
template<typename T>
T L1(mt::device_vector<T> &&a){
    return sqrtf(abs(a));
}

template<typename T>
T L1(mt::device_vector<T> &a){
    return L1(std::move(a));
}


/* L2 */
template<typename T>
T L2(mt::device_vector<T> &&a){

    mt::device_vector<T> s;  /* Scalor */

    if constexpr (std::is_same<T, float>{}) {
        cublasSnrm2(
            g_handle,
            a.size(),
            a.data().get(),
            1,
            s.data().get()
        );
    }else if constexpr (std::is_same<T, double>{}) {
        cublasDnrm2(
            g_handle,
            a.size(),
            a.data().get(),
            1,
            s.data().get()
        );
    }

    return s.data()[0];

}

template<typename T>
T L2(mt::device_vector<T> &a){
    return L2(std::move(a));
}

template<typename T>
union _f2i{
    T f;
    int i;
};

/* inverse matrix　*/
template<typename T>
auto INV(mt::device_vector<T> &a){
    assert( a.col() == a.row() );

    int n = a.col();
    mt::device_vector<T> A(a);

    mt::device_vector<T> B(n,n);
    print(B);

    union _f2i<T> d;
    d.f = 1.0;
    cuMemsetD2D32((CUdeviceptr)B.data().get(), (n+1)*sizeof(T), d.i, 1, n);

    print(B);

    cusolverStatus_t status;
    int worksize;

    // dense LAPACK
    cusolverDnHandle_t handle;

    status = cusolverDnCreate(&handle);
    assert( status == CUSOLVER_STATUS_SUCCESS );

    if constexpr (std::is_same<T, float>{}) {
        status = cusolverDnSgetrf_bufferSize(
              handle,
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
              handle,
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
               handle,
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
              handle,
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
              handle,
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
               handle,
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

    cusolverDnDestroy(handle);

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
    
    auto Rand = Rand_s<T>();

    thrust::transform(
        index_sequence_begin,
        index_sequence_begin + a.size(),
        ret.begin(),
        Rand
    );

    return ret;
}

template <typename T>
auto rand(mt::device_vector<T> &a){
    return rand(std::move(a));
}




int main(){


    cublasCreate(&g_handle);
    cublasSetPointerMode(g_handle,CUBLAS_POINTER_MODE_DEVICE); 

    thrust::host_vector<float> y1 = {5,-6};
    thrust::host_vector<float> y2 = {7,4,3,3};      

    thrust::host_vector<thrust::host_vector<double>> h2 = {
                                        {1,2},{3,4}
    };

    mt::device_vector t1(h2);
    mt::device_vector t2(y2, VectorType::Col);

    //mt::device_queue<decltype(&t1)> p;
    
    //p.push(&t1);
    //p.push(&t2);

    //INV(t1);

    print(t1*t1);

    auto ans = INV(t1);
    print(ans);

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



    cublasDestroy(g_handle);



}