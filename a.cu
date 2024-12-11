#include <iostream>
#include <vector>
#include <cublas_v2.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/gather.h>
#include <thrust/reduce.h>
#include <thrust/random.h>


cublasHandle_t g_handle;

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


template <typename T>
auto transpose(thrust::host_vector<T>& src, size_t m, size_t n)
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
auto transpose(thrust::device_vector<T>& src, size_t m, size_t n)
{
    thrust::device_vector<T> dst(m *n);
    
    thrust::counting_iterator<size_t> indices(0);

    thrust::gather
    (thrust::make_transform_iterator(indices, transpose_index(n, m)),
    thrust::make_transform_iterator(indices, transpose_index(n, m)) + dst.size(),
    src.begin(),dst.begin());

    return dst;
}


enum class VectorType
{ 
  Row,
  Col
};

template<typename T = float>
class ThrustMat{
public:

    /* Scalor */
    ThrustMat(){
        this -> row_ = 1;
        this -> col_ = 1;
        d.resize(1, 0);

    }

    /* Vector */
    ThrustMat(int sz, VectorType type){
        if(type == VectorType::Row){
            this -> row_ = sz;
            this -> col_ = 1;    
        }else if(type == VectorType::Col){
            this -> row_ = 1;
            this -> col_ = sz;
        }
        
        d.resize(sz, 0);
    }


    /* Matrix */
    ThrustMat(int r, int c){
        this -> row_ = r;
        this -> col_ = c;
        d.resize(r*c, 0);
    }

    /* copy constructor */
    ThrustMat(ThrustMat &a){
       // type_ = a.type_;
        row_ = a.row();
        col_ = a.col();
        d = a.d;
    }


    /* Vector */
    ThrustMat(thrust::host_vector<T> &hv, VectorType type){
        if(type == VectorType::Row){
            row_ = hv.size();
            col_ = 1;
        }else if(type == VectorType::Col){
            row_ = 1;
            col_ = hv.size();
        }
        
        d = hv;
    }

    /* Vector */
    ThrustMat(thrust::device_vector<T> &dv, VectorType type){
        if(type == VectorType::Row){
            row_ = dv.size();
            col_ = 1;
        }else if(type == VectorType::Col){
            row_ = 1;
            col_ = dv.size();
        }
        
        d = dv;
    }

    /* Matrix */
    ThrustMat(thrust::device_vector<T> &dv, int row, int col){
        row_ = row;
        col_ = col;
        
        d = dv;
    }

    /* Matrix */
    ThrustMat(thrust::host_vector<thrust::host_vector<T>> &hvv){
        row_ = hvv.size();
        col_ = hvv[0].size();

        thrust::host_vector<T> temp;

        for(auto &hv: hvv){
            for(auto u :hv){
                temp.push_back(u);
            }
        }

        auto v = transpose(temp, row_, col_);
            
        d = v;
    }

    ~ThrustMat(){
    }

    T* get(){
        //return thrust::raw_pointer_cast(d.data());
        return d.data().get();
    }

    int size(){
        return d.size();

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

    void print()
    {
        std::cout << "------" << std::endl;
        
        thrust::host_vector<T> out = transpose(d, col_, row_);
//        thrust::copy(d.begin(), d.end() , std::ostream_iterator<float>(std::cout, " "));
//        std::cout << std::endl;

        for(int i=0; i<row_; ++i){
            thrust::copy(out.begin() + col_ * i, out.begin() + col_ * (i + 1) , std::ostream_iterator<float>(std::cout, " "));    
            std::cout << std::endl;
        }

    }

    auto begin(){
        return d.begin();
    }

    auto end(){
        return d.end();
    }

    auto data(){
        return d.data();

    }

    void reserve(size_t sz){
        d.reserve(sz);

    }

    void insert(ThrustMat<T> &b){
        d.insert(d.end(), b.begin(), b.end());

    }


    auto operator=(ThrustMat<T> &a){
        if(this == &a) return *this;
        this -> row_ = a.row();
        this -> col_ = a.col();
        this -> d = a.d;
        return *this;
    }

    void trans(){
        d = transpose(d, col_, row_);
        int temp = row_;
        row_ = col_;
        col_ = temp;
    }

private:
    int row_;
    int col_;
        
    thrust::device_vector<T> d;
};

/* add */

template <typename T>
auto operator+(ThrustMat<T> &&a, ThrustMat<T> &&b){
    assert(a.size() == b.size());

    ThrustMat<T> ret(b);
    thrust::transform(a.begin(), a.end(), b.begin(), ret.begin(), thrust::plus<T>());

/*
    if constexpr (std::is_same<T, float>{}) {
        float p = 1.0;
    
        cublasSaxpy(
            g_handle, a.size(), &p,
            a.get(), 1,
            ret.get(), 1
        );

    }else if constexpr (std::is_same<T, double>{}){
        double p = 1.0;
    
        cublasDaxpy(
            g_handle, a.size(), &p,
            a.get(), 1,
            ret.get(), 1
        );

    }
*/
    return ret;
}

template <typename T>
auto operator+(ThrustMat<T> &a, ThrustMat<T> &b){
    ThrustMat<T> ret = std::move(a) + std::move(b);
    
    return ret;
}

template <typename T>
auto operator+(ThrustMat<T> &&a, ThrustMat<T> &b){
    ThrustMat<T> ret = std::move(a) + std::move(b);
    
    return ret;
}

template <typename T>
auto operator+(ThrustMat<T> &a, ThrustMat<T> &&b){
    ThrustMat<T> ret = std::move(a) + std::move(b);
    
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
auto operator+(ThrustMat<T> &&a, T b){
    ThrustMat<T> ret(a);
    auto add = add_s<T>(b);
    thrust::transform(a.begin(), a.end(), ret.begin(), add);
    
    return ret;
}

template <typename T>
auto operator+(ThrustMat<T> &a, T b){
    ThrustMat<T> ret = std::move(a) + b;
    return ret;
}

template <typename T>
auto operator+(T a, ThrustMat<T> &&b){
    ThrustMat<T> ret(b);
    auto add = add_s<T>(a);
    thrust::transform(b.begin(), b.end(), ret.begin(), add);
    
    return ret;
}

template <typename T>
auto operator+(T a, ThrustMat<T> &b){
    ThrustMat<T> ret = a + std::move(b);
    return ret;
}


/* sub */

template <typename T>
auto operator-(ThrustMat<T> &&a, ThrustMat<T> &&b){

    assert(a.size() == b.size());

    ThrustMat<T> ret(b);
    thrust::transform(a.begin(), a.end(), b.begin(), ret.begin(), thrust::minus<T>());

/*
    if constexpr (std::is_same<T, float>{}) {
        float p = -1.0;
    
        cublasSaxpy(
            g_handle, a.size(), &p,
            b.get(), 1,
            ret.get(), 1
        );

    }else if constexpr (std::is_same<T, double>{}){
        double p = -1.0;
    
        cublasDaxpy(
            g_handle, a.size(), &p,
            b.get(), 1,
            ret.get(), 1
        );

    }
*/
    return ret;
}

template <typename T>
auto operator-(ThrustMat<T> &a, ThrustMat<T> &b){
    ThrustMat<T> ret = std::move(a) + std::move(b);
    
    return ret;
}

template <typename T>
auto operator-(ThrustMat<T> &&a, ThrustMat<T> &b){
    ThrustMat<T> ret = std::move(a) - std::move(b);
    
    return ret;
}

template <typename T>
auto operator-(ThrustMat<T> &a, ThrustMat<T> &&b){
    ThrustMat<T> ret = std::move(a) - std::move(b);
    
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
auto operator-(ThrustMat<T> &&a, T b){
    ThrustMat<T> ret(a);
    auto sub = sub_s<T>(b);
    thrust::transform(a.begin(), a.end(), ret.begin(), sub);
    
    return ret;
}

template <typename T>
auto operator-(ThrustMat<T> &a, T b){
    ThrustMat<T> ret = std::move(a) - b;
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
auto operator-(T a, ThrustMat<T> &&b){
    ThrustMat<T> ret(b);
    auto sub = sub_s_2<T>(a);
    thrust::transform(b.begin(), b.end(), ret.begin(), sub);
    
    return ret;
}

template <typename T>
auto operator-(T a, ThrustMat<T> &b){
    ThrustMat<T> ret = a - std::move(b);
    return ret;
}

/* mul */
template <typename T>
auto operator*(ThrustMat<T> &&a, ThrustMat<T> &&b){

    assert(a.size() == b.size());
    ThrustMat<T> ret(b);

    thrust::transform(a.begin(), a.end(), b.begin(), ret.begin(), thrust::multiplies<T>());

    return ret;

}

template <typename T>
auto operator*(ThrustMat<T> &a, ThrustMat<T> &b){

    ThrustMat<T> ret = std::move(a) * std::move(b);

    return ret;

}

template <typename T>
auto operator*(ThrustMat<T> &&a, ThrustMat<T> &b){

    ThrustMat<T> ret = std::move(a) * std::move(b);

    return ret;

}

template <typename T>
auto operator*(ThrustMat<T> &a, ThrustMat<T> &&b){

    ThrustMat<T> ret = std::move(a) * std::move(b);

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
auto operator*(ThrustMat<T> &&a, T b){
    ThrustMat<T> ret(a);
    auto mul = mul_s<T>(b);
    thrust::transform(a.begin(), a.end(), ret.begin(), mul);
    
    return ret;
}

template <typename T>
auto operator*(ThrustMat<T> &a, T b){
    ThrustMat<T> ret = std::move(a) * b;
    return ret;
}


template <typename T>
auto operator*(T a, ThrustMat<T> &&b){
    ThrustMat<T> ret(b);
    auto mul = mul_s<T>(a);
    thrust::transform(b.begin(), b.end(), ret.begin(), mul);
    
    return ret;
}

template <typename T>
auto operator*(T a, ThrustMat<T> &b){
    ThrustMat<T> ret = a * std::move(b);
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
auto OPP(ThrustMat<T> &&a){
    ThrustMat<T> ret(a);    
    auto opp = opp_s<T>();
    thrust::transform(a.begin(), a.end(), ret.begin(), opp);

    return ret;

}

template<typename T>
auto OPP(ThrustMat<T> &a){
    ThrustMat<T> ret = OPP(std::move(a));
    return ret;
}

/* div */
template <typename T>
auto operator/(ThrustMat<T> &&a, ThrustMat<T> &&b){
    assert(a.size() == b.size());

    ThrustMat<T> ret(b);
    thrust::transform(a.begin(), a.end(), b.begin(), ret.begin(), thrust::divides<T>());
    return ret;
}

template <typename T>
auto operator/(ThrustMat<T> &&a, ThrustMat<T> &b){
    
    ThrustMat<T> ret = std::move(a) / std::move(b);
    return ret;
}

template <typename T>
auto operator/(ThrustMat<T> &a, ThrustMat<T> &&b){


    ThrustMat<T> ret = std::move(a) / std::move(b);
    return ret;
}

template <typename T>
auto operator/(ThrustMat<T> &a, ThrustMat<T> &b){
    
    ThrustMat<T> ret = std::move(a) / std::move(b);
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
auto operator/(ThrustMat<T> &&a, T b){
    ThrustMat<T> ret(a);
    auto div = div_s<T>(b);
    thrust::transform(a.begin(), a.end(), ret.begin(), div);

    return ret;
}

template <typename T>
auto operator/(ThrustMat<T> &a, T b){
    ThrustMat<T> ret = std::move(a) / b;
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
auto operator/(T a, ThrustMat<T> &&b){
    ThrustMat<T> ret(b);
    auto div = div_s_2<T>(a);
    thrust::transform(b.begin(), b.end(), ret.begin(), div);
    
    return ret;
}

template <typename T>
auto operator/(T a, ThrustMat<T> &b){
    ThrustMat<T> ret = a / std::move(b);
    return ret;
}

/*  Matrix dot */

template<typename T>
auto mDot(ThrustMat<T> &&a, ThrustMat<T> &&b){
    
    assert(a.col() == b.row());
    ThrustMat<T> ret(a.row(), b.col());

    if constexpr (std::is_same<T, float>{}) {
        float alpha = 1.0;
        float beta = 0.0;
        cublasSgemm(    
            g_handle,
            CUBLAS_OP_N, //行列A 転置有無
            CUBLAS_OP_N, //行列B 転置有無
            a.row(),    // 行列Aの行数
            b.row(),    // 行列Bの列数
            a.col(),    // 行列Aの列数(=行列Ｂの行数)
            &alpha,     // 行列の積に掛ける値(なければ1)
            a.get(),    // 行列A
            a.row(),    // 行列Aの行数
            b.get(),    // 行列B
            b.row(),    // 行列Bの行数
            &beta,      // 行列Cに掛けるスカラ値(なければ0)
            ret.get(),  // 行列Cの初期値 兼 出力先
            ret.row()   // 行列Cの行数
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
            a.get(),     // 行列A
            a.row(),     // 行列Aの行数
            b.get(),     // 行列B
            b.row(),     // 行列Bの行数
            &beta,       // 行列Cに掛けるスカラ値(なければ0)
            ret.get(),   // 行列Cの初期値 兼 出力先
            ret.row()    // 行列Cの行数
        );
    }

    return ret;
}


template<typename T>
auto mDot(ThrustMat<T> &a, ThrustMat<T> &b){
    ThrustMat<T> ret = mDot(std::move(a), std::move(b));

    return ret;
}

template<typename T>
auto mDot(ThrustMat<T> &&a, ThrustMat<T> &b){
    ThrustMat<T> ret = mDot(std::move(a), std::move(b));
    return ret;
}

template<typename T>
auto mDot(ThrustMat<T> &a, ThrustMat<T> &&b){
    ThrustMat<T> ret = mDot(std::move(a), std::move(b)); 
    return ret;
}

/* vector inner product */
template<typename T>
T vDot(ThrustMat<T> &&a, ThrustMat<T> &&b){
    assert(a.size() == b.size());
    
    ThrustMat<T> s;  /* Scalor */

    if constexpr (std::is_same<T, float>{}) {    
        cublasSdot(
            g_handle,
            a.size(),
            a.get(),    // ベクトルA
            1,
            b.get(),    // ベクトルB
            1,
            s.get()
        );

    }else if constexpr(std::is_same<T, double>{}){
        cublasDdot(
            g_handle,
            a.size(),
            a.get(),    // ベクトルA
            1,
            b.get(),    // ベクトルB
            1,
            s.get()
        );

    }

    T ret = s.data()[0];

    return ret;

}


template<typename T>
auto vDot(ThrustMat<T> &a, ThrustMat<T> &&b){
    auto ret = vDot(std::move(a), std::move(b));
    return ret;
}

template<typename T>
auto vDot(ThrustMat<T> &&a, ThrustMat<T> &b){
    auto ret = vDot(std::move(a), std::move(b));
    return ret;
}

template<typename T>
auto vDot(ThrustMat<T> &a, ThrustMat<T> &b){
    auto ret = vDot(std::move(a), std::move(b));
    return ret;
}

/* Matrix dot */
template<typename T>
auto dot(ThrustMat<T> &&a, bool T1 , ThrustMat<T> &&b, bool T2 ){
    
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

    ThrustMat<T> ret(a_row, b_col);

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
            a.get(),        // 行列A
            a_row,          // 行列Aの行数
            b.get(),        // 行列B
            b_row,          // 行列Bの行数
            &beta,          // 行列Cに掛けるスカラ値(なければ0)
            ret.get(),      // 行列Cの初期値 兼 出力先
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
                a.get(),     // 行列A
                a_row,       // 行列Aの行数
                b.get(),     // 行列B
                b_row,       // 行列Bの行数
                &beta,       // 行列Cに掛けるスカラ値(なければ0)
                ret.get(),   // 行列Cの初期値 兼 出力先
                ret.row()    // 行列Cの行数
            );
    }

    return ret;
}



template<typename T>
auto dot(ThrustMat<T> &&a, bool T1 , ThrustMat<T> &b, bool T2 ){
    ThrustMat<T> ret = dot(std::move(a), T1, std::move(b), T2); 
    return ret;

}

template<typename T>
auto dot(ThrustMat<T> &a, bool T1 , ThrustMat<T> &&b, bool T2 ){
    ThrustMat<T> ret = dot(std::move(a), T1, std::move(b), T2); 
    return ret;

}


template<typename T>
auto dot(ThrustMat<T> &a, bool T1 , ThrustMat<T> &b, bool T2 ){
    ThrustMat<T> ret = dot(std::move(a), T1, std::move(b), T2); 
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
auto cos(ThrustMat<T> &&a){
    ThrustMat<T> ret(a);
    auto Cos = Cos_s<T>();
    thrust::transform(a.begin(), a.end(), ret.begin(), Cos);
    
    return ret;
}

template <typename T>
auto cos(ThrustMat<T> &a){
    ThrustMat<T> ret = cos(std::move(a));
    
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
auto sin(ThrustMat<T> &&a){
    ThrustMat<T> ret(a);
    auto Sin = Sin_s<T>();
    thrust::transform(a.begin(), a.end(), ret.begin(), Sin);
    
    return ret;
}

template <typename T>
auto sin(ThrustMat<T> &a){
    ThrustMat<T> ret = sin(std::move(a));
    
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
auto tan(ThrustMat<T> &&a){
    ThrustMat<T> ret(a);
    auto Tan = Tan_s<T>();
    thrust::transform(a.begin(), a.end(), ret.begin(), Tan);
    
    return ret;
}

template <typename T>
auto tan(ThrustMat<T> &a){
    ThrustMat<T> ret = tan(std::move(a));
    
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
auto exp(ThrustMat<T> &&a){
    ThrustMat<T> ret(a);
    auto Exp = Exp_s<T>();
    thrust::transform(a.begin(), a.end(), ret.begin(), Exp);
    
    return ret;
}

template <typename T>
auto exp(ThrustMat<T> &a){
    ThrustMat<T> ret = exp(std::move(a));
    
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
auto log(ThrustMat<T> &&a){
    ThrustMat<T> ret(a);
    auto Log = Log_s<T>();
    thrust::transform(a.begin(), a.end(), ret.begin(), Log);
    
    return ret;
}

template <typename T>
auto log(ThrustMat<T> &a){
    ThrustMat<T> ret = log(std::move(a));
    
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
auto sqrt(ThrustMat<T> &&a){
    ThrustMat<T> ret(a);
    auto Sqrt = Sqrt_s<T>();
    thrust::transform(a.begin(), a.end(), ret.begin(), Sqrt);
    
    return ret;
}

template <typename T>
auto sqrt(ThrustMat<T> &a){
    ThrustMat<T> ret = sqrt(std::move(a));
    
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
auto pow(ThrustMat<T> &&a, U b){
    ThrustMat<T> ret(a);
    auto Pow = Pow_s<T,U>(b);
    thrust::transform(a.begin(), a.end(), ret.begin(), Pow);
    
    return ret;
}

template <typename T, typename U>
auto pow(ThrustMat<T> &a, U b){
    ThrustMat<T> ret = pow(std::move(a), b);
    
    return ret;
}



/* resize */
template<typename T>
auto resize(ThrustMat<T> &&a, int r, int c){
    assert(a.size() == r*c);
    ThrustMat<T> ret = a;
    ret.setRow = r;
    ret.setCol = c;
    return ret;

}


template<typename T>
auto resize(ThrustMat<T> &a, int r, int c){
    assert(a.size() == r*c);
    ThrustMat<T> ret = resize(std::move(a), r, c);
    
    return ret;

}



struct rm2cm_idx_functor : public thrust::unary_function<int, int>
{
  int r;
  int c;

  rm2cm_idx_functor(int _r, int _c) : r(_r), c(_c) {};

  __host__ __device__
  int operator() (int idx)  {
    unsigned my_r = idx/c;
    unsigned my_c = idx%c;
    return (my_c * r) + my_r;
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
auto concatH(ThrustMat<T> &&a, ThrustMat<T> &&b){
    assert(a.row() == b.row());
    
    ThrustMat<T> ret(a);

    ret.reserve(a.size() + b.size());
    ret.setCol(a.col() + b.col());
    ret.insert(b);
   
    return ret;
}

template<typename T>
auto concatH(ThrustMat<T> &&a, ThrustMat<T> &b){
    
    ThrustMat<T> ret = concatH(std::move(a), std::move(b));
    return ret;
}

template<typename T>
auto concatH(ThrustMat<T> &a, ThrustMat<T> &&b){
    
    ThrustMat<T> ret = concatH(std::move(a), std::move(b));
    return ret;
}


template<typename T>
auto concatH(ThrustMat<T> &a, ThrustMat<T> &b){
    
    ThrustMat<T> ret = concatH(std::move(a), std::move(b));
    return ret;
}


/* Concat Vertical */



template<typename T>
auto concatV(ThrustMat<T> &&a, ThrustMat<T> &&b){
    assert(a.col() == b.col());

    ThrustMat<T> ret(a.row() + b.row(), a.col());

    int offset = 0;
    copyMat(a.data(), ret.data(), a.row(), a.col(), ret.row(), offset);
    offset = a.col()*a.row();
    copyMat(b.data(), ret.data(), b.row(), b.col(), ret.row(), offset);

    return ret;
}


template<typename T>
auto concatV(ThrustMat<T> &a, ThrustMat<T> &b){
   
    ThrustMat<T> ret = concatV(std::move(a), std::move(b));
    
    return ret;
}


template<typename T>
auto concatV(ThrustMat<T> &&a, ThrustMat<T> &b){
   
    ThrustMat<T> ret = concatV(std::move(a), std::move(b));
    
    return ret;
}

template<typename T>
auto concatV(ThrustMat<T> &a, ThrustMat<T> &&b){
   
    ThrustMat<T> ret = concatV(std::move(a), std::move(b));
    
    return ret;
}

/* TransPosition  */
template<typename T>
auto TP(ThrustMat<T> &&a){
    
    ThrustMat<T> ret(a);
    ret.trans();

    return ret;
}

template<typename T>
auto TP(ThrustMat<T> &a){
    ThrustMat<T> ret = TP(std::move(a));
    
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
auto relu(ThrustMat<T> &&a){
    ThrustMat<T> ret(a);
    auto Relu = Relu_s<T>();
    thrust::transform(a.begin(), a.end(), ret.begin(), Relu);
    
    return ret;
}

template <typename T>
auto relu(ThrustMat<T> &a){
    ThrustMat<T> ret = relu(std::move(a));
    
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
auto sigmoid(ThrustMat<T> &&a){
    ThrustMat<T> ret(a);
    auto Sig = Sig_s<T>();
    thrust::transform(a.begin(), a.end(), ret.begin(), Sig);
    
    return ret;
}

template <typename T>
auto sigmoid(ThrustMat<T> &a){
    ThrustMat<T> ret = sigmoid(std::move(a));
    
    return ret;
}

/* Summation Vertical direction */
template <typename T>
auto sumV(ThrustMat<T> &&a){

    ThrustMat<T> ret(1, a.col());
    thrust::device_vector<T> d_ones(a.row(), 1.0);

    T alpha = 1.0;
    T beta  = 0.0;  
          
    if constexpr (std::is_same<T, float>{}) {
          cublasSgemv(g_handle, 
                        CUBLAS_OP_T, 
                        a.row(), 
                        a.col(), 
                        &alpha, 
                        a.get(), 
                        a.row(), 
                        thrust::raw_pointer_cast(d_ones.data()), 
                        1, 
                        &beta, 
                        ret.get(), 
                        1
                    );

    }
    else  if constexpr (std::is_same<T, double>{}) {
          cublasDgemv(g_handle, 
                        CUBLAS_OP_T, 
                        a.row(), 
                        a.col(), 
                        &alpha, 
                        a.get(), 
                        a.row(), 
                        thrust::raw_pointer_cast(d_ones.data()), 
                        1, 
                        &beta, 
                        ret.get(), 
                        1
                    );

    }

    return ret;

}

template <typename T>
auto sumV(ThrustMat<T> &a){
    
    return sumV(std::move(a));

}

/* Summation Horizontal direction */
template <typename T>
auto sumH(ThrustMat<T> &&a){

    ThrustMat<T> ret(a.row(), 1);
    thrust::device_vector<T> d_ones(a.col(), 1.0);

    T alpha = 1.0;
    T beta  = 0.0;  
          
    if constexpr (std::is_same<T, float>{}) {
          cublasSgemv(g_handle, 
                        CUBLAS_OP_N, 
                        a.row(), 
                        a.col(), 
                        &alpha, 
                        a.get(), 
                        a.row(), 
                        thrust::raw_pointer_cast(d_ones.data()), 
                        1, 
                        &beta, 
                        ret.get(), 
                        1
                    );

    }
    else if constexpr (std::is_same<T, double>{}) {
          cublasDgemv(g_handle, 
                        CUBLAS_OP_N, 
                        a.row(), 
                        a.col(), 
                        &alpha, 
                        a.get(), 
                        a.row(), 
                        thrust::raw_pointer_cast(d_ones.data()), 
                        1, 
                        &beta, 
                        ret.get(), 
                        1
                    );

    }

    return ret;
}

template <typename T>
auto sumH(ThrustMat<T> &a){
    return sumH(std::move(a));
}


/* RANDOM */

struct GenRand
{
    __host__ __device__
    float operator () (int idx)
    {
        thrust::default_random_engine randEng;
        thrust::uniform_real_distribution<float> uniDist;
        randEng.discard(idx);
        return uniDist(randEng);
    }
};

template <typename T>
auto rand(ThrustMat<T> &&a){
    ThrustMat<T> ret(a);
 
    thrust::transform(
        thrust::make_counting_iterator(0),
        thrust::make_counting_iterator(a.size()),
        ret.begin(),
        GenRand()
    );

    return ret;
}

template <typename T>
auto rand(ThrustMat<T> &a){
    return rand(std::move(a));
}

int main(){
   
    thrust::host_vector<thrust::host_vector<float>> x = { //  4,2
                                                            {0, 2},
                                                            {1, 2},
                                                            {0, 3},
                                                            {1, 4}
                                                        };

    thrust::host_vector<thrust::host_vector<float>> z = { //  4,2
                                                            {0, 2},
                                                            {1, 2},
                                                            {0, 3},
                                                            {1, 4}
                                                        };


    thrust::host_vector<float> y = {5,4,0,1};   

    thrust::host_vector<float> w =  {1,1,2,2};   
    thrust::host_vector<thrust::host_vector<float>> b =  {
                                                            {0,5,5,0}, 
                                                            {0,5,5,0}
                                                            };        

    ThrustMat Dx(x);
    ThrustMat Dz(z);
    ThrustMat Dy(y, VectorType::Row);

    ThrustMat Dw(w, VectorType::Row);
    ThrustMat Db(b);

    cublasCreate(&g_handle);

    concatH(Dx, Dz).print();
    
    //auto b = concatH(mDot((vDot(d1 ,d1) * d2 - 100.0f), d4),d1);
    //b.print();

    //auto a = sigmoid(relu(b));
    
    //a.print();

    //auto m3 = concatH(d2,  d1);
 
    //d2.print();
    //TP(d2).print();

    //m3.print();


    cublasDestroy(g_handle);





}