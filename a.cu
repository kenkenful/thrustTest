#include <iostream>
#include <vector>
#include <cublas_v2.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/gather.h>
#include <thrust/reduce.h>


cublasHandle_t g_handle;

/*
template <typename T>
inline auto trans(thrust::host_vector<T>& mat, std::size_t row_size, std::size_t col_size) {
    thrust::host_vector<T> ret(col_size * row_size);

    for (auto i = 0; i < col_size; ++i) {
        for (auto j = 0; j < row_size; ++j) {
            ret[i * row_size + j] = mat[j * col_size + i];
        }
    }
    return ret;
}   

template <typename T>
inline auto trans(thrust::device_vector<T>& mat, std::size_t row_size, std::size_t col_size) {
    thrust::device_vector<T> ret(col_size * row_size);

    for (auto i = 0; i < col_size; ++i) {
        for (auto j = 0; j < row_size; ++j) {
            ret[i * row_size + j] = mat[j * col_size + i];
        }
    }
    return ret;
}   
*/

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
class CMat{
public:

    /* Scalor */
    CMat(){
        this -> row_ = 1;
        this -> col_ = 1;
        d.resize(1, 0);

    }

    /* Vector */
    CMat(int sz, VectorType type){
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
    CMat(int r, int c){
        this -> row_ = r;
        this -> col_ = c;
        d.resize(r*c, 0);
    }

    /* copy constructor */
    CMat(CMat &a){
       // type_ = a.type_;
        row_ = a.row();
        col_ = a.col();
        d = a.d;
    }


    /* Vector */
    CMat(thrust::host_vector<T> &hv, VectorType type){
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
    CMat(thrust::device_vector<T> &dv, VectorType type){
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
    CMat(thrust::host_vector<thrust::host_vector<T>> &hvv){
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

    ~CMat(){
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

    auto operator=(CMat<T> &a){
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


    auto first_val(){
        return d[0];
    }

private:
    int row_;
    int col_;
        
    thrust::device_vector<T> d;
};

/* add */

template <typename T>
auto operator+(CMat<T> &&a, CMat<T> &&b){
    assert(a.size() == b.size());

    CMat<T> ret(b);
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
auto operator+(CMat<T> &a, CMat<T> &b){
    CMat<T> ret = std::move(a) + std::move(b);
    
    return ret;
}

template <typename T>
auto operator+(CMat<T> &&a, CMat<T> &b){
    CMat<T> ret = std::move(a) + std::move(b);
    
    return ret;
}

template <typename T>
auto operator+(CMat<T> &a, CMat<T> &&b){
    CMat<T> ret = std::move(a) + std::move(b);
    
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
auto operator+(CMat<T> &&a, T b){
    CMat<T> ret(a);
    auto add = add_s<T>(b);
    thrust::transform(a.begin(), a.end(), ret.begin(), add);
    
    return ret;
}

template <typename T>
auto operator+(CMat<T> &a, T b){
    CMat<T> ret = std::move(a) + b;
    return ret;
}

template <typename T>
auto operator+(T a, CMat<T> &&b){
    CMat<T> ret(b);
    auto add = add_s<T>(a);
    thrust::transform(b.begin(), b.end(), ret.begin(), add);
    
    return ret;
}

template <typename T>
auto operator+(T a, CMat<T> &b){
    CMat<T> ret = a + std::move(b);
    return ret;
}


/* sub */

template <typename T>
auto operator-(CMat<T> &&a, CMat<T> &&b){

    assert(a.size() == b.size());

    CMat<T> ret(b);
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
auto operator-(CMat<T> &a, CMat<T> &b){
    CMat<T> ret = std::move(a) + std::move(b);
    
    return ret;
}

template <typename T>
auto operator-(CMat<T> &&a, CMat<T> &b){
    CMat<T> ret = std::move(a) - std::move(b);
    
    return ret;
}

template <typename T>
auto operator-(CMat<T> &a, CMat<T> &&b){
    CMat<T> ret = std::move(a) - std::move(b);
    
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
auto operator-(CMat<T> &&a, T b){
    CMat<T> ret(a);
    auto sub = sub_s<T>(b);
    thrust::transform(a.begin(), a.end(), ret.begin(), sub);
    
    return ret;
}

template <typename T>
auto operator-(CMat<T> &a, T b){
    CMat<T> ret = std::move(a) - b;
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
auto operator-(T a, CMat<T> &&b){
    CMat<T> ret(b);
    auto sub = sub_s_2<T>(a);
    thrust::transform(b.begin(), b.end(), ret.begin(), sub);
    
    return ret;
}

template <typename T>
auto operator-(T a, CMat<T> &b){
    CMat<T> ret = a - std::move(b);
    return ret;
}

/* mul */
template <typename T>
auto operator*(CMat<T> &&a, CMat<T> &&b){

    assert(a.size() == b.size());
    CMat<T> ret(b);

    thrust::transform(a.begin(), a.end(), b.begin(), ret.begin(), thrust::multiplies<T>());

    return ret;

}

template <typename T>
auto operator*(CMat<T> &a, CMat<T> &b){

    CMat<T> ret = std::move(a) * std::move(b);

    return ret;

}

template <typename T>
auto operator*(CMat<T> &&a, CMat<T> &b){

    CMat<T> ret = std::move(a) * std::move(b);

    return ret;

}

template <typename T>
auto operator*(CMat<T> &a, CMat<T> &&b){

    CMat<T> ret = std::move(a) * std::move(b);

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
auto operator*(CMat<T> &&a, T b){
    CMat<T> ret(a);
    auto mul = mul_s<T>(b);
    thrust::transform(a.begin(), a.end(), ret.begin(), mul);
    
    return ret;
}

template <typename T>
auto operator*(CMat<T> &a, T b){
    CMat<T> ret = std::move(a) * b;
    return ret;
}


template <typename T>
auto operator*(T a, CMat<T> &&b){
    CMat<T> ret(b);
    auto mul = mul_s<T>(a);
    thrust::transform(b.begin(), b.end(), ret.begin(), mul);
    
    return ret;
}

template <typename T>
auto operator*(T a, CMat<T> &b){
    CMat<T> ret = a * std::move(b);
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
auto OPP(CMat<T> &&a){
    CMat<T> ret(a);    
    auto opp = opp_s<T>();
    thrust::transform(a.begin(), a.end(), ret.begin(), opp);

    return ret;

}

template<typename T>
auto OPP(CMat<T> &a){
    CMat<T> ret = OPP(std::move(a));
    return ret;
}

/* div */
template <typename T>
auto operator/(CMat<T> &&a, CMat<T> &&b){
    assert(a.size() == b.size());

    CMat<T> ret(b);
    thrust::transform(a.begin(), a.end(), b.begin(), ret.begin(), thrust::divides<T>());
    return ret;
}

template <typename T>
auto operator/(CMat<T> &&a, CMat<T> &b){
    
    CMat<T> ret = std::move(a) / std::move(b);
    return ret;
}

template <typename T>
auto operator/(CMat<T> &a, CMat<T> &&b){


    CMat<T> ret = std::move(a) / std::move(b);
    return ret;
}

template <typename T>
auto operator/(CMat<T> &a, CMat<T> &b){
    
    CMat<T> ret = std::move(a) / std::move(b);
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
auto operator/(CMat<T> &&a, T b){
    CMat<T> ret(a);
    auto div = div_s<T>(b);
    thrust::transform(a.begin(), a.end(), ret.begin(), div);

    return ret;
}

template <typename T>
auto operator/(CMat<T> &a, T b){
    CMat<T> ret = std::move(a) / b;
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
auto operator/(T a, CMat<T> &&b){
    CMat<T> ret(b);
    auto div = div_s_2<T>(a);
    thrust::transform(b.begin(), b.end(), ret.begin(), div);
    
    return ret;
}

template <typename T>
auto operator/(T a, CMat<T> &b){
    CMat<T> ret = a / std::move(b);
    return ret;
}

/*  Matrix dot */

template<typename T>
auto mDot(CMat<T> &&a, CMat<T> &&b){
    
    assert(a.col() == b.row());
    CMat<T> ret(a.row(), b.col());

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
auto mDot(CMat<T> &a, CMat<T> &b){
    CMat<T> ret = mDot(std::move(a), std::move(b));

    return ret;
}

template<typename T>
auto mDot(CMat<T> &&a, CMat<T> &b){
    CMat<T> ret = mDot(std::move(a), std::move(b));
    return ret;
}

template<typename T>
auto mDot(CMat<T> &a, CMat<T> &&b){
    CMat<T> ret = mDot(std::move(a), std::move(b)); 
    return ret;
}

/* vector inner product */
template<typename T>
auto vDot(CMat<T> &&a, CMat<T> &&b){
    assert(a.size() == b.size());
    
    CMat<T> s;  /* Scalor */

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

    T ret = s.first_val();

    return ret;

}


template<typename T>
auto vDot(CMat<T> &a, CMat<T> &&b){
    auto ret = vDot(std::move(a), std::move(b));
    return ret;
}

template<typename T>
auto vDot(CMat<T> &&a, CMat<T> &b){
    auto ret = vDot(std::move(a), std::move(b));
    return ret;
}

template<typename T>
auto vDot(CMat<T> &a, CMat<T> &b){
    auto ret = vDot(std::move(a), std::move(b));
    return ret;
}

/* Matrix dot */
template<typename T>
auto dot(CMat<T> &&a, bool T1 , CMat<T> &&b, bool T2 ){
    
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

    CMat<T> ret(a_row, b_col);

    if constexpr (std::is_same<T, float>{}) {
        float alpha = 1.0;
        float beta = 0.0;
        cublasSgemm(    
            g_handle,
            t1, //行列A 転置有無
            t2, //行列B 転置有無
            a_row,    // 行列Aの行数
            b_row,    // 行列Bの列数
            a_col,    // 行列Aの列数(=行列Ｂの行数)
            &alpha,     // 行列の積に掛ける値(なければ1)
            a.get(),    // 行列A
            a_row,    // 行列Aの行数
            b.get(),    // 行列B
            b_row,    // 行列Bの行数
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
                a_row,     // 行列Aの行数
                b_row,     // 行列Bの列数
                a_col,     // 行列Aの列数(=行列Ｂの行数)
                &alpha,      // 行列の積に掛ける値(なければ1)
                a.get(),     // 行列A
                a_row,     // 行列Aの行数
                b.get(),     // 行列B
                b_row,     // 行列Bの行数
                &beta,       // 行列Cに掛けるスカラ値(なければ0)
                ret.get(),   // 行列Cの初期値 兼 出力先
                ret.row()    // 行列Cの行数
            );
    }

    return ret;
}



template<typename T>
auto dot(CMat<T> &&a, bool T1 , CMat<T> &b, bool T2 ){
    CMat<T> ret = dot(std::move(a), T1, std::move(b), T2); 
    return ret;

}

template<typename T>
auto dot(CMat<T> &a, bool T1 , CMat<T> &&b, bool T2 ){
    CMat<T> ret = dot(std::move(a), T1, std::move(b), T2); 
    return ret;

}


template<typename T>
auto dot(CMat<T> &a, bool T1 , CMat<T> &b, bool T2 ){
    CMat<T> ret = dot(std::move(a), T1, std::move(b), T2); 
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
auto cos(CMat<T> &&a){
    CMat<T> ret(a);
    auto Cos = Cos_s<T>();
    thrust::transform(a.begin(), a.end(), ret.begin(), Cos);
    
    return ret;
}

template <typename T>
auto cos(CMat<T> &a){
    CMat<T> ret = cos(std::move(a));
    
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
auto sin(CMat<T> &&a){
    CMat<T> ret(a);
    auto Sin = Sin_s<T>();
    thrust::transform(a.begin(), a.end(), ret.begin(), Sin);
    
    return ret;
}

template <typename T>
auto sin(CMat<T> &a){
    CMat<T> ret = sin(std::move(a));
    
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
auto tan(CMat<T> &&a){
    CMat<T> ret(a);
    auto Tan = Tan_s<T>();
    thrust::transform(a.begin(), a.end(), ret.begin(), Tan);
    
    return ret;
}

template <typename T>
auto tan(CMat<T> &a){
    CMat<T> ret = tan(std::move(a));
    
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
auto exp(CMat<T> &&a){
    CMat<T> ret(a);
    auto Exp = Exp_s<T>();
    thrust::transform(a.begin(), a.end(), ret.begin(), Exp);
    
    return ret;
}

template <typename T>
auto exp(CMat<T> &a){
    CMat<T> ret = exp(std::move(a));
    
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
auto log(CMat<T> &&a){
    CMat<T> ret(a);
    auto Log = Log_s<T>();
    thrust::transform(a.begin(), a.end(), ret.begin(), Log);
    
    return ret;
}

template <typename T>
auto log(CMat<T> &a){
    CMat<T> ret = log(std::move(a));
    
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
auto sqrt(CMat<T> &&a){
    CMat<T> ret(a);
    auto Sqrt = Sqrt_s<T>();
    thrust::transform(a.begin(), a.end(), ret.begin(), Sqrt);
    
    return ret;
}

template <typename T>
auto sqrt(CMat<T> &a){
    CMat<T> ret = sqrt(std::move(a));
    
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
auto pow(CMat<T> &&a, U b){
    CMat<T> ret(a);
    auto Pow = Pow_s<T,U>(b);
    thrust::transform(a.begin(), a.end(), ret.begin(), Pow);
    
    return ret;
}

template <typename T, typename U>
auto pow(CMat<T> &a, U b){
    CMat<T> ret = pow(std::move(a), b);
    
    return ret;
}



/* resize */
template<typename T>
auto resize(CMat<T> &&a, int r, int c){
    assert(a.size() == r*c);
    CMat<T> ret = a;
    ret.setRow = r;
    ret.setCol = c;
    return ret;

}


template<typename T>
auto resize(CMat<T> &a, int r, int c){
    assert(a.size() == r*c);
    CMat<T> ret = resize(std::move(a), r, c);
    
    return ret;

}



/* Concat Horizon */
template<typename T>
auto concatH(CMat<T> &&a, CMat<T> &&b){
    assert(a.row() == b.row());
    CMat<T> ret(a.row(), a.col() + b.col());
    thrust::copy(a.begin(), a.end(), ret.begin()); 
    thrust::copy(b.begin(), b.end(), ret.begin() + a.size()); 
    
    return ret;
}

template<typename T>
auto concatH(CMat<T> &&a, CMat<T> &b){
    
    CMat<T> ret = concatH(std::move(a), std::move(b));
    return ret;
}

template<typename T>
auto concatH(CMat<T> &a, CMat<T> &&b){
    
    CMat<T> ret = concatH(std::move(a), std::move(b));
    return ret;
}


template<typename T>
auto concatH(CMat<T> &a, CMat<T> &b){
    
    CMat<T> ret = concatH(std::move(a), std::move(b));
    return ret;
}


/* Concat Vertical */
template<typename T>
auto concatV(CMat<T> &&a, CMat<T> &&b){
    assert(a.col() == b.col());

    CMat<T> ret(a.row() + b.row(), a.col());

    for(int i=0; i< a.col(); ++i){
        thrust::copy(a.begin() + a.row() * i, a.begin() + a.row() * (i + 1), ret.begin() + ret.row() * i );
        thrust::copy(b.begin() + b.row() * i, b.begin() + b.row() * (i + 1), ret.begin() + ret.row() * i + a.row() ); 
    }
    
    return ret;
}


template<typename T>
auto concatV(CMat<T> &a, CMat<T> &b){
   
    CMat<T> ret = concatV(std::move(a), std::move(b));
    
    return ret;
}


template<typename T>
auto concatV(CMat<T> &&a, CMat<T> &b){
   
    CMat<T> ret = concatV(std::move(a), std::move(b));
    
    return ret;
}

template<typename T>
auto concatV(CMat<T> &a, CMat<T> &&b){
   
    CMat<T> ret = concatV(std::move(a), std::move(b));
    
    return ret;
}

/* TransPosition  */
template<typename T>
auto TP(CMat<T> &&a){
    
    CMat<T> ret(a);
    ret.trans();

    return ret;
}

template<typename T>
auto TP(CMat<T> &a){
    CMat<T> ret = TP(std::move(a));
    
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
auto relu(CMat<T> &&a){
    CMat<T> ret(a);
    auto Relu = Relu_s<T>();
    thrust::transform(a.begin(), a.end(), ret.begin(), Relu);
    
    return ret;
}

template <typename T>
auto relu(CMat<T> &a){
    CMat<T> ret = relu(std::move(a));
    
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
auto sigmoid(CMat<T> &&a){
    CMat<T> ret(a);
    auto Sig = Sig_s<T>();
    thrust::transform(a.begin(), a.end(), ret.begin(), Sig);
    
    return ret;
}

template <typename T>
auto sigmoid(CMat<T> &a){
    CMat<T> ret = sigmoid(std::move(a));
    
    return ret;
}

/* Summation Vertical direction */
template <typename T>
auto sumV(CMat<T> &&a){
    thrust::device_vector<T> d;
    for(int i=0; i<a.col();++i){
        auto x = thrust::reduce(a.begin() + a.row() * i, a.begin()  + a.row() * (i + 1));
        d.push_back(x);
    }

    CMat<T> ret(d, VectorType::Col);

    return ret;

}

template <typename T>
auto sumV(CMat<T> &a){
    
    return sumV(std::move(a));

}

/* Summation Horizontal direction */
template <typename T>
auto sumH(CMat<T> &&a){
    CMat<T> ret = sumV(TP(a));
    ret.trans();

    return ret;

}

template <typename T>
auto sumH(CMat<T> &a){
    return sumH(std::move(a));
}

int main(){
   
    thrust::host_vector<thrust::host_vector<float>> hh = { //  4,3
                                                            {1,2,3},
                                                            {2,2,3},
                                                            {2,2,3},
                                                            {2,2,3}
                                                        };    
    
    thrust::host_vector<thrust::host_vector<float>> cc = { // 3,4
                                                            {-1,2,3,-5},
                                                            {4,-5,-6,6},
                                                            {7,8,-9,11}
                                                        };   


    thrust::host_vector<float> h = {7,7,7}; 
    
    //CMat m1(h);
    CMat d1(h, VectorType::Row);
    CMat d2(cc);
    CMat d3(20, VectorType::Row);
    CMat d4(hh);

    cublasCreate(&g_handle);

    sumV(d4).print();

    
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