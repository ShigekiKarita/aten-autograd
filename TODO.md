# more const

test how Variable's assign-op works harmless? (in RNN)

# Make Variable.grad Variable

and support double backwards

# Port Torch functions
basically over-write methods (e.g., `tanh_()`) are not to be ported to Variable


## operator overloads
| at::Tensor method                         | atnn::Variable |
| :---                                      | :--:           |
| Tensor operator-() const;                 | done           |
| Tensor& operator+(const Tensor & other);  | done           |
| Tensor& operator+(Scalar other);          |                |
| Tensor& operator-(const Tensor & other);  | done           |
| Tensor& operator-(Scalar other);          |                |
| Tensor& operator*(const Tensor & other);  | done           |
| Tensor& operator*(Scalar other);          |                |
| Tensor& operator/(const Tensor & other);  | done           |
| Tensor& operator/(Scalar other);          |                |
| Tensor operator[](int64_t idx) const;     | done           |
| Tensor& operator+=(const Tensor & other); |                |
| Tensor& operator+=(Scalar other);         |                |
| Tensor& operator-=(const Tensor & other); |                |
| Tensor& operator-=(Scalar other);         |                |
| Tensor& operator*=(const Tensor & other); |                |
| Tensor& operator*=(Scalar other);         |                |
| Tensor& operator/=(const Tensor & other); |                |
| Tensor& operator/=(Scalar other);         |                |


## fill/scatter
| at::Tensor method                                                                  | atnn::Variable |
| :---                                                                               | :--:           |
| Tensor & fill_(Scalar value);                                                      |                |
| bool is_contiguous() const;                                                        | done           |
| Tensor & masked_fill_(const Tensor & mask, Scalar value);                          |                |
| Tensor & masked_fill_(const Tensor & mask, const Tensor & value);                  |                |
| Tensor & masked_scatter_(const Tensor & mask, const Tensor & source);              |                |
| Tensor masked_select(const Tensor & mask) const;                                   |                |
| Tensor transpose(int64_t dim0, int64_t dim1) const;                                | done           |
| Tensor t() const;                                                                  | done           |
| Tensor nonzero() const;                                                            |                |
| Tensor contiguous() const;                                                         | done           |
| Tensor clone() const;                                                              |                |
| Tensor view(IntList size) const;                                                   | done           |
| Tensor index_select(int64_t dim, const Tensor & index) const;                      | done           |
| Tensor take(const Tensor & index) const;                                           |                |
| Tensor & put_(const Tensor & index, const Tensor & source, bool accumulate=false); |                |
| Tensor & index_add_(int64_t dim, const Tensor & index, const Tensor & source);     |                |
| Tensor & index_fill_(int64_t dim, const Tensor & index, Scalar value);             |                |
| Tensor & index_fill_(int64_t dim, const Tensor & index, const Tensor & value);     |                |
| Tensor narrow(int64_t dimension, int64_t start, int64_t length) const;             |                |
| Tensor unfold(int64_t dimension, int64_t size, int64_t step) const;                |                |
| Tensor & scatter_(int64_t dim, const Tensor & index, const Tensor & src);          |                |
| Tensor & scatter_(int64_t dim, const Tensor & index, Scalar value);                |                |
| Tensor & scatter_add_(int64_t dim, const Tensor & index, const Tensor & src);      |                |
| Tensor gather(int64_t dim, const Tensor & index) const;                            |                |


## logical functions

maybe unused (cannot backprop?) but useful for mask ops

| at::Tensor method                              | atnn::Variable   |
| :---                                           | :--:             |
| bool equal(const Tensor & other) const;        | (use operator==) |
| Tensor __and__(Scalar other) const;            |                  |
| Tensor __and__(const Tensor & other) const;    |                  |
| Tensor __or__(Scalar other) const;             |                  |
| Tensor __or__(const Tensor & other) const;     |                  |
| Tensor __xor__(Scalar other) const;            |                  |
| Tensor __xor__(const Tensor & other) const;    |                  |
| Tensor __lshift__(Scalar other) const;         |                  |
| Tensor __lshift__(const Tensor & other) const; |                  |
| Tensor __rshift__(Scalar other) const;         |                  |
| Tensor __rshift__(const Tensor & other) const; |                  |
| Tensor lt(Scalar other) const;                 |                  |
| Tensor lt(const Tensor & other) const;         |                  |
| Tensor gt(Scalar other) const;                 |                  |
| Tensor gt(const Tensor & other) const;         |                  |
| Tensor le(Scalar other) const;                 |                  |
| Tensor le(const Tensor & other) const;         |                  |
| Tensor ge(Scalar other) const;                 |                  |
| Tensor ge(const Tensor & other) const;         |                  |
| Tensor eq(Scalar other) const;                 |                  |
| Tensor eq(const Tensor & other) const;         |                  |
| Tensor ne(Scalar other) const;                 |                  |
| Tensor ne(const Tensor & other) const;         |                  |
| bool all() const;                              | done             |
| bool any() const;                              | done             |




## stat functions (all )
| at::Tensor method                                                                                     | atnn::Variable |
| :---                                                                                                  | :--:           |
| std::tuple<Tensor,Tensor> min(int64_t dim, bool keepdim=false) const;                                 |                |
| Tensor min(const Tensor & other) const;                                                               |                |
| Tensor min() const;                                                                                   |                |
| std::tuple<Tensor,Tensor> max(int64_t dim, bool keepdim=false) const;                                 |                |
| Tensor max(const Tensor & other) const;                                                               |                |
| Tensor max() const;                                                                                   |                |
| std::tuple<Tensor,Tensor> kthvalue(int64_t k, int64_t dim=-1, bool keepdim=false) const;              |                |
| std::tuple<Tensor,Tensor> mode(int64_t dim=-1, bool keepdim=false) const;                             |                |
| std::tuple<Tensor,Tensor> median(int64_t dim, bool keepdim=false) const;                              |                |
| Tensor median() const;                                                                                |                |
| std::tuple<Tensor,Tensor> sort(int64_t dim=-1, bool descending=false) const;                          |                |
| std::tuple<Tensor,Tensor> topk(int64_t k, int64_t dim=-1, bool largest=true, bool sorted=true) const; |                |
| Tensor mean(int64_t dim, bool keepdim=false) const;                                                   |                |
| Tensor mean() const;                                                                                  |                |
| Tensor var(int64_t dim, bool unbiased=true, bool keepdim=false) const;                                |                |
| Tensor var(bool unbiased=true) const;                                                                 |                |
| Tensor std(int64_t dim, bool unbiased=true, bool keepdim=false) const;                                |                |
| Tensor std(bool unbiased=true) const;                                                                 |                |
| Tensor norm(Scalar p, int64_t dim, bool keepdim=false) const;                                         |                |
| Tensor norm(Scalar p=2) const;                                                                        |                |
| Tensor renorm(Scalar p, int64_t dim, Scalar maxnorm) const;                                           |                |
| Tensor dist(const Tensor & other, Scalar p=2) const;                                                  |                |
| Tensor reciprocal() const;                                                                            |                |


## math functions
| at::Tensor method                                                 | atnn::Variable |
| :---                                                              | :--:           |
| Tensor abs() const;                                               |                |
| Tensor sigmoid() const;                                           | (use Sigmoid)  |
| Tensor log() const;                                               |                |
| Tensor log1p() const;                                             |                |
| Tensor lgamma() const;                                            |                |
| Tensor exp() const;                                               |                |
| Tensor cos() const;                                               |                |
| Tensor acos() const;                                              |                |
| Tensor cosh() const;                                              |                |
| Tensor sin() const;                                               |                |
| Tensor asin() const;                                              |                |
| Tensor sinh() const;                                              |                |
| Tensor tan() const;                                               |                |
| Tensor atan() const;                                              |                |
| Tensor tanh() const;                                              | (use Tanh)     |
| Tensor erf() const;                                               |                |
| Tensor erfinv() const;                                            |                |
| Tensor sqrt() const;                                              | done           |
| Tensor rsqrt() const;                                             |                |
| Tensor ceil() const;                                              |                |
| Tensor floor() const;                                             |                |
| Tensor round() const;                                             |                |
| Tensor trunc() const;                                             |                |
| Tensor frac() const;                                              |                |
| Tensor neg() const;                                               |                |
| Tensor atan2(const Tensor & other) const;                         |                |
| Tensor pow(Scalar exponent) const;                                | (double only)  |
| Tensor pow(const Tensor & exponent) const;                        | done           |
| Tensor lerp(const Tensor & end, Scalar weight) const;             |                |
| Tensor histc(int64_t bins=100, Scalar min=0, Scalar max=0) const; |                |
| Tensor sum(int64_t dim, bool keepdim=false) const;                | done           |
| Tensor sum() const;                                               | done           |
| Tensor prod(int64_t dim, bool keepdim=false) const;               |                |
| Tensor prod() const;                                              |                |
| Tensor cumsum(int64_t dim) const;                                 |                |
| Tensor cumprod(int64_t dim) const;                                |                |
| Tensor sign() const;                                              |                |
| Tensor trace() const;                                             |                |
| Tensor fmod(Scalar other) const;                                  |                |
| Tensor fmod(const Tensor & other) const;                          |                |
| Tensor remainder(Scalar other) const;                             |                |
| Tensor remainder(const Tensor & other) const;                     |                |
| Tensor clamp(Scalar min, Scalar max) const;                       |                |
| Tensor clamp(Scalar min) const;                                   |                |


## Matrix (BLAS) functions
| at::Tensor method                                                                                  | atnn::Variable |
| :---                                                                                               | :--:           |
| Tensor dot(const Tensor & tensor) const;                                                           | (use Linear)   |
| Tensor tril(int64_t diagonal=0) const;                                                             |                |
| Tensor triu(int64_t diagonal=0) const;                                                             |                |
| Tensor cross(const Tensor & other, int64_t dim=-1) const;                                          |                |
| Tensor diag(int64_t diagonal=0) const;                                                             |                |
| Tensor addmm(const Tensor & mat1, const Tensor & mat2, Scalar beta=1, Scalar alpha=1) const;       | (use Linear)   |
| Tensor addmv(const Tensor & mat, const Tensor & vec, Scalar beta=1, Scalar alpha=1) const;         | (use Linear)   |
| Tensor addr(const Tensor & vec1, const Tensor & vec2, Scalar beta=1, Scalar alpha=1) const;        |                |
| Tensor ger(const Tensor & vec2) const;                                                             |                |
| Tensor mv(const Tensor & vec) const;                                                               |                |
| Tensor mm(const Tensor & mat2) const;                                                              |                |
| Tensor bmm(const Tensor & mat2) const;                                                             |                |
| Tensor addbmm(const Tensor & batch1, const Tensor & batch2, Scalar beta=1, Scalar alpha=1) const;  |                |
| Tensor baddbmm(const Tensor & batch1, const Tensor & batch2, Scalar beta=1, Scalar alpha=1) const; |                |
| Tensor addcmul(const Tensor & tensor1, const Tensor & tensor2, Scalar value=1) const;              |                |
| Tensor addcdiv(const Tensor & tensor1, const Tensor & tensor2, Scalar value=1) const;              |                |


## Matrix (LAPACK) functions

maybe more difficult to backprop

| at::Tensor method                                                                                                         | atnn::Variable |
| :---                                                                                                                      | :--:           |
| std::tuple<Tensor,Tensor> gesv(const Tensor & A) const;                                                                   |                |
| std::tuple<Tensor,Tensor> gels(const Tensor & A) const;                                                                   |                |
| std::tuple<Tensor,Tensor> trtrs(const Tensor & A, bool upper=true, bool transpose=false, bool unitriangular=false) const; |                |
| std::tuple<Tensor,Tensor> symeig(bool eigenvectors=false, bool upper=true) const;                                         |                |
| std::tuple<Tensor,Tensor> eig(bool eigenvectors=false) const;                                                             |                |
| std::tuple<Tensor,Tensor,Tensor> svd(bool some=true) const;                                                               |                |
| Tensor inverse() const;                                                                                                   |                |
| Tensor potrf(bool upper=true) const;                                                                                      |                |
| Tensor potrs(const Tensor & input2, bool upper=true) const;                                                               |                |
| Tensor potri(bool upper=true) const;                                                                                      |                |
| std::tuple<Tensor,Tensor> pstrf(bool upper=true, Scalar tol=-1) const;                                                    |                |
| std::tuple<Tensor,Tensor> qr() const;                                                                                     |                |
| std::tuple<Tensor,Tensor> geqrf() const;                                                                                  |                |
| Tensor orgqr(const Tensor & input2) const;                                                                                |                |
| Tensor ormqr(const Tensor & input2, const Tensor & input3, bool left=true, bool transpose=false) const;                   |                |
| std::tuple<Tensor,Tensor> btrifact(const Tensor & info={}, bool pivot=true) const;                                        |                |
| Tensor btrisolve(const Tensor & LU_data, const Tensor & LU_pivots) const;                                                 |                |


## random functions

implement stochastic functions by policy gradient

| at::Tensor method                                                                                     | atnn::Variable |
| :---                                                                                                  | :--:           |
| Tensor & random_(int64_t from, int64_t to, Generator * generator=nullptr);                            |                |
| Tensor & random_(int64_t to, Generator * generator=nullptr);                                          |                |
| Tensor & random_(Generator * generator=nullptr);                                                      |                |
| Tensor multinomial(int64_t num_samples, bool replacement=false, Generator * generator=nullptr) const; |                |
| Tensor & uniform_(double from=0, double to=1, Generator * generator=nullptr);                         |                |
| Tensor & normal_(double mean=0, double std=1, Generator * generator=nullptr);                         |                |
| Tensor & cauchy_(double median=0, double sigma=1, Generator * generator=nullptr);                     |                |
| Tensor & log_normal_(double mean=1, double std=2, Generator * generator=nullptr);                     |                |
| Tensor & exponential_(double lambd=1, Generator * generator=nullptr);                                 |                |
| Tensor & geometric_(double p, Generator * generator=nullptr);                                         |                |


## shape functions
| at::Tensor method                                                                 | atnn::Variable |
| :---                                                                              | :--:           |
| Tensor slice(int64_t start=0, int64_t end=9223372036854775807, int64_t step=1, int64_t dim=0) const; | |
| int64_t numel() const;                                                            | done           |
| Tensor select(int64_t dim, int64_t sliceIndex) const;                             | done           |
| Tensor _unnarrow(int64_t dimension, int64_t offset, int64_t dimSize) const;       |                |
| Tensor & assign_(const Tensor & src);                                             |                |
| Tensor as_strided(IntList size, IntList stride, int64_t storage_offset=-1) const; |                |
| Tensor & as_strided_(IntList size, IntList stride, int64_t storage_offset=-1);    |                |
| Tensor & reshape_(IntList size, IntList stride);                                  |                |
| Tensor type_as(const Tensor & other) const;                                       |                |
| Tensor expand_as(const Tensor & other) const;                                     |                |
| std::vector<Tensor> split(int64_t split_size, int64_t dim=0) const;               | done           |
| std::vector<Tensor> chunk(int64_t chunks, int64_t dim=0) const;                   | done           |
| int64_t size(int64_t dim) const;                                                  | done           |
| int64_t stride(int64_t dim) const;                                                |                |
| bool is_same_size(const Tensor & other) const;                                    | done           |
| Tensor permute(IntList dims) const;                                               | done           |
| Tensor expand(IntList size) const;                                                |                |
| Tensor squeeze() const;                                                           | done           |
| Tensor squeeze(int64_t dim) const;                                                | done           |
| Tensor unsqueeze(int64_t dim) const;                                              | done           |
| bool is_signed() const;                                                           | done           |


## other
| at::Tensor method           | atnn::Variable |
| :---                        | :--:           |
| int64_t get_device() const; | done           |
