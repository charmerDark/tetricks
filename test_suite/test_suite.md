einsum("i,i->", a, b) // Vector dot product  
einsum("ij,j->i", A, x) // TTV
einsum("ik,kj->ij", A, B) // TTM 
einsum("bij,bjk->bik", A, B) // Batch matrix multiplication  
einsum("i,j->ij", a, b) // Outer product of two vectors  
sqrt(einsum("ij,ij->", A, A)) // Frobenius norm of a matrix  
einsum("ij,ij->", S, T) // Tensor contraction (stress-strain work)  
einsum("i,ij,j->", x, A, y) // Bilinear form  
einsum("ijk,ijk->", D, W) // Weighted sum over multiple axes  
einsum("ii", A) // Trace of a matrix  

einsum("ij,ij->", A, A) + einsum("ij,ij->", B, B) -  C *einsum("ij,ij->", A, B) // Frobenius norm difference between matrices  
einsum("i,i->", I, R) * A + einsum("i,i->", V, I) * A // Energy in an electrical network  
einsum("i,ij,j->", x, A, y) - einsum("i,ij,j->", x, B, y) // Bilinear form difference  
einsum("ijk,ijk->ij", S, W1) / einsum("ijk,ijk->ij", S, W2) // Elementwise ratio of two contracted tensors  
einsum("ij,ij->", S, T) / einsum("ij,ij->", T, T) // Stress-to-strain energy ratio  

einsum("bqd,bkd->bqk", Q, K) / sqrt(d) // Self-attention score computation  
einsum("bqk,bvd->bqd", P, V) // Softmax attention output  
einsum("hbqd,hdo->bqo", H, O) // Multi-head attention aggregation  
einsum("bihw,oihw->bo", P, F) // Convolution via Toeplitz multiplication  
einsum("bik,bjk->ij", X, Y) + λ * einsum("ij->ij", W) // Batch gradient accumulation with L2 regularization  
einsum("bi,bj->ij", U, V) / n // Cross-covariance in representation learning  
einsum("bci,bcj->ij", F, F) / (B * C) // Gram matrix for style transfer  
einsum("ij,j->i", J, v) - einsum("ij,j->i", J0, v) // Jacobian–vector product difference  
einsum("bik,kl,bjl->bij", H1, M, H2) // Tensor contraction for bilinear attention  
einsum("bnhd,hdf->bnf", X, W1) / einsum("bnhd,hdf->bnf", X, W2) // Factorized 4D weight multiplication with normalization  