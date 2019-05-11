/******************************************************************************/
/*    Copyright 2018 Jing Zhang and Ming Chen                                 */   
/*                                                                            */
/*    This file is part of StKE, version 1.0                                  */
/*                                                                            */
/*    StKE is free software: you can redistribute it and/or modify            */
/*    it under the terms of the GNU General Public License as published by    */
/*    the Free Software Foundation, either version 3 of the License, or       */
/*    (at your option) any later version.                                     */
/*                                                                            */
/*    StKE is distributed in the hope that it will be useful,                 */
/*    but WITHOUT ANY WARRANTY; without even the implied warranty of          */
/*    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the           */
/*    GNU General Public License for more details.                            */
/*                                                                            */
/*    You should have received a copy of the GNU General Public License       */
/*    along with StKE.  If not, see <https://www.gnu.org/licenses/>.          */
/******************************************************************************/


#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <string.h>
#include "ke.h"
#include "cblas.h"

/************************
 * Basic data structures
 ************************/
ke_vector* 
alloc_ke_vector(int length)
{
	ke_vector* ptr = NULL;
	if (length >= 0) {
		ptr = (ke_vector*)malloc(sizeof(ke_vector));

		if (ptr) {
			ptr->dims = length;
			ptr->data = (REAL*)calloc(sizeof(REAL), length);

			if (!ptr->data) goto clean;
		}
	}
	return ptr;	

clean:
	free_ke_vector(ptr);
	return NULL;
}

void 
free_ke_vector(ke_vector* ptr)
{
	if (ptr) {
		ptr->dims = 0;
		free(ptr->data);
		ptr->data = NULL;
		free(ptr);
	}
}

ke_matrix* 
alloc_ke_matrix(int rows, int cols)
{
	ke_matrix* ptr = NULL;
	if (rows >= 0 && cols >= 0) {
		ptr = (ke_matrix*)malloc(sizeof(ke_matrix));
		
		if (ptr) {
			ptr->dims[0] = rows;
			ptr->dims[1] = cols;
			ptr->data = (REAL*)calloc(sizeof(REAL), rows*cols);

			if (!ptr->data) goto clean;
		} else {
			goto clean;
		}
	}
	return ptr;

clean:
	free_ke_matrix(ptr);
	return NULL;
}

ke_matrix*
clone_ke_matrix(const ke_matrix* src)
{
	ke_matrix* ptr = NULL;
	int row, col;
	if (src) {
		row = src->dims[0];
		col = src->dims[1];
		ptr = alloc_ke_matrix(row, col);
		if(ptr) {
			(void)memcpy(ptr->data, src->data, row*col*sizeof(REAL));
		} else {
			goto clean;
		}
	}
	return ptr;

clean:
	free_ke_matrix(ptr);
	return NULL;
}

void 
free_ke_matrix(ke_matrix* ptr)
{
	if (ptr) {
		ptr->dims[0] = 0;
		ptr->dims[1] = 0;
		free(ptr->data);
		ptr->data = NULL;
		free(ptr);
	}
}

void
ke_matrix_print(const ke_matrix* ptr, FILE *out)
{
	int i, j;
	if (ptr) {
		const int lda = ptr->dims[1];
		for (i=0; i<ptr->dims[0]; ++i) {
			for (j=0; j<ptr->dims[1]; ++j) {
				fprintf(out, "%8.6e ", ptr->data[i*lda+j]);
			}
			fprintf(out, "\n");
		}
		fprintf(out, "\n");
	}
}

ke_tensor3d* 
alloc_ke_tensor3d(int dim0, int dim1, int dim2)
{
	ke_tensor3d* ptr = NULL;
	if (dim0 >=0 && dim1 >= 0 && dim2 >= 0) {
		ptr = (ke_tensor3d*)malloc(sizeof(ke_tensor3d));
		
		if (ptr) {
			ptr->dims[0] = dim0;
			ptr->dims[1] = dim1;
			ptr->dims[2] = dim2;
			ptr->data = (REAL*)calloc(sizeof(REAL), dim0*dim1*dim2);

			if (!ptr->data) goto clean;
		} else {
			goto clean;
		}
	}
	return ptr;

clean:
	free_ke_tensor3d(ptr);
	return NULL;
}

void 
free_ke_tensor3d(ke_tensor3d* ptr)
{
	if (ptr) {
		ptr->dims[0] = 0;
		ptr->dims[1] = 0;
		ptr->dims[2] = 0;
		free(ptr->data);
		ptr->data = NULL;
		free(ptr);
	}
}

void
ke_tensor3d_print(const ke_tensor3d* ptr, FILE *fp)
{
	int i, j, k;
	if (ptr) {
		//const int dim0 = ptr->dims[0];
		const int dim1 = ptr->dims[1];
		const int dim2 = ptr->dims[2];
		fprintf(fp, "---\n");
		for (i=0; i<ptr->dims[0]; ++i) {
			for (j=0; j<ptr->dims[1]; ++j) {
				for (k=0; k<ptr->dims[2]; ++k) {
					fprintf(fp, "%8.6e ", ptr->data[i*dim1*dim2+j*dim2+k]);
				}
				fprintf(fp, "\n");
			}
			fprintf(fp, "\n");
		}
		fprintf(fp, "---\n");
	}
}

ke_mlp* 
alloc_ke_mlp(int num_layers)
{
	ke_mlp* ptr = NULL;
	if (num_layers > 0) {
		ptr = (ke_mlp*)malloc(sizeof(ke_mlp));
		if (!ptr) goto clean;

		ptr->num_layers = num_layers;
		ptr->w_mats = (ke_matrix**)calloc(sizeof(ke_matrix*), num_layers);
		ptr->b_vecs = (ke_vector**)calloc(sizeof(ke_vector*), num_layers);
		ptr->a_types = (act_func_t*)calloc(sizeof(act_func_t), num_layers);

		if (!ptr->w_mats || !ptr->b_vecs || !ptr->a_types) 
			goto clean;
	}
	return ptr;

clean:
	free_ke_mlp(ptr);	
	return NULL;
}

void 
free_ke_mlp(ke_mlp* ptr)
{
	int i;
	if (ptr) {
		if (ptr->w_mats) {
			for (i=0; i<ptr->num_layers; ++i) {
				free_ke_matrix(ptr->w_mats[i]);
				ptr->w_mats[i] = NULL;
			}
			free(ptr->w_mats);
			ptr->w_mats = NULL;
		}
		if (ptr->b_vecs) {
			for (i=0; i<ptr->num_layers; ++i) {
				free_ke_vector(ptr->b_vecs[i]);
				ptr->b_vecs[i] = NULL;
			}
			free(ptr->b_vecs);
			ptr->b_vecs = NULL;
		}
		free(ptr->a_types);
		ptr->a_types = NULL;
		ptr->num_layers = 0;
		free(ptr);
	}
}

/******************
 * Math
 ******************/

static
void ke_gemm(const ke_matrix* x, const ke_matrix* w,
			 ke_matrix* z)
{
	if (x && w && z) {
		const int M = x->dims[0];		// # of rows of x
		const int N = w->dims[1]; 		// # of cols of w
		const int K = x->dims[1]; 		// # of cols of x, and # of rows of w
		REAL alpha = 1.0;
		REAL beta = 0.0;
		const int lda = K;
		const int ldb = N;
		const int ldc = N;

		cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
					M, N, K, alpha, x->data, lda, w->data, ldb,
					beta, z->data, ldc);
	}
}

static
void
ke_add_bias(ke_matrix* z, const ke_vector* b)
{
	int i, j;
	if (z && b) {
		assert(z->dims[1] == b->dims);

		const int lda = z->dims[1];
		for (i=0; i<z->dims[0]; ++i) {
			for (j=0; j<z->dims[1]; ++j) {
				z->data[i*lda+j] += b->data[j];
			}
		}
	}
}

static
void
ke_activation(ke_matrix* z, act_func_t act_type)
{
	int i;
	if (act_type == IDENTITY) {
		return;
	} else if (act_type == TANH) {
		int dims = z->dims[0]*z->dims[1];
		for (i=0; i<dims; ++i) {
			z->data[i] = tanh(z->data[i]);
		}
	} else if (act_type == RECTIFY) {
		int dims = z->dims[0]*z->dims[1];
        for (i=0; i<dims; ++i) {
            z->data[i] = fmax(0.0f, z->data[i]); 
        }
	} else {
		assert(0 && "UNKNOWN activation functions");
	}
}

static
ke_matrix*
ke_mlp_one_forward(const ke_matrix* x, const ke_matrix* w, 
				   const ke_vector* b, act_func_t act_type)
{
	assert(x->dims[1] == w->dims[0] && 
		   w->dims[1] == b->dims);
	ke_matrix *z = alloc_ke_matrix(x->dims[0], w->dims[1]);
	ke_gemm(x, w, z);
	ke_add_bias(z, b);
	ke_activation(z, act_type);

	return z;
}

static
void
ke_grad_activation(ke_matrix* z, act_func_t act_type)
{
    int i;
	if (act_type == IDENTITY) {
		int dims = z->dims[0]*z->dims[1];
        for (i=0; i<dims; ++i) {
            z->data[i] = 1.0f;
        }
	} else if (act_type == TANH) {
		int dims = z->dims[0]*z->dims[1];
        for (i=0; i<dims; ++i) {
            z->data[i] = 1.0f - z->data[i]*z->data[i];
        }
	} else if (act_type == RECTIFY) {
		int dims = z->dims[0]*z->dims[1];
        for (i=0; i<dims; ++i) {
			z->data[i] = (z->data[i] < 0.f) ? 0.f : (z->data[i] > 0.f);
        }
	} else {
		assert(0 && "UNKNOWN activation functions");
	}
}

static
ke_tensor3d*
ke_mlp_grad_init(ke_matrix* z, const ke_matrix* w,
				 const ke_vector* b, act_func_t act_type)
{
	int i, n, j;
	const int J = z->dims[1];
	const int N = z->dims[0];
	const int IL = w->dims[0];
	assert(w->dims[1] == J);

	// f'(z(n, j)) * w(i, j)
	ke_tensor3d *gz = alloc_ke_tensor3d(J, N, IL);
	ke_grad_activation(z, act_type);
	
	for (j=0; j<J; ++j) {
		for (n=0; n<N; ++n) {
			for (i=0; i<IL; ++i) {
				gz->data[j*N*IL + n*IL + i] = z->data[n*J+j] * w->data[i*J+j];
			}
		}
	}
	return gz;
}

static
ke_tensor3d*
ke_mlp_grad_backward(ke_matrix* z, ke_tensor3d* g,
					 const ke_matrix* w,
                     const ke_vector* b, act_func_t act_type)
{
	// g(j, n, i_L)
	// z(n, i_L)
	// w(i_{L-1}, i_L)
	// gz(j, n, i_{L-1})
	
	int i, j;
	const int J = g->dims[0];
	const int N = g->dims[1];
	const int IL = g->dims[2];
	const int ILm = w->dims[0];
	
	assert(z->dims[0] == N && z->dims[1] == IL);
	assert(w->dims[1] == IL);
	
	ke_tensor3d *gz = alloc_ke_tensor3d(J, N, ILm);

	// g(j, n, i_L) * f'(z(n, i_L))
	ke_grad_activation(z, act_type);
	const int len = N*IL;
	for (j=0; j<J; ++j) {
		for (i=0; i<len; ++i) {
			g->data[j*len+i] *= z->data[i];
		}
	}

	// g(j, n, i_L) * tran(w(i_{L-1}, i_L))
	{
		const int MM = J*N;				// # of rows of g
		const int NN = ILm;				// # of cols of trans(w)
		const int KK = IL;				
        REAL alpha = 1.0;
        REAL beta = 0.0;
        const int lda = KK;
        const int ldb = KK;
        const int ldc = NN;

        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    MM, NN, KK, alpha, g->data, lda, w->data, ldb,
                    beta, gz->data, ldc);
	}
	return gz;
}

void
ke_mlp_compute_grads(const ke_mlp* mlp, const ke_matrix* x_mat,
					 ke_matrix** fwd_out, ke_tensor3d** bwd_grad)
{
	int i;
	ke_matrix   **fwd_layers = NULL;
	int num_layers = 0;
	if (mlp && x_mat) {
		num_layers = mlp->num_layers;
		if (num_layers<=0) goto exit;

	    fwd_layers = (ke_matrix**)calloc(sizeof(ke_matrix*), num_layers);
		if (!fwd_layers) goto exit;

		// run forward 
		const ke_matrix *last_x = x_mat;
		for (i=0; i<num_layers; ++i) {
			fwd_layers[i] = ke_mlp_one_forward(last_x, mlp->w_mats[i], 
											   mlp->b_vecs[i], mlp->a_types[i]);
			last_x = fwd_layers[i];
			//#fprintf(stdout, "FWD : %d\n", i);
			//#ke_matrix_print(last_x, stdout);
		}
		*fwd_out = clone_ke_matrix(last_x);

		// run backward
		ke_tensor3d *last_gz = NULL;
		for (i=num_layers-1; i>=0; --i) {
			ke_tensor3d *gz = NULL;
			if (i==num_layers-1) {
				gz = ke_mlp_grad_init(
						fwd_layers[i], mlp->w_mats[i],
						mlp->b_vecs[i], mlp->a_types[i]);
			} else {
				gz = ke_mlp_grad_backward(
						fwd_layers[i], last_gz, mlp->w_mats[i],
						mlp->b_vecs[i], mlp->a_types[i]);
			}
			//#ke_tensor3d_print(gz, stdout);
			free_ke_tensor3d(last_gz);
			last_gz = gz;
		}		
		*bwd_grad = last_gz;
	}

exit:
	if (fwd_layers) {
		for (i=0; i<num_layers; ++i) {
			free_ke_matrix(fwd_layers[i]);
			fwd_layers[i] = NULL;
		}
		free(fwd_layers);
	}
	return;
}

