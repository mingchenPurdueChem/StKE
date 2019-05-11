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

#ifndef KINETIC_EMBEDDING_H_
#define KINETIC_EMBEDDING_H_

#ifdef __cplusplus
extern "C" {
#endif

#include <stdio.h>

/**
 * Kinetic Embedding Runtime Gradient Library.
 *
 * -Jing
 *
 * We use the simplest MLP as the mapping function, as shown blow,
 *
 * 		Y_{n,j} = activation(\sum_{i}{X_{n,i}*W_{i,j}} + b_{j})
 * where n is number of data points, i is the input dimension and 
 * j is the output dimension.
 *
 * The gradients are computed,
 *
 * 		G_{j,n,i} = \frac{\partial Y_{n,j}}{\partial X_{n,i}}
 * where j is the output parameter dimension (i.e., alpha, beta, ...),
 * n is the sample index and i is the input parameter dimension (i.e.
 * dihedral anlges ...).
 *
 *
 * Example code:
 *
 * 		const char *fname = "parameters.json";
 * 		ke_mlp* mlp = ke_mlp_read_from_json(fname);
 * 		
 * 		const int num_samples = 1;
 * 		const int num_dihedrals = 20;
 * 		ke_matrix *x_mat = alloc_ke_matrix(num_samples, num_dihedrals);
 *		// assign matrix element values....
 *   	ke_tensor3d *grad = ke_mlp_compute_grads(mlp, x_mat);
 *
 * 		// if you want to print it out, 
 *   	ke_tensor3d_print(grad, stdout);
 *      
 *		// do whatever on gradients...
 */

#define REAL float

struct ke_vector {
	int dims;			// length
	REAL *data;			// data
};
typedef struct ke_vector ke_vector;

struct ke_matrix {
	int dims[2];		// rows, cols
	REAL *data;			// data, row-major
};
typedef struct ke_matrix ke_matrix;

struct ke_tensor3d {
	int dims[3];		// dim 0,1,2
	REAL *data;			// data, row-major
};
typedef struct ke_tensor3d ke_tensor3d;

/**
 * Allocate a vector for given length.
 * @return the vector pointer; NULL if failed.
 */
ke_vector*   alloc_ke_vector(int length);

/**
 * Free a vector.
 * Safe for NULL input.
 */
void 		 free_ke_vector(ke_vector* ptr);

/**
 * Allocate a matrix for given size. (row major)
 * @return the matrix pointer; NULL if failed.
 */
ke_matrix*   alloc_ke_matrix(int rows, int cols);

/**
 * Clone a matrix.
 * @return the matrix pointer; NULL if failed.
 */
ke_matrix* clone_ke_matrix(const ke_matrix* src);

/**
 * Free a matrix.
 * Safe for NULL input.
 */
void	     free_ke_matrix(ke_matrix* ptr);

/**
 * Allocate a 3D tensor for given size. (row major)
 * @return the tensor pointer; NULL if failed.
 */
ke_tensor3d* alloc_ke_tensor3d(int dim0, int dim1, int dim2);

/**
 * Free a 3D tensor.
 * Safe for NULL input.
 */
void         free_ke_tensor3d(ke_tensor3d* ptr);

/**
 * Activation function types.
 * MAKE SURE this matches the name definitions.
 */
typedef enum {IDENTITY = 0, TANH, RECTIFY} act_func_t;

/**
 * Activation function names.
 * MAKE SURE this matches the enum definitions.
 */
static char const * ACT_FUNC_NAME[] = {
    "IDENTITY",
    "TANH",
    "RECTIFY",
};

/**
 * Multi-Layer Perceptron.
 * @note Weight matrices and bias vectors are stored in forward direction.
 * 
 * Example:
 
	// Create a MLP with two Layers.
	ke_mlp *mlp = alloc_ke_mlp(2);
	assert(mlp->num_layers == 2);

	const int input_dim = 10;
	const int layer0_dim = 32;
	const int layer1_dim = 64;

	// allocate 1st layer weights
	mlp->w_mats[0] = alloc_ke_matrix(input_dim, layer0_dim);
	mlp->b_vecs[0] = alloc_ke_vector(layer0_dim);

	// allocate 1st layer weights
    mlp->w_mats[1] = alloc_ke_matrix(layer0_dim, layer1_dim);
    mlp->b_vecs[1] = alloc_ke_vector(layer1_dim);

	// do your work ...

	// Free everything
	free_ke_mlp(mlp);	

 */
struct ke_mlp {
	int 		num_layers;		///< number of neural layers
	ke_matrix 	**w_mats;		///< Weight matrices
	ke_vector 	**b_vecs;		///< Bias vectors
	act_func_t 	*a_types;		///< activation functions
};
typedef struct ke_mlp ke_mlp;

/**
 * Allocate a MLP struct.
 * @return mlp pointer; NULL if failed.
 */ 
ke_mlp*    	alloc_ke_mlp(int num_layers);

/**
 * Free a MLP.
 * @note all matrices and vectors stored in the MLP are freed.
 */
void		free_ke_mlp(ke_mlp* ptr);

/**
 * Read a MLP from a JSON file.
 * @return a MLP pointer; NULL if failed.
 * @note this is NOT efficient to store very large matrices; but
 *       we want to keep the flexibility to read and modify it.
 */
ke_mlp* ke_mlp_read_from_json(const char *fname);

/**
 * Write a MLP to a JSON file.
 */
void ke_mlp_write_to_json(const ke_mlp *mlp, const char *fname);

/**
 * Compute gradients.
 * @return gradient of (output parameter id, sample id, input parameter id).
 */
void ke_mlp_compute_grads(const ke_mlp* mlp, const ke_matrix* x_mat,
                          ke_matrix** fwd_out, ke_tensor3d** bwd_grad);

void ke_matrix_print(const ke_matrix* ptr, FILE *out);
void ke_tensor3d_print(const ke_tensor3d* ptr, FILE *fp);


#ifdef __cplusplus
}  // extern "C" 
#endif

#endif //KINETIC_EMBEDDING_H_
