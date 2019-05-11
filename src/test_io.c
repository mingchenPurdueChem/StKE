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

#include "ke.h"
#include <stdlib.h>
#include <assert.h>

void test_free_write()
{
	ke_mlp* mlp = alloc_ke_mlp(2);
	free_ke_mlp(mlp);

	// access freed memory ...
	ke_mlp_write_to_json(mlp, "test.json");
}

void test_write_read_null_mlp()
{
	ke_mlp_write_to_json(NULL, "test.json");
	ke_mlp* mlp = ke_mlp_read_from_json("test.json");
	assert(mlp == NULL);
}

void test_empty_mlp()
{
	ke_mlp* mlp = alloc_ke_mlp(2);
    ke_mlp_write_to_json(mlp, "test.json");
    free_ke_mlp(mlp);

    mlp = ke_mlp_read_from_json("test.json");
    ke_mlp_write_to_json(mlp, "test2.json");
    free_ke_mlp(mlp);
}

void test_real_mlp()
{
    int i;
    ke_mlp* mlp = alloc_ke_mlp(2);
    for (i=0; i<2; ++i) {
        mlp->w_mats[i] = alloc_ke_matrix(3,3);
        mlp->b_vecs[i] = alloc_ke_vector(3);
		mlp->a_types[i] = TANH;
    }

    ke_mlp_write_to_json(mlp, "test.json");
	free_ke_mlp(mlp);

	mlp = ke_mlp_read_from_json("test.json");

	mlp->w_mats[0]->data[2] = 0.000012345678;
	mlp->w_mats[0]->data[0] = -0.000012345678;

	ke_mlp_write_to_json(mlp, "test2.json");
	free_ke_mlp(mlp);
}

void test_fwd()
{
	int i;
	ke_matrix *x_mat = alloc_ke_matrix(2, 3);
	//ke_matrix *fwd_out = alloc_ke_matrix(2, 2);
	for (i=0; i<6; ++i) {
		x_mat->data[i] = i * 0.1;
	}
	ke_matrix_print(x_mat, stdout);

    ke_mlp* mlp = alloc_ke_mlp(3);
	mlp->w_mats[0] = alloc_ke_matrix(3, 2);
	mlp->b_vecs[0] = alloc_ke_vector(2);
	mlp->a_types[0] = TANH;

	for (i=0; i<6; ++i) {
		mlp->w_mats[0]->data[i] = i * 0.1;
	}
	ke_matrix_print(mlp->w_mats[0], stdout);
	mlp->b_vecs[0]->data[0] = 0.1;
	mlp->b_vecs[0]->data[1] = -0.1;

	mlp->w_mats[1] = alloc_ke_matrix(2, 2);
	mlp->b_vecs[1] = alloc_ke_vector(2);
	mlp->a_types[1] = TANH;
	mlp->w_mats[1]->data[0] = 0.5;
	mlp->w_mats[1]->data[1] = 2.;
	mlp->w_mats[1]->data[2] = -0.5;
	mlp->w_mats[1]->data[3] = 3.;
	mlp->b_vecs[1]->data[0] = 0.3;
	mlp->b_vecs[1]->data[1] = -0.4;


    mlp->w_mats[2] = alloc_ke_matrix(2, 3);
    mlp->b_vecs[2] = alloc_ke_vector(3);
    mlp->a_types[2] = TANH;
    mlp->w_mats[2]->data[0] = 0.3;
    mlp->w_mats[2]->data[1] = 0.4;
    mlp->w_mats[2]->data[2] = -0.11;
    mlp->w_mats[2]->data[3] = -0.5;
    mlp->w_mats[2]->data[4] = 0.3;
    mlp->w_mats[2]->data[5] = 0.7;
	mlp->b_vecs[2]->data[0] = 0.1;
	mlp->b_vecs[2]->data[1] = -0.4;
	mlp->b_vecs[2]->data[2] = 0.1;

	ke_mlp_write_to_json(mlp, "test2.json");
	ke_tensor3d *grad = NULL;
	ke_matrix *fwd_out = NULL;
	ke_mlp_compute_grads(mlp, x_mat,&fwd_out,&grad);
	ke_tensor3d_print(grad, stdout);

	free_ke_mlp(mlp);
	free_ke_matrix(x_mat);
	free_ke_tensor3d(grad);

/****
 *
 * -1.573186e-01 -6.332517e-01 -1.109185e+00 
 *  -5.167172e-02 -1.833379e-01 -3.150040e-01 
 *
 *  6.834613e-02 3.453924e-01 6.224387e-01 
 *  8.759822e-03 8.917383e-02 1.695878e-01 
 *
 *  2.100867e-01 8.846595e-01 1.559232e+00 
 *  4.264355e-02 1.722681e-01 3.018926e-01
 */
}

int main()
{
	//test_empty_mlp();
	//test_real_mlp();
	//test_write_read_null_mlp();
	test_fwd();
}
