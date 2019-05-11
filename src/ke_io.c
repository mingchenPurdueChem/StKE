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
#include <string.h>
#include "json-c/json.h"
#include "ke.h"

#if DEBUG
#ifdef DEBUG_JSON
#define CHECK_PTR(x, msg) \
	if (!x) { \
		printf("FAILED at  %s\n", msg); \
		goto clean; \
	} else { \
		printf("PASSED %s\n", msg); \
	}

#endif 	//DEBUG_JSON
#endif  //DEBUG

#ifndef CHECK_PTR
#define CHECK_PTR(x, msg) \
	if (!x) goto clean;

#endif

#define JSON_SAFE_OBJECT_GET(jobj, TAG, jout) \
	jret = json_object_object_get_ex(jobj, TAG, &jout); \
	if (!jret) goto clean; 


#define JSON_VERSION        1000
#define JSON_TAG_VERSION    "version"
#define JSON_TAG_LAYERS     "layers"
#define JSON_TAG_NUM_LAYERS "num_layers"
#define JSON_TAG_W_DIM0     "w_dim0"
#define JSON_TAG_W_DIM1     "w_dim1"
#define JSON_TAG_WEIGHTS    "weights"
#define JSON_TAG_B_DIM0     "b_dim0"
#define JSON_TAG_BIASES     "biases"
#define JSON_TAG_ACTFUNC    "activation"

// Version sepcific parsers.
static ke_mlp* parse_json_v1(json_object *jobj);
static void pack_json_v1(const ke_mlp *mlp, json_object *jobj);

static const char * act_func_type_to_str(act_func_t act)
{
	return ACT_FUNC_NAME[(int)(act)];
}

static act_func_t act_func_type_from_str(const char *str)
{
	int i;
	const int num_act_types = sizeof(ACT_FUNC_NAME)/sizeof(ACT_FUNC_NAME[0]);
	for (i=0; i<num_act_types; ++i) {
		if (strncmp(str, ACT_FUNC_NAME[i], strlen(ACT_FUNC_NAME[i])) == 0) {
			return (act_func_t)(i);
		}
	}
	fprintf(stderr, "WARNING: PARSE `%s' as IDENTITY.", str);
	return IDENTITY;
}

ke_mlp*
ke_mlp_read_from_json(const char *fname)
{
	ke_mlp *mlp = NULL;
	json_bool jret;
	json_object *jtmp = NULL;
	json_object *jobj = json_object_from_file(fname);
	CHECK_PTR(jobj, "JSON");

	JSON_SAFE_OBJECT_GET(jobj, JSON_TAG_VERSION, jtmp);
	const int version = json_object_get_int(jtmp);
	const int ver_major = version / 1000;
	const int ver_minor = version % 1000;

	(void) ver_minor;
	(void) ver_major;
	if (ver_major == 1) {
		mlp = parse_json_v1(jobj);
	} else {
		mlp = NULL;
	}

exit:
	json_object_put(jobj);
    return mlp;

clean:
	goto exit;
}

static
ke_mlp* 
parse_json_v1(json_object *jobj)
{
	int i, j;
	json_bool jret;
	ke_mlp *mlp = NULL;
	json_object *jtmp = NULL;
	if (jobj) {
		JSON_SAFE_OBJECT_GET(jobj, JSON_TAG_NUM_LAYERS, jtmp);

		int num_layers = json_object_get_int(jtmp);
		mlp = alloc_ke_mlp(num_layers);
		json_object *jarray = NULL;
		JSON_SAFE_OBJECT_GET(jobj, JSON_TAG_LAYERS, jarray);

		for (i=0; i<num_layers; ++i) {
			json_object *jlayer = json_object_array_get_idx(jarray, i);
			CHECK_PTR(jlayer, "LAYER i");
			
			// Parse W Matrix
			JSON_SAFE_OBJECT_GET(jlayer, JSON_TAG_W_DIM0, jtmp);
			int w_dim0 = json_object_get_int(jtmp);
			JSON_SAFE_OBJECT_GET(jlayer, JSON_TAG_W_DIM1, jtmp);
			int w_dim1 = json_object_get_int(jtmp);
			mlp->w_mats[i] = alloc_ke_matrix(w_dim0, w_dim1);
			CHECK_PTR(mlp->w_mats[i], "alloc w_mat");

			json_object *jweights = NULL;
			JSON_SAFE_OBJECT_GET(jlayer, JSON_TAG_WEIGHTS, jweights);
			for (j=0; j<w_dim0*w_dim1; ++j) {
				jtmp = json_object_array_get_idx(jweights, j);
				CHECK_PTR(jtmp, "W i");
				mlp->w_mats[i]->data[j] = (REAL)json_object_get_double(jtmp);
			}

			// Parse B Vector
			JSON_SAFE_OBJECT_GET(jlayer, JSON_TAG_B_DIM0, jtmp);
			int b_dim = json_object_get_int(jtmp);
			mlp->b_vecs[i] = alloc_ke_vector(b_dim);
			CHECK_PTR(mlp->b_vecs[i], "alloc b_vec");

			json_object *jbias = NULL;
			JSON_SAFE_OBJECT_GET(jlayer, JSON_TAG_BIASES, jbias);
			for (j=0; j<b_dim; ++j) {
				jtmp = json_object_array_get_idx(jbias, j);
				CHECK_PTR(jtmp, "B i");
				mlp->b_vecs[i]->data[j] = (REAL)json_object_get_double(jtmp);
			}

			// Parse Activation Func
			JSON_SAFE_OBJECT_GET(jlayer, JSON_TAG_ACTFUNC, jtmp);
			const char * act_name = json_object_get_string(jtmp);
			mlp->a_types[i] = act_func_type_from_str(act_name);
		}
	}

exit:
	return mlp;

clean:
	free_ke_mlp(mlp);
	mlp = NULL;
	goto exit;
}

static 
json_object*
ke_mlp_to_json(const ke_mlp *mlp)
{
	const int version = JSON_VERSION;
    const int ver_major = version / 1000;
    const int ver_minor = version % 1000;

    json_object *jobj = json_object_new_object();
	json_object_object_add(jobj, JSON_TAG_VERSION, 
						   json_object_new_int(version));

	if (ver_major == 1) {
		pack_json_v1(mlp, jobj);
	} else {
		fprintf(stderr, "JSON_VERSION `%d' is not supported!", 
				version);
	}
	return jobj;
}

static
void
pack_json_v1(const ke_mlp *mlp, json_object *jobj)
{
	int i, j;
    if (mlp) {
        int num_layers = mlp->num_layers;
        json_object_object_add(jobj, JSON_TAG_NUM_LAYERS, 
                               json_object_new_int(num_layers));
        json_object *jarray = json_object_new_array();

        for (i=0; i<num_layers; ++i) {
            json_object *jlayer = json_object_new_object();
            const ke_matrix *w_mat = mlp->w_mats[i];
            if (w_mat) {
                json_object_object_add(jlayer, JSON_TAG_W_DIM0,
                                       json_object_new_int(w_mat->dims[0]));
                json_object_object_add(jlayer, JSON_TAG_W_DIM1,
                                       json_object_new_int(w_mat->dims[1]));
                json_object *jweights = json_object_new_array();
                for (j=0; j<w_mat->dims[0]*w_mat->dims[1]; ++j) {
                    json_object_array_add(jweights, 
                                          json_object_new_double(w_mat->data[j]));
                }
                json_object_object_add(jlayer, JSON_TAG_WEIGHTS, jweights);
            }

            const ke_vector *b_vec = mlp->b_vecs[i];
            if (b_vec) {
                json_object_object_add(jlayer, JSON_TAG_B_DIM0,            
                                       json_object_new_int(b_vec->dims));
                json_object *jbias = json_object_new_array();
                for (j=0; j<b_vec->dims; ++j) {
                    json_object_array_add(jbias,
                                          json_object_new_double(b_vec->data[j]));
                }
                json_object_object_add(jlayer, JSON_TAG_BIASES, jbias);
            }
			json_object_object_add(jlayer, JSON_TAG_ACTFUNC, 
				   json_object_new_string(act_func_type_to_str(mlp->a_types[i])));
			json_object_array_add(jarray, jlayer);
        }
        json_object_object_add(jobj, JSON_TAG_LAYERS, jarray);
    }
    return;
}

void
ke_mlp_write_to_json(const ke_mlp *mlp, const char *fname)
{
    json_object *jobj = ke_mlp_to_json(mlp);
    FILE *fp = fopen(fname, "w");
    if (fp) {
        fprintf(fp, "%s", json_object_to_json_string_ext(jobj, JSON_C_TO_STRING_PRETTY));
    }

exit:
    json_object_put(jobj);
	if (fp) fclose(fp);
}

