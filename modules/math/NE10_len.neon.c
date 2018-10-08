/*
 *  Copyright 2011-16 ARM Limited and Contributors.
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions are met:
 *    * Redistributions of source code must retain the above copyright
 *      notice, this list of conditions and the following disclaimer.
 *    * Redistributions in binary form must reproduce the above copyright
 *      notice, this list of conditions and the following disclaimer in the
 *      documentation and/or other materials provided with the distribution.
 *    * Neither the name of ARM Limited nor the
 *      names of its contributors may be used to endorse or promote products
 *      derived from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY ARM LIMITED AND CONTRIBUTORS "AS IS" AND
 *  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 *  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 *  DISCLAIMED. IN NO EVENT SHALL ARM LIMITED AND CONTRIBUTORS BE LIABLE FOR ANY
 *  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 *  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 *  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 *  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 *  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/*
 * NE10 Library : math/NE10_len.neon.c
 */

#include <assert.h>
#include <arm_neon.h>

#include "NE10.h"
#include "macros.h"


inline float32x4_t ne10_len_util(float32x4_t len2)
{
// e0 ~= 1.0 / sqrt(len2) ~= 1.0 / len
float32x4_t e0 = vrsqrteq_f32(len2);
float32x4_t e1 = vrsqrtsq_f32(e0 * len2, e0) * e0;
// len ~= 1.0 / len * (len * len) ~= len
float32x4_t len = e1 * len2;

const float32x4_t d = {0.5, 0.5, 0.5, 0.5};
len = (len + len2 / len) * d;

#ifdef NE10_MORE_PRECISE_SQRT
    uint32_t cnt = 3;
    const uint32x4_t m = {~63u, ~63u, ~63u, ~63u};
    while (cnt--)
    {
        float32x4_t l = (len + len2 / len) * d;
        uint32x4_t diff = vreinterpretq_u32_f32(l) ^ vreinterpretq_u32_f32(len);
        uint32x4_t mask_diff = diff & m;
        ne10_uint32_t result = vmaxvq_u32(mask_diff);

        len = l;
        if (!result)
        {
            break;
        }
    }
#endif

return len;
}

ne10_result_t ne10_len_vec2f_neon(ne10_float32_t *dst, ne10_vec2f_t *src, ne10_uint32_t count)
{
    ne10_uint32_t cnt = count >> 2u;
    while (cnt--)
    {
        float32x4x2_t a = vld2q_f32((ne10_float32_t *)src);
        src += 4;
        float32x4_t a02 = vmulq_f32(a.val[0], a.val[0]);
        float32x4_t len2 = vmlaq_f32(a02, a.val[1], a.val[1]);

        vst1q_f32((ne10_float32_t *)dst, ne10_len_util(len2));
        dst += 4;
    }
    count &= 3u;
    // Scalar
    while (count--)
    {
        *dst++ = sqrt((src->x * src->x) + (src->y * src->y));
        src++;
    }
    return NE10_OK;
}

ne10_result_t ne10_len_vec3f_neon(ne10_float32_t *dst, ne10_vec3f_t *src, ne10_uint32_t count)
{
    ne10_uint32_t cnt = count >> 2u;
    while (cnt--)
    {
        float32x4x3_t a = vld3q_f32((ne10_float32_t *)src);
        src += 4;
        float32x4_t len2 = vmulq_f32(a.val[0], a.val[0]);
        len2 = vmlaq_f32(len2, a.val[1], a.val[1]);
        len2 = vmlaq_f32(len2, a.val[2], a.val[2]);

        vst1q_f32((ne10_float32_t *)dst, ne10_len_util(len2));
        dst += 4;
    }
    count &= 3u;
    // Scalar
    while (count--)
    {
        *dst++ = sqrt((src->x * src->x) + (src->y * src->y) + (src->z * src->z));
        src++;
    }
    return NE10_OK;
}

ne10_result_t ne10_len_vec4f_neon(ne10_float32_t *dst, ne10_vec4f_t *src, ne10_uint32_t count)
{
    ne10_uint32_t cnt = count >> 2u;
    while (cnt--)
    {
        float32x4x4_t a = vld4q_f32((ne10_float32_t *)src);
        src += 4;
        float32x4_t len2 = vmulq_f32(a.val[0], a.val[0]);
        len2 = vmlaq_f32(len2, a.val[1], a.val[1]);
        len2 = vmlaq_f32(len2, a.val[2], a.val[2]);
        len2 = vmlaq_f32(len2, a.val[3], a.val[3]);

        vst1q_f32((ne10_float32_t *)dst, ne10_len_util(len2));
        dst += 4;
    }
    count &= 3u;
    // Scalar
    while (count--)
    {
        *dst++ = sqrt((src->x * src->x) + (src->y * src->y) + (src->z * src->z) + (src->w * src->w));
        src++;
    }
    return NE10_OK;
}
