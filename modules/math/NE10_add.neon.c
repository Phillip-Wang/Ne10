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
 * NE10 Library : math/NE10_add.neon.c
 */

#include <assert.h>
#include <arm_neon.h>

#include "NE10.h"
#include "macros.h"

ne10_result_t ne10_add_float_neon (ne10_float32_t * dst, ne10_float32_t * src1, ne10_float32_t * src2, ne10_uint32_t count)
{
    ne10_uint32_t cnt = count >> 4u;
    while (cnt--)
    {
        float32x4_t a0 = vld1q_f32(src1); src1 += 4;
        float32x4_t b0 = vld1q_f32(src2); src2 += 4;
        float32x4_t a1 = vld1q_f32(src1); src1 += 4;
        float32x4_t b1 = vld1q_f32(src2); src2 += 4;
        float32x4_t a2 = vld1q_f32(src1); src1 += 4;
        float32x4_t b2 = vld1q_f32(src2); src2 += 4;
        float32x4_t a3 = vld1q_f32(src1); src1 += 4;
        float32x4_t b3 = vld1q_f32(src2); src2 += 4;
        vst1q_f32(dst, vaddq_f32(a0, b0)); dst += 4;
        vst1q_f32(dst, vaddq_f32(a1, b1)); dst += 4;
        vst1q_f32(dst, vaddq_f32(a2, b2)); dst += 4;
        vst1q_f32(dst, vaddq_f32(a3, b3)); dst += 4;
    }
    count &= 15u;

    cnt = count >> 2u;
    while (cnt--)
    {
        float32x4_t a0 = vld1q_f32(src1); src1 += 4;
        float32x4_t b0 = vld1q_f32(src2); src2 += 4;
        vst1q_f32(dst, vaddq_f32(a0, b0)); dst += 4;
    }
    count &= 3u;

    // Scalar
    while (count--)
    {
        *dst++ = (*src1++) + (*src2++);
    }
    return NE10_OK;
}

ne10_result_t ne10_add_vec2f_neon (ne10_vec2f_t * dst, ne10_vec2f_t * src1, ne10_vec2f_t * src2, ne10_uint32_t count)
{
    ne10_float32_t *dst_f32 = (ne10_float32_t *)dst;
    ne10_float32_t *src1_f32 = (ne10_float32_t *)src1;
    ne10_float32_t *src2_f32 = (ne10_float32_t *)src2;

    ne10_uint32_t cnt = count >> 3u;
    while (cnt--)
    {
        float32x4_t a0 = vld1q_f32(src1_f32); src1_f32 += 4;
        float32x4_t a1 = vld1q_f32(src1_f32); src1_f32 += 4;
        float32x4_t a2 = vld1q_f32(src1_f32); src1_f32 += 4;
        float32x4_t a3 = vld1q_f32(src1_f32); src1_f32 += 4;
        float32x4_t b0 = vld1q_f32(src2_f32); src2_f32 += 4;
        float32x4_t b1 = vld1q_f32(src2_f32); src2_f32 += 4;
        float32x4_t b2 = vld1q_f32(src2_f32); src2_f32 += 4;
        float32x4_t b3 = vld1q_f32(src2_f32); src2_f32 += 4;
        vst1q_f32(dst_f32, vaddq_f32(a0, b0)); dst_f32 += 4;
        vst1q_f32(dst_f32, vaddq_f32(a1, b1)); dst_f32 += 4;
        vst1q_f32(dst_f32, vaddq_f32(a2, b2)); dst_f32 += 4;
        vst1q_f32(dst_f32, vaddq_f32(a3, b3)); dst_f32 += 4;
    }
    count &= 7u;

    cnt = count >> 1u;
    while (cnt--)
    {
        float32x4_t a0 = vld1q_f32(src1_f32); src1_f32 += 4;
        float32x4_t b0 = vld1q_f32(src2_f32); src2_f32 += 4;
        vst1q_f32(dst_f32, vaddq_f32(a0, b0)); dst_f32 += 4;
    }

    if (count & 0x1u)
    {
        *dst_f32++ = (*src1_f32++) + (*src2_f32++);
        *dst_f32++ = (*src1_f32++) + (*src2_f32++);
    }
    return NE10_OK;
}

ne10_result_t ne10_add_vec3f_neon (ne10_vec3f_t * dst, ne10_vec3f_t * src1, ne10_vec3f_t * src2, ne10_uint32_t count)
{
    ne10_uint32_t cnt = count >> 2u;
    while (cnt--)
    {
        ne10_float32_t *dst_f32 = (ne10_float32_t *)dst;
        ne10_float32_t *src1_f32 = (ne10_float32_t *)src1;
        ne10_float32_t *src2_f32 = (ne10_float32_t *)src2;

        float32x4_t a0 = vld1q_f32(src1_f32); src1_f32 += 4;
        float32x4_t a1 = vld1q_f32(src1_f32); src1_f32 += 4;
        float32x4_t a2 = vld1q_f32(src1_f32); src1_f32 += 4;

        float32x4_t b0 = vld1q_f32(src2_f32); src2_f32 += 4;
        float32x4_t b1 = vld1q_f32(src2_f32); src2_f32 += 4;
        float32x4_t b2 = vld1q_f32(src2_f32); src2_f32 += 4;

        vst1q_f32(dst_f32, vaddq_f32(a0, b0)); dst_f32 += 4;
        vst1q_f32(dst_f32, vaddq_f32(a1, b1)); dst_f32 += 4;
        vst1q_f32(dst_f32, vaddq_f32(a2, b2)); dst_f32 += 4;

        dst = (ne10_vec3f_t *)dst_f32;
        src1 = (ne10_vec3f_t *)src1_f32;
        src2 = (ne10_vec3f_t *)src2_f32;
    }

    // Scalar
    count &= 3u;
    while (count--)
    {
        dst->x = src1->x + src2->x;
        dst->y = src1->y + src2->y;
        dst->z = src1->z + src2->z;
        dst++; src1++; src2++;
    }
    return NE10_OK;
}

ne10_result_t ne10_add_vec4f_neon (ne10_vec4f_t * dst, ne10_vec4f_t * src1, ne10_vec4f_t * src2, ne10_uint32_t count)
{
    ne10_uint32_t cnt = count >> 2u;
    while (cnt--)
    {
        float32x4_t a0 = vld1q_f32((ne10_float32_t *)src1++);
        float32x4_t b0 = vld1q_f32((ne10_float32_t *)src2++);
        float32x4_t a1 = vld1q_f32((ne10_float32_t *)src1++);
        float32x4_t b1 = vld1q_f32((ne10_float32_t *)src2++);
        float32x4_t a2 = vld1q_f32((ne10_float32_t *)src1++);
        float32x4_t b2 = vld1q_f32((ne10_float32_t *)src2++);
        float32x4_t a3 = vld1q_f32((ne10_float32_t *)src1++);
        float32x4_t b3 = vld1q_f32((ne10_float32_t *)src2++);

        vst1q_f32((ne10_float32_t *)dst++, vaddq_f32(a0, b0));
        vst1q_f32((ne10_float32_t *)dst++, vaddq_f32(a1, b1));
        vst1q_f32((ne10_float32_t *)dst++, vaddq_f32(a2, b2));
        vst1q_f32((ne10_float32_t *)dst++, vaddq_f32(a3, b3));
    }

    // Scalar
    count &= 3u;
    while (count--)
    {
        float32x4_t a0 = vld1q_f32((ne10_float32_t *)src1++);
        float32x4_t b0 = vld1q_f32((ne10_float32_t *)src2++);
        vst1q_f32((ne10_float32_t *)dst++, vaddq_f32(a0, b0));
    }
    return NE10_OK;
}
