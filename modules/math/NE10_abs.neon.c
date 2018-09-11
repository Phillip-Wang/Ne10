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
 * NE10 Library : math/NE10_abs.neon.c
 */

#include <assert.h>
#include <arm_neon.h>

#include "NE10.h"
#include "macros.h"

ne10_result_t ne10_abs_float_neon (ne10_float32_t * dst, ne10_float32_t * src, ne10_uint32_t count)
{
    if (count == 0) return NE10_OK;

    const ne10_uint32_t mask = ~0x7;
    ne10_uint32_t i = 0;
    for (; i < (count & mask); i+=8)
    {
        float32x4_t tmp0 = vld1q_f32(src); src += 4;
        float32x4_t tmp1 = vld1q_f32(src); src += 4;
        vst1q_f32(dst, vabsq_f32(tmp0)); dst += 4;
        vst1q_f32(dst, vabsq_f32(tmp1)); dst += 4;
    }

    for (; i < count; i++)
    {
        *dst++ = fabs(*src++);
    }

    return NE10_OK;
}

ne10_result_t ne10_abs_vec2f_neon (ne10_vec2f_t * dst, ne10_vec2f_t * src, ne10_uint32_t count)
{
    if (count == 0) return NE10_OK;

    const ne10_uint32_t mask = ~0x3;
    ne10_uint32_t i = 0;
    ne10_float32_t *src_f32 = (ne10_float32_t *)src;
    ne10_float32_t *dst_f32 = (ne10_float32_t *)dst;

    for (; i < (count & mask); i+=4)
    {
        float32x4_t tmp0 = vld1q_f32(src_f32); src_f32 += 4;
        float32x4_t tmp1 = vld1q_f32(src_f32); src_f32 += 4;
        vst1q_f32(dst_f32, vabsq_f32(tmp0)); dst_f32 += 4;
        vst1q_f32(dst_f32, vabsq_f32(tmp1)); dst_f32 += 4;
    }

    for (; i < count; i++)
    {
        *dst_f32++ = fabs(*src_f32++);
        *dst_f32++ = fabs(*src_f32++);
    }

    return NE10_OK;
}

ne10_result_t ne10_abs_vec3f_neon (ne10_vec3f_t * dst, ne10_vec3f_t * src, ne10_uint32_t count)
{
    if (count == 0) return NE10_OK;

    const ne10_uint32_t mask = ~0x3;
    ne10_uint32_t i = 0;
    ne10_float32_t *src_f32 = (ne10_float32_t *)src;
    ne10_float32_t *dst_f32 = (ne10_float32_t *)dst;

    for (; i < (count & mask); i+=4)
    {
        float32x4_t tmp0 = vld1q_f32(src_f32); src_f32 += 4;
        float32x4_t tmp1 = vld1q_f32(src_f32); src_f32 += 4;
        float32x4_t tmp2 = vld1q_f32(src_f32); src_f32 += 4;
        vst1q_f32(dst_f32, vabsq_f32(tmp0)); dst_f32 += 4;
        vst1q_f32(dst_f32, vabsq_f32(tmp1)); dst_f32 += 4;
        vst1q_f32(dst_f32, vabsq_f32(tmp2)); dst_f32 += 4;
    }

    for (; i < count; i++)
    {
        *dst_f32++ = fabs(*src_f32++);
        *dst_f32++ = fabs(*src_f32++);
        *dst_f32++ = fabs(*src_f32++);
    }

    return NE10_OK;
}

ne10_result_t ne10_abs_vec4f_neon (ne10_vec4f_t * dst, ne10_vec4f_t * src, ne10_uint32_t count)
{
    if (count == 0) return NE10_OK;

    const ne10_uint32_t mask = ~0x3;
    ne10_uint32_t i = 0;
    ne10_float32_t *src_f32 = (ne10_float32_t *)src;
    ne10_float32_t *dst_f32 = (ne10_float32_t *)dst;

    for (; i < (count & mask); i+=4)
    {
        float32x4_t tmp0 = vld1q_f32(src_f32); src_f32 += 4;
        float32x4_t tmp1 = vld1q_f32(src_f32); src_f32 += 4;
        float32x4_t tmp2 = vld1q_f32(src_f32); src_f32 += 4;
        float32x4_t tmp3 = vld1q_f32(src_f32); src_f32 += 4;
        vst1q_f32(dst_f32, vabsq_f32(tmp0)); dst_f32 += 4;
        vst1q_f32(dst_f32, vabsq_f32(tmp1)); dst_f32 += 4;
        vst1q_f32(dst_f32, vabsq_f32(tmp2)); dst_f32 += 4;
        vst1q_f32(dst_f32, vabsq_f32(tmp3)); dst_f32 += 4;
    }

    for (; i < count; i++)
    {
        float32x4_t tmp0 = vld1q_f32(src_f32); src_f32 += 4;
        vst1q_f32(dst_f32, vabsq_f32(tmp0)); dst_f32 += 4;
    }

    return NE10_OK;
}
