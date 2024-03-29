
#pragma once
#ifndef UK_EXO_H
#define UK_EXO_H

#ifdef __cplusplus
extern "C" {
#endif


#include <stdint.h>
#include <stdbool.h>

// Compiler feature macros adapted from Hedley (public domain)
// https://github.com/nemequ/hedley

#if defined(__has_builtin)
#  define EXO_HAS_BUILTIN(builtin) __has_builtin(builtin)
#else
#  define EXO_HAS_BUILTIN(builtin) (0)
#endif

#if EXO_HAS_BUILTIN(__builtin_assume)
#  define EXO_ASSUME(expr) __builtin_assume(expr)
#elif EXO_HAS_BUILTIN(__builtin_unreachable)
#  define EXO_ASSUME(expr) \
      ((void)((expr) ? 1 : (__builtin_unreachable(), 1)))
#else
#  define EXO_ASSUME(expr) ((void)(expr))
#endif


struct exo_win_1f32{
    float * const data;
    const int_fast32_t strides[1];
};
struct exo_win_1f32c{
    const float * const data;
    const int_fast32_t strides[1];
};
struct exo_win_2f32{
    float * const data;
    const int_fast32_t strides[2];
};
// gemm_NEON_12x4_beta0(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 12] @DRAM,
//     B : f32[KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][4, 12] @DRAM
// )
void gemm_NEON_12x4_beta0( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_NEON_12x4_beta1(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 12] @DRAM,
//     B : f32[KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][4, 12] @DRAM
// )
void gemm_NEON_12x4_beta1( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_NEON_12x8_beta0(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 12] @DRAM,
//     B : f32[KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][8, 12] @DRAM
// )
void gemm_NEON_12x8_beta0( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_NEON_12x8_beta1(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 12] @DRAM,
//     B : f32[KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][8, 12] @DRAM
// )
void gemm_NEON_12x8_beta1( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_NEON_16x4_beta0(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 16] @DRAM,
//     B : f32[KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][4, 16] @DRAM
// )
void gemm_NEON_16x4_beta0( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_NEON_16x4_beta1(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 16] @DRAM,
//     B : f32[KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][4, 16] @DRAM
// )
void gemm_NEON_16x4_beta1( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_NEON_20x4_beta0(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 20] @DRAM,
//     B : f32[KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][4, 20] @DRAM
// )
void gemm_NEON_20x4_beta0( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_NEON_20x4_beta1(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 20] @DRAM,
//     B : f32[KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][4, 20] @DRAM
// )
void gemm_NEON_20x4_beta1( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_NEON_24x4_beta0(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 24] @DRAM,
//     B : f32[KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][4, 24] @DRAM
// )
void gemm_NEON_24x4_beta0( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_NEON_24x4_beta1(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 24] @DRAM,
//     B : f32[KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][4, 24] @DRAM
// )
void gemm_NEON_24x4_beta1( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_NEON_4x12_beta0(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 4] @DRAM,
//     B : f32[KC, 12] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][12, 4] @DRAM
// )
void gemm_NEON_4x12_beta0( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_NEON_4x12_beta1(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 4] @DRAM,
//     B : f32[KC, 12] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][12, 4] @DRAM
// )
void gemm_NEON_4x12_beta1( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_NEON_4x16_beta0(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 4] @DRAM,
//     B : f32[KC, 16] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][16, 4] @DRAM
// )
void gemm_NEON_4x16_beta0( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_NEON_4x16_beta1(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 4] @DRAM,
//     B : f32[KC, 16] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][16, 4] @DRAM
// )
void gemm_NEON_4x16_beta1( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_NEON_4x24_beta0(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 4] @DRAM,
//     B : f32[KC, 24] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][24, 4] @DRAM
// )
void gemm_NEON_4x24_beta0( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_NEON_4x24_beta1(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 4] @DRAM,
//     B : f32[KC, 24] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][24, 4] @DRAM
// )
void gemm_NEON_4x24_beta1( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_NEON_4x4_beta0(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 4] @DRAM,
//     B : f32[KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][4, 4] @DRAM
// )
void gemm_NEON_4x4_beta0( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_NEON_4x4_beta1(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 4] @DRAM,
//     B : f32[KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][4, 4] @DRAM
// )
void gemm_NEON_4x4_beta1( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_NEON_4x8_beta0(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 4] @DRAM,
//     B : f32[KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][8, 4] @DRAM
// )
void gemm_NEON_4x8_beta0( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_NEON_4x8_beta1(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 4] @DRAM,
//     B : f32[KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][8, 4] @DRAM
// )
void gemm_NEON_4x8_beta1( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_NEON_8x12_beta0(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 8] @DRAM,
//     B : f32[KC, 12] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][12, 8] @DRAM
// )
void gemm_NEON_8x12_beta0( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_NEON_8x12_beta1(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 8] @DRAM,
//     B : f32[KC, 12] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][12, 8] @DRAM
// )
void gemm_NEON_8x12_beta1( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_NEON_8x4_beta0(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 8] @DRAM,
//     B : f32[KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][4, 8] @DRAM
// )
void gemm_NEON_8x4_beta0( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_NEON_8x4_beta1(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 8] @DRAM,
//     B : f32[KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][4, 8] @DRAM
// )
void gemm_NEON_8x4_beta1( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_NEON_8x8_beta0(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 8] @DRAM,
//     B : f32[KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][8, 8] @DRAM
// )
void gemm_NEON_8x8_beta0( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );

// gemm_NEON_8x8_beta1(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 8] @DRAM,
//     B : f32[KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][8, 8] @DRAM
// )
void gemm_NEON_8x8_beta1( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C );



#ifdef __cplusplus
}
#endif
#endif  // UK_EXO_H
