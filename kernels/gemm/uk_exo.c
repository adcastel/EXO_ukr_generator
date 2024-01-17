#include "uk_exo.h"



#include <stdio.h>
#include <stdlib.h>

#include <arm_neon.h>


// gemm_NEON_12x4_beta0(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 12] @DRAM,
//     B : f32[KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][4, 12] @DRAM
// )
void gemm_NEON_12x4_beta0( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C ) {
float32x4_t C_reg_0_0;
float32x4_t C_reg_0_1;
float32x4_t C_reg_0_2;
float32x4_t C_reg_1_0;
float32x4_t C_reg_1_1;
float32x4_t C_reg_1_2;
float32x4_t C_reg_2_0;
float32x4_t C_reg_2_1;
float32x4_t C_reg_2_2;
float32x4_t C_reg_3_0;
float32x4_t C_reg_3_1;
float32x4_t C_reg_3_2;
C_reg_0_0 = vmovq_n_f32(0.0f);
C_reg_0_1 = vmovq_n_f32(0.0f);
C_reg_0_2 = vmovq_n_f32(0.0f);
C_reg_1_0 = vmovq_n_f32(0.0f);
C_reg_1_1 = vmovq_n_f32(0.0f);
C_reg_1_2 = vmovq_n_f32(0.0f);
C_reg_2_0 = vmovq_n_f32(0.0f);
C_reg_2_1 = vmovq_n_f32(0.0f);
C_reg_2_2 = vmovq_n_f32(0.0f);
C_reg_3_0 = vmovq_n_f32(0.0f);
C_reg_3_1 = vmovq_n_f32(0.0f);
C_reg_3_2 = vmovq_n_f32(0.0f);
float32x4_t A_reg_0;
float32x4_t A_reg_1;
float32x4_t A_reg_2;
float32x4_t B_reg_0;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = vld1q_f32(&A[(k) * (12)]);
  A_reg_1 = vld1q_f32(&A[(k) * (12) + 4]);
  A_reg_2 = vld1q_f32(&A[(k) * (12) + 8]);
  B_reg_0 = vld1q_f32(&B[(k) * 4]);
  C_reg_0_0 = vfmaq_laneq_f32(C_reg_0_0, A_reg_0, B_reg_0, (0));
  C_reg_1_0 = vfmaq_laneq_f32(C_reg_1_0, A_reg_0, B_reg_0, (1));
  C_reg_2_0 = vfmaq_laneq_f32(C_reg_2_0, A_reg_0, B_reg_0, (2));
  C_reg_3_0 = vfmaq_laneq_f32(C_reg_3_0, A_reg_0, B_reg_0, (3));
  C_reg_0_1 = vfmaq_laneq_f32(C_reg_0_1, A_reg_1, B_reg_0, (0));
  C_reg_1_1 = vfmaq_laneq_f32(C_reg_1_1, A_reg_1, B_reg_0, (1));
  C_reg_2_1 = vfmaq_laneq_f32(C_reg_2_1, A_reg_1, B_reg_0, (2));
  C_reg_3_1 = vfmaq_laneq_f32(C_reg_3_1, A_reg_1, B_reg_0, (3));
  C_reg_0_2 = vfmaq_laneq_f32(C_reg_0_2, A_reg_2, B_reg_0, (0));
  C_reg_1_2 = vfmaq_laneq_f32(C_reg_1_2, A_reg_2, B_reg_0, (1));
  C_reg_2_2 = vfmaq_laneq_f32(C_reg_2_2, A_reg_2, B_reg_0, (2));
  C_reg_3_2 = vfmaq_laneq_f32(C_reg_3_2, A_reg_2, B_reg_0, (3));
}
vst1q_f32(&C.data[0], C_reg_0_0);
vst1q_f32(&C.data[(4) * (C.strides[1])], C_reg_0_1);
vst1q_f32(&C.data[(8) * (C.strides[1])], C_reg_0_2);
vst1q_f32(&C.data[C.strides[0]], C_reg_1_0);
vst1q_f32(&C.data[C.strides[0] + (4) * (C.strides[1])], C_reg_1_1);
vst1q_f32(&C.data[C.strides[0] + (8) * (C.strides[1])], C_reg_1_2);
vst1q_f32(&C.data[(2) * (C.strides[0])], C_reg_2_0);
vst1q_f32(&C.data[(2) * (C.strides[0]) + (4) * (C.strides[1])], C_reg_2_1);
vst1q_f32(&C.data[(2) * (C.strides[0]) + (8) * (C.strides[1])], C_reg_2_2);
vst1q_f32(&C.data[(3) * (C.strides[0])], C_reg_3_0);
vst1q_f32(&C.data[(3) * (C.strides[0]) + (4) * (C.strides[1])], C_reg_3_1);
vst1q_f32(&C.data[(3) * (C.strides[0]) + (8) * (C.strides[1])], C_reg_3_2);
}

// gemm_NEON_12x4_beta1(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 12] @DRAM,
//     B : f32[KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][4, 12] @DRAM
// )
void gemm_NEON_12x4_beta1( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C ) {
float *Ba = (float*) malloc(KC * 4 * sizeof(*Ba));
free(Ba);
float *Cb = (float*) malloc(4 * 12 * sizeof(*Cb));
free(Cb);
float32x4_t C_reg_0_0;
float32x4_t C_reg_0_1;
float32x4_t C_reg_0_2;
float32x4_t C_reg_1_0;
float32x4_t C_reg_1_1;
float32x4_t C_reg_1_2;
float32x4_t C_reg_2_0;
float32x4_t C_reg_2_1;
float32x4_t C_reg_2_2;
float32x4_t C_reg_3_0;
float32x4_t C_reg_3_1;
float32x4_t C_reg_3_2;
C_reg_0_0 = vld1q_f32(&C.data[0]);
C_reg_0_1 = vld1q_f32(&C.data[(4) * (C.strides[1])]);
C_reg_0_2 = vld1q_f32(&C.data[(8) * (C.strides[1])]);
C_reg_1_0 = vld1q_f32(&C.data[C.strides[0]]);
C_reg_1_1 = vld1q_f32(&C.data[C.strides[0] + (4) * (C.strides[1])]);
C_reg_1_2 = vld1q_f32(&C.data[C.strides[0] + (8) * (C.strides[1])]);
C_reg_2_0 = vld1q_f32(&C.data[(2) * (C.strides[0])]);
C_reg_2_1 = vld1q_f32(&C.data[(2) * (C.strides[0]) + (4) * (C.strides[1])]);
C_reg_2_2 = vld1q_f32(&C.data[(2) * (C.strides[0]) + (8) * (C.strides[1])]);
C_reg_3_0 = vld1q_f32(&C.data[(3) * (C.strides[0])]);
C_reg_3_1 = vld1q_f32(&C.data[(3) * (C.strides[0]) + (4) * (C.strides[1])]);
C_reg_3_2 = vld1q_f32(&C.data[(3) * (C.strides[0]) + (8) * (C.strides[1])]);
float32x4_t A_reg_0;
float32x4_t A_reg_1;
float32x4_t A_reg_2;
float32x4_t B_reg_0;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = vld1q_f32(&A[(k) * (12)]);
  A_reg_1 = vld1q_f32(&A[(k) * (12) + 4]);
  A_reg_2 = vld1q_f32(&A[(k) * (12) + 8]);
  B_reg_0 = vld1q_f32(&B[(k) * 4]);
  C_reg_0_0 = vfmaq_laneq_f32(C_reg_0_0, A_reg_0, B_reg_0, (0));
  C_reg_1_0 = vfmaq_laneq_f32(C_reg_1_0, A_reg_0, B_reg_0, (1));
  C_reg_2_0 = vfmaq_laneq_f32(C_reg_2_0, A_reg_0, B_reg_0, (2));
  C_reg_3_0 = vfmaq_laneq_f32(C_reg_3_0, A_reg_0, B_reg_0, (3));
  C_reg_0_1 = vfmaq_laneq_f32(C_reg_0_1, A_reg_1, B_reg_0, (0));
  C_reg_1_1 = vfmaq_laneq_f32(C_reg_1_1, A_reg_1, B_reg_0, (1));
  C_reg_2_1 = vfmaq_laneq_f32(C_reg_2_1, A_reg_1, B_reg_0, (2));
  C_reg_3_1 = vfmaq_laneq_f32(C_reg_3_1, A_reg_1, B_reg_0, (3));
  C_reg_0_2 = vfmaq_laneq_f32(C_reg_0_2, A_reg_2, B_reg_0, (0));
  C_reg_1_2 = vfmaq_laneq_f32(C_reg_1_2, A_reg_2, B_reg_0, (1));
  C_reg_2_2 = vfmaq_laneq_f32(C_reg_2_2, A_reg_2, B_reg_0, (2));
  C_reg_3_2 = vfmaq_laneq_f32(C_reg_3_2, A_reg_2, B_reg_0, (3));
}
vst1q_f32(&C.data[0], C_reg_0_0);
vst1q_f32(&C.data[(4) * (C.strides[1])], C_reg_0_1);
vst1q_f32(&C.data[(8) * (C.strides[1])], C_reg_0_2);
vst1q_f32(&C.data[C.strides[0]], C_reg_1_0);
vst1q_f32(&C.data[C.strides[0] + (4) * (C.strides[1])], C_reg_1_1);
vst1q_f32(&C.data[C.strides[0] + (8) * (C.strides[1])], C_reg_1_2);
vst1q_f32(&C.data[(2) * (C.strides[0])], C_reg_2_0);
vst1q_f32(&C.data[(2) * (C.strides[0]) + (4) * (C.strides[1])], C_reg_2_1);
vst1q_f32(&C.data[(2) * (C.strides[0]) + (8) * (C.strides[1])], C_reg_2_2);
vst1q_f32(&C.data[(3) * (C.strides[0])], C_reg_3_0);
vst1q_f32(&C.data[(3) * (C.strides[0]) + (4) * (C.strides[1])], C_reg_3_1);
vst1q_f32(&C.data[(3) * (C.strides[0]) + (8) * (C.strides[1])], C_reg_3_2);
}

// gemm_NEON_12x8_beta0(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 12] @DRAM,
//     B : f32[KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][8, 12] @DRAM
// )
void gemm_NEON_12x8_beta0( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C ) {
float32x4_t C_reg_0_0;
float32x4_t C_reg_0_1;
float32x4_t C_reg_0_2;
float32x4_t C_reg_1_0;
float32x4_t C_reg_1_1;
float32x4_t C_reg_1_2;
float32x4_t C_reg_2_0;
float32x4_t C_reg_2_1;
float32x4_t C_reg_2_2;
float32x4_t C_reg_3_0;
float32x4_t C_reg_3_1;
float32x4_t C_reg_3_2;
float32x4_t C_reg_4_0;
float32x4_t C_reg_4_1;
float32x4_t C_reg_4_2;
float32x4_t C_reg_5_0;
float32x4_t C_reg_5_1;
float32x4_t C_reg_5_2;
float32x4_t C_reg_6_0;
float32x4_t C_reg_6_1;
float32x4_t C_reg_6_2;
float32x4_t C_reg_7_0;
float32x4_t C_reg_7_1;
float32x4_t C_reg_7_2;
C_reg_0_0 = vmovq_n_f32(0.0f);
C_reg_0_1 = vmovq_n_f32(0.0f);
C_reg_0_2 = vmovq_n_f32(0.0f);
C_reg_1_0 = vmovq_n_f32(0.0f);
C_reg_1_1 = vmovq_n_f32(0.0f);
C_reg_1_2 = vmovq_n_f32(0.0f);
C_reg_2_0 = vmovq_n_f32(0.0f);
C_reg_2_1 = vmovq_n_f32(0.0f);
C_reg_2_2 = vmovq_n_f32(0.0f);
C_reg_3_0 = vmovq_n_f32(0.0f);
C_reg_3_1 = vmovq_n_f32(0.0f);
C_reg_3_2 = vmovq_n_f32(0.0f);
C_reg_4_0 = vmovq_n_f32(0.0f);
C_reg_4_1 = vmovq_n_f32(0.0f);
C_reg_4_2 = vmovq_n_f32(0.0f);
C_reg_5_0 = vmovq_n_f32(0.0f);
C_reg_5_1 = vmovq_n_f32(0.0f);
C_reg_5_2 = vmovq_n_f32(0.0f);
C_reg_6_0 = vmovq_n_f32(0.0f);
C_reg_6_1 = vmovq_n_f32(0.0f);
C_reg_6_2 = vmovq_n_f32(0.0f);
C_reg_7_0 = vmovq_n_f32(0.0f);
C_reg_7_1 = vmovq_n_f32(0.0f);
C_reg_7_2 = vmovq_n_f32(0.0f);
float32x4_t A_reg_0;
float32x4_t A_reg_1;
float32x4_t A_reg_2;
float32x4_t B_reg_0;
float32x4_t B_reg_1;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = vld1q_f32(&A[(k) * (12)]);
  A_reg_1 = vld1q_f32(&A[(k) * (12) + 4]);
  A_reg_2 = vld1q_f32(&A[(k) * (12) + 8]);
  B_reg_0 = vld1q_f32(&B[(k) * 8]);
  B_reg_1 = vld1q_f32(&B[(k) * 8 + 4]);
  C_reg_0_0 = vfmaq_laneq_f32(C_reg_0_0, A_reg_0, B_reg_0, (0));
  C_reg_1_0 = vfmaq_laneq_f32(C_reg_1_0, A_reg_0, B_reg_0, (1));
  C_reg_2_0 = vfmaq_laneq_f32(C_reg_2_0, A_reg_0, B_reg_0, (2));
  C_reg_3_0 = vfmaq_laneq_f32(C_reg_3_0, A_reg_0, B_reg_0, (3));
  C_reg_0_1 = vfmaq_laneq_f32(C_reg_0_1, A_reg_1, B_reg_0, (0));
  C_reg_1_1 = vfmaq_laneq_f32(C_reg_1_1, A_reg_1, B_reg_0, (1));
  C_reg_2_1 = vfmaq_laneq_f32(C_reg_2_1, A_reg_1, B_reg_0, (2));
  C_reg_3_1 = vfmaq_laneq_f32(C_reg_3_1, A_reg_1, B_reg_0, (3));
  C_reg_0_2 = vfmaq_laneq_f32(C_reg_0_2, A_reg_2, B_reg_0, (0));
  C_reg_1_2 = vfmaq_laneq_f32(C_reg_1_2, A_reg_2, B_reg_0, (1));
  C_reg_2_2 = vfmaq_laneq_f32(C_reg_2_2, A_reg_2, B_reg_0, (2));
  C_reg_3_2 = vfmaq_laneq_f32(C_reg_3_2, A_reg_2, B_reg_0, (3));
  C_reg_4_0 = vfmaq_laneq_f32(C_reg_4_0, A_reg_0, B_reg_1, (0));
  C_reg_5_0 = vfmaq_laneq_f32(C_reg_5_0, A_reg_0, B_reg_1, (1));
  C_reg_6_0 = vfmaq_laneq_f32(C_reg_6_0, A_reg_0, B_reg_1, (2));
  C_reg_7_0 = vfmaq_laneq_f32(C_reg_7_0, A_reg_0, B_reg_1, (3));
  C_reg_4_1 = vfmaq_laneq_f32(C_reg_4_1, A_reg_1, B_reg_1, (0));
  C_reg_5_1 = vfmaq_laneq_f32(C_reg_5_1, A_reg_1, B_reg_1, (1));
  C_reg_6_1 = vfmaq_laneq_f32(C_reg_6_1, A_reg_1, B_reg_1, (2));
  C_reg_7_1 = vfmaq_laneq_f32(C_reg_7_1, A_reg_1, B_reg_1, (3));
  C_reg_4_2 = vfmaq_laneq_f32(C_reg_4_2, A_reg_2, B_reg_1, (0));
  C_reg_5_2 = vfmaq_laneq_f32(C_reg_5_2, A_reg_2, B_reg_1, (1));
  C_reg_6_2 = vfmaq_laneq_f32(C_reg_6_2, A_reg_2, B_reg_1, (2));
  C_reg_7_2 = vfmaq_laneq_f32(C_reg_7_2, A_reg_2, B_reg_1, (3));
}
vst1q_f32(&C.data[0], C_reg_0_0);
vst1q_f32(&C.data[(4) * (C.strides[1])], C_reg_0_1);
vst1q_f32(&C.data[(8) * (C.strides[1])], C_reg_0_2);
vst1q_f32(&C.data[C.strides[0]], C_reg_1_0);
vst1q_f32(&C.data[C.strides[0] + (4) * (C.strides[1])], C_reg_1_1);
vst1q_f32(&C.data[C.strides[0] + (8) * (C.strides[1])], C_reg_1_2);
vst1q_f32(&C.data[(2) * (C.strides[0])], C_reg_2_0);
vst1q_f32(&C.data[(2) * (C.strides[0]) + (4) * (C.strides[1])], C_reg_2_1);
vst1q_f32(&C.data[(2) * (C.strides[0]) + (8) * (C.strides[1])], C_reg_2_2);
vst1q_f32(&C.data[(3) * (C.strides[0])], C_reg_3_0);
vst1q_f32(&C.data[(3) * (C.strides[0]) + (4) * (C.strides[1])], C_reg_3_1);
vst1q_f32(&C.data[(3) * (C.strides[0]) + (8) * (C.strides[1])], C_reg_3_2);
vst1q_f32(&C.data[(4) * (C.strides[0])], C_reg_4_0);
vst1q_f32(&C.data[(4) * (C.strides[0]) + (4) * (C.strides[1])], C_reg_4_1);
vst1q_f32(&C.data[(4) * (C.strides[0]) + (8) * (C.strides[1])], C_reg_4_2);
vst1q_f32(&C.data[(5) * (C.strides[0])], C_reg_5_0);
vst1q_f32(&C.data[(5) * (C.strides[0]) + (4) * (C.strides[1])], C_reg_5_1);
vst1q_f32(&C.data[(5) * (C.strides[0]) + (8) * (C.strides[1])], C_reg_5_2);
vst1q_f32(&C.data[(6) * (C.strides[0])], C_reg_6_0);
vst1q_f32(&C.data[(6) * (C.strides[0]) + (4) * (C.strides[1])], C_reg_6_1);
vst1q_f32(&C.data[(6) * (C.strides[0]) + (8) * (C.strides[1])], C_reg_6_2);
vst1q_f32(&C.data[(7) * (C.strides[0])], C_reg_7_0);
vst1q_f32(&C.data[(7) * (C.strides[0]) + (4) * (C.strides[1])], C_reg_7_1);
vst1q_f32(&C.data[(7) * (C.strides[0]) + (8) * (C.strides[1])], C_reg_7_2);
}

// gemm_NEON_12x8_beta1(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 12] @DRAM,
//     B : f32[KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][8, 12] @DRAM
// )
void gemm_NEON_12x8_beta1( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C ) {
float *Ba = (float*) malloc(KC * 8 * sizeof(*Ba));
free(Ba);
float *Cb = (float*) malloc(8 * 12 * sizeof(*Cb));
free(Cb);
float32x4_t C_reg_0_0;
float32x4_t C_reg_0_1;
float32x4_t C_reg_0_2;
float32x4_t C_reg_1_0;
float32x4_t C_reg_1_1;
float32x4_t C_reg_1_2;
float32x4_t C_reg_2_0;
float32x4_t C_reg_2_1;
float32x4_t C_reg_2_2;
float32x4_t C_reg_3_0;
float32x4_t C_reg_3_1;
float32x4_t C_reg_3_2;
float32x4_t C_reg_4_0;
float32x4_t C_reg_4_1;
float32x4_t C_reg_4_2;
float32x4_t C_reg_5_0;
float32x4_t C_reg_5_1;
float32x4_t C_reg_5_2;
float32x4_t C_reg_6_0;
float32x4_t C_reg_6_1;
float32x4_t C_reg_6_2;
float32x4_t C_reg_7_0;
float32x4_t C_reg_7_1;
float32x4_t C_reg_7_2;
C_reg_0_0 = vld1q_f32(&C.data[0]);
C_reg_0_1 = vld1q_f32(&C.data[(4) * (C.strides[1])]);
C_reg_0_2 = vld1q_f32(&C.data[(8) * (C.strides[1])]);
C_reg_1_0 = vld1q_f32(&C.data[C.strides[0]]);
C_reg_1_1 = vld1q_f32(&C.data[C.strides[0] + (4) * (C.strides[1])]);
C_reg_1_2 = vld1q_f32(&C.data[C.strides[0] + (8) * (C.strides[1])]);
C_reg_2_0 = vld1q_f32(&C.data[(2) * (C.strides[0])]);
C_reg_2_1 = vld1q_f32(&C.data[(2) * (C.strides[0]) + (4) * (C.strides[1])]);
C_reg_2_2 = vld1q_f32(&C.data[(2) * (C.strides[0]) + (8) * (C.strides[1])]);
C_reg_3_0 = vld1q_f32(&C.data[(3) * (C.strides[0])]);
C_reg_3_1 = vld1q_f32(&C.data[(3) * (C.strides[0]) + (4) * (C.strides[1])]);
C_reg_3_2 = vld1q_f32(&C.data[(3) * (C.strides[0]) + (8) * (C.strides[1])]);
C_reg_4_0 = vld1q_f32(&C.data[(4) * (C.strides[0])]);
C_reg_4_1 = vld1q_f32(&C.data[(4) * (C.strides[0]) + (4) * (C.strides[1])]);
C_reg_4_2 = vld1q_f32(&C.data[(4) * (C.strides[0]) + (8) * (C.strides[1])]);
C_reg_5_0 = vld1q_f32(&C.data[(5) * (C.strides[0])]);
C_reg_5_1 = vld1q_f32(&C.data[(5) * (C.strides[0]) + (4) * (C.strides[1])]);
C_reg_5_2 = vld1q_f32(&C.data[(5) * (C.strides[0]) + (8) * (C.strides[1])]);
C_reg_6_0 = vld1q_f32(&C.data[(6) * (C.strides[0])]);
C_reg_6_1 = vld1q_f32(&C.data[(6) * (C.strides[0]) + (4) * (C.strides[1])]);
C_reg_6_2 = vld1q_f32(&C.data[(6) * (C.strides[0]) + (8) * (C.strides[1])]);
C_reg_7_0 = vld1q_f32(&C.data[(7) * (C.strides[0])]);
C_reg_7_1 = vld1q_f32(&C.data[(7) * (C.strides[0]) + (4) * (C.strides[1])]);
C_reg_7_2 = vld1q_f32(&C.data[(7) * (C.strides[0]) + (8) * (C.strides[1])]);
float32x4_t A_reg_0;
float32x4_t A_reg_1;
float32x4_t A_reg_2;
float32x4_t B_reg_0;
float32x4_t B_reg_1;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = vld1q_f32(&A[(k) * (12)]);
  A_reg_1 = vld1q_f32(&A[(k) * (12) + 4]);
  A_reg_2 = vld1q_f32(&A[(k) * (12) + 8]);
  B_reg_0 = vld1q_f32(&B[(k) * 8]);
  B_reg_1 = vld1q_f32(&B[(k) * 8 + 4]);
  C_reg_0_0 = vfmaq_laneq_f32(C_reg_0_0, A_reg_0, B_reg_0, (0));
  C_reg_1_0 = vfmaq_laneq_f32(C_reg_1_0, A_reg_0, B_reg_0, (1));
  C_reg_2_0 = vfmaq_laneq_f32(C_reg_2_0, A_reg_0, B_reg_0, (2));
  C_reg_3_0 = vfmaq_laneq_f32(C_reg_3_0, A_reg_0, B_reg_0, (3));
  C_reg_0_1 = vfmaq_laneq_f32(C_reg_0_1, A_reg_1, B_reg_0, (0));
  C_reg_1_1 = vfmaq_laneq_f32(C_reg_1_1, A_reg_1, B_reg_0, (1));
  C_reg_2_1 = vfmaq_laneq_f32(C_reg_2_1, A_reg_1, B_reg_0, (2));
  C_reg_3_1 = vfmaq_laneq_f32(C_reg_3_1, A_reg_1, B_reg_0, (3));
  C_reg_0_2 = vfmaq_laneq_f32(C_reg_0_2, A_reg_2, B_reg_0, (0));
  C_reg_1_2 = vfmaq_laneq_f32(C_reg_1_2, A_reg_2, B_reg_0, (1));
  C_reg_2_2 = vfmaq_laneq_f32(C_reg_2_2, A_reg_2, B_reg_0, (2));
  C_reg_3_2 = vfmaq_laneq_f32(C_reg_3_2, A_reg_2, B_reg_0, (3));
  C_reg_4_0 = vfmaq_laneq_f32(C_reg_4_0, A_reg_0, B_reg_1, (0));
  C_reg_5_0 = vfmaq_laneq_f32(C_reg_5_0, A_reg_0, B_reg_1, (1));
  C_reg_6_0 = vfmaq_laneq_f32(C_reg_6_0, A_reg_0, B_reg_1, (2));
  C_reg_7_0 = vfmaq_laneq_f32(C_reg_7_0, A_reg_0, B_reg_1, (3));
  C_reg_4_1 = vfmaq_laneq_f32(C_reg_4_1, A_reg_1, B_reg_1, (0));
  C_reg_5_1 = vfmaq_laneq_f32(C_reg_5_1, A_reg_1, B_reg_1, (1));
  C_reg_6_1 = vfmaq_laneq_f32(C_reg_6_1, A_reg_1, B_reg_1, (2));
  C_reg_7_1 = vfmaq_laneq_f32(C_reg_7_1, A_reg_1, B_reg_1, (3));
  C_reg_4_2 = vfmaq_laneq_f32(C_reg_4_2, A_reg_2, B_reg_1, (0));
  C_reg_5_2 = vfmaq_laneq_f32(C_reg_5_2, A_reg_2, B_reg_1, (1));
  C_reg_6_2 = vfmaq_laneq_f32(C_reg_6_2, A_reg_2, B_reg_1, (2));
  C_reg_7_2 = vfmaq_laneq_f32(C_reg_7_2, A_reg_2, B_reg_1, (3));
}
vst1q_f32(&C.data[0], C_reg_0_0);
vst1q_f32(&C.data[(4) * (C.strides[1])], C_reg_0_1);
vst1q_f32(&C.data[(8) * (C.strides[1])], C_reg_0_2);
vst1q_f32(&C.data[C.strides[0]], C_reg_1_0);
vst1q_f32(&C.data[C.strides[0] + (4) * (C.strides[1])], C_reg_1_1);
vst1q_f32(&C.data[C.strides[0] + (8) * (C.strides[1])], C_reg_1_2);
vst1q_f32(&C.data[(2) * (C.strides[0])], C_reg_2_0);
vst1q_f32(&C.data[(2) * (C.strides[0]) + (4) * (C.strides[1])], C_reg_2_1);
vst1q_f32(&C.data[(2) * (C.strides[0]) + (8) * (C.strides[1])], C_reg_2_2);
vst1q_f32(&C.data[(3) * (C.strides[0])], C_reg_3_0);
vst1q_f32(&C.data[(3) * (C.strides[0]) + (4) * (C.strides[1])], C_reg_3_1);
vst1q_f32(&C.data[(3) * (C.strides[0]) + (8) * (C.strides[1])], C_reg_3_2);
vst1q_f32(&C.data[(4) * (C.strides[0])], C_reg_4_0);
vst1q_f32(&C.data[(4) * (C.strides[0]) + (4) * (C.strides[1])], C_reg_4_1);
vst1q_f32(&C.data[(4) * (C.strides[0]) + (8) * (C.strides[1])], C_reg_4_2);
vst1q_f32(&C.data[(5) * (C.strides[0])], C_reg_5_0);
vst1q_f32(&C.data[(5) * (C.strides[0]) + (4) * (C.strides[1])], C_reg_5_1);
vst1q_f32(&C.data[(5) * (C.strides[0]) + (8) * (C.strides[1])], C_reg_5_2);
vst1q_f32(&C.data[(6) * (C.strides[0])], C_reg_6_0);
vst1q_f32(&C.data[(6) * (C.strides[0]) + (4) * (C.strides[1])], C_reg_6_1);
vst1q_f32(&C.data[(6) * (C.strides[0]) + (8) * (C.strides[1])], C_reg_6_2);
vst1q_f32(&C.data[(7) * (C.strides[0])], C_reg_7_0);
vst1q_f32(&C.data[(7) * (C.strides[0]) + (4) * (C.strides[1])], C_reg_7_1);
vst1q_f32(&C.data[(7) * (C.strides[0]) + (8) * (C.strides[1])], C_reg_7_2);
}

// gemm_NEON_16x4_beta0(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 16] @DRAM,
//     B : f32[KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][4, 16] @DRAM
// )
void gemm_NEON_16x4_beta0( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C ) {
float32x4_t C_reg_0_0;
float32x4_t C_reg_0_1;
float32x4_t C_reg_0_2;
float32x4_t C_reg_0_3;
float32x4_t C_reg_1_0;
float32x4_t C_reg_1_1;
float32x4_t C_reg_1_2;
float32x4_t C_reg_1_3;
float32x4_t C_reg_2_0;
float32x4_t C_reg_2_1;
float32x4_t C_reg_2_2;
float32x4_t C_reg_2_3;
float32x4_t C_reg_3_0;
float32x4_t C_reg_3_1;
float32x4_t C_reg_3_2;
float32x4_t C_reg_3_3;
C_reg_0_0 = vmovq_n_f32(0.0f);
C_reg_0_1 = vmovq_n_f32(0.0f);
C_reg_0_2 = vmovq_n_f32(0.0f);
C_reg_0_3 = vmovq_n_f32(0.0f);
C_reg_1_0 = vmovq_n_f32(0.0f);
C_reg_1_1 = vmovq_n_f32(0.0f);
C_reg_1_2 = vmovq_n_f32(0.0f);
C_reg_1_3 = vmovq_n_f32(0.0f);
C_reg_2_0 = vmovq_n_f32(0.0f);
C_reg_2_1 = vmovq_n_f32(0.0f);
C_reg_2_2 = vmovq_n_f32(0.0f);
C_reg_2_3 = vmovq_n_f32(0.0f);
C_reg_3_0 = vmovq_n_f32(0.0f);
C_reg_3_1 = vmovq_n_f32(0.0f);
C_reg_3_2 = vmovq_n_f32(0.0f);
C_reg_3_3 = vmovq_n_f32(0.0f);
float32x4_t A_reg_0;
float32x4_t A_reg_1;
float32x4_t A_reg_2;
float32x4_t A_reg_3;
float32x4_t B_reg_0;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = vld1q_f32(&A[(k) * (16)]);
  A_reg_1 = vld1q_f32(&A[(k) * (16) + 4]);
  A_reg_2 = vld1q_f32(&A[(k) * (16) + 8]);
  A_reg_3 = vld1q_f32(&A[(k) * (16) + 12]);
  B_reg_0 = vld1q_f32(&B[(k) * 4]);
  C_reg_0_0 = vfmaq_laneq_f32(C_reg_0_0, A_reg_0, B_reg_0, (0));
  C_reg_1_0 = vfmaq_laneq_f32(C_reg_1_0, A_reg_0, B_reg_0, (1));
  C_reg_2_0 = vfmaq_laneq_f32(C_reg_2_0, A_reg_0, B_reg_0, (2));
  C_reg_3_0 = vfmaq_laneq_f32(C_reg_3_0, A_reg_0, B_reg_0, (3));
  C_reg_0_1 = vfmaq_laneq_f32(C_reg_0_1, A_reg_1, B_reg_0, (0));
  C_reg_1_1 = vfmaq_laneq_f32(C_reg_1_1, A_reg_1, B_reg_0, (1));
  C_reg_2_1 = vfmaq_laneq_f32(C_reg_2_1, A_reg_1, B_reg_0, (2));
  C_reg_3_1 = vfmaq_laneq_f32(C_reg_3_1, A_reg_1, B_reg_0, (3));
  C_reg_0_2 = vfmaq_laneq_f32(C_reg_0_2, A_reg_2, B_reg_0, (0));
  C_reg_1_2 = vfmaq_laneq_f32(C_reg_1_2, A_reg_2, B_reg_0, (1));
  C_reg_2_2 = vfmaq_laneq_f32(C_reg_2_2, A_reg_2, B_reg_0, (2));
  C_reg_3_2 = vfmaq_laneq_f32(C_reg_3_2, A_reg_2, B_reg_0, (3));
  C_reg_0_3 = vfmaq_laneq_f32(C_reg_0_3, A_reg_3, B_reg_0, (0));
  C_reg_1_3 = vfmaq_laneq_f32(C_reg_1_3, A_reg_3, B_reg_0, (1));
  C_reg_2_3 = vfmaq_laneq_f32(C_reg_2_3, A_reg_3, B_reg_0, (2));
  C_reg_3_3 = vfmaq_laneq_f32(C_reg_3_3, A_reg_3, B_reg_0, (3));
}
vst1q_f32(&C.data[0], C_reg_0_0);
vst1q_f32(&C.data[(4) * (C.strides[1])], C_reg_0_1);
vst1q_f32(&C.data[(8) * (C.strides[1])], C_reg_0_2);
vst1q_f32(&C.data[(12) * (C.strides[1])], C_reg_0_3);
vst1q_f32(&C.data[C.strides[0]], C_reg_1_0);
vst1q_f32(&C.data[C.strides[0] + (4) * (C.strides[1])], C_reg_1_1);
vst1q_f32(&C.data[C.strides[0] + (8) * (C.strides[1])], C_reg_1_2);
vst1q_f32(&C.data[C.strides[0] + (12) * (C.strides[1])], C_reg_1_3);
vst1q_f32(&C.data[(2) * (C.strides[0])], C_reg_2_0);
vst1q_f32(&C.data[(2) * (C.strides[0]) + (4) * (C.strides[1])], C_reg_2_1);
vst1q_f32(&C.data[(2) * (C.strides[0]) + (8) * (C.strides[1])], C_reg_2_2);
vst1q_f32(&C.data[(2) * (C.strides[0]) + (12) * (C.strides[1])], C_reg_2_3);
vst1q_f32(&C.data[(3) * (C.strides[0])], C_reg_3_0);
vst1q_f32(&C.data[(3) * (C.strides[0]) + (4) * (C.strides[1])], C_reg_3_1);
vst1q_f32(&C.data[(3) * (C.strides[0]) + (8) * (C.strides[1])], C_reg_3_2);
vst1q_f32(&C.data[(3) * (C.strides[0]) + (12) * (C.strides[1])], C_reg_3_3);
}

// gemm_NEON_16x4_beta1(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 16] @DRAM,
//     B : f32[KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][4, 16] @DRAM
// )
void gemm_NEON_16x4_beta1( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C ) {
float *Ba = (float*) malloc(KC * 4 * sizeof(*Ba));
free(Ba);
float *Cb = (float*) malloc(4 * 16 * sizeof(*Cb));
free(Cb);
float32x4_t C_reg_0_0;
float32x4_t C_reg_0_1;
float32x4_t C_reg_0_2;
float32x4_t C_reg_0_3;
float32x4_t C_reg_1_0;
float32x4_t C_reg_1_1;
float32x4_t C_reg_1_2;
float32x4_t C_reg_1_3;
float32x4_t C_reg_2_0;
float32x4_t C_reg_2_1;
float32x4_t C_reg_2_2;
float32x4_t C_reg_2_3;
float32x4_t C_reg_3_0;
float32x4_t C_reg_3_1;
float32x4_t C_reg_3_2;
float32x4_t C_reg_3_3;
C_reg_0_0 = vld1q_f32(&C.data[0]);
C_reg_0_1 = vld1q_f32(&C.data[(4) * (C.strides[1])]);
C_reg_0_2 = vld1q_f32(&C.data[(8) * (C.strides[1])]);
C_reg_0_3 = vld1q_f32(&C.data[(12) * (C.strides[1])]);
C_reg_1_0 = vld1q_f32(&C.data[C.strides[0]]);
C_reg_1_1 = vld1q_f32(&C.data[C.strides[0] + (4) * (C.strides[1])]);
C_reg_1_2 = vld1q_f32(&C.data[C.strides[0] + (8) * (C.strides[1])]);
C_reg_1_3 = vld1q_f32(&C.data[C.strides[0] + (12) * (C.strides[1])]);
C_reg_2_0 = vld1q_f32(&C.data[(2) * (C.strides[0])]);
C_reg_2_1 = vld1q_f32(&C.data[(2) * (C.strides[0]) + (4) * (C.strides[1])]);
C_reg_2_2 = vld1q_f32(&C.data[(2) * (C.strides[0]) + (8) * (C.strides[1])]);
C_reg_2_3 = vld1q_f32(&C.data[(2) * (C.strides[0]) + (12) * (C.strides[1])]);
C_reg_3_0 = vld1q_f32(&C.data[(3) * (C.strides[0])]);
C_reg_3_1 = vld1q_f32(&C.data[(3) * (C.strides[0]) + (4) * (C.strides[1])]);
C_reg_3_2 = vld1q_f32(&C.data[(3) * (C.strides[0]) + (8) * (C.strides[1])]);
C_reg_3_3 = vld1q_f32(&C.data[(3) * (C.strides[0]) + (12) * (C.strides[1])]);
float32x4_t A_reg_0;
float32x4_t A_reg_1;
float32x4_t A_reg_2;
float32x4_t A_reg_3;
float32x4_t B_reg_0;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = vld1q_f32(&A[(k) * (16)]);
  A_reg_1 = vld1q_f32(&A[(k) * (16) + 4]);
  A_reg_2 = vld1q_f32(&A[(k) * (16) + 8]);
  A_reg_3 = vld1q_f32(&A[(k) * (16) + 12]);
  B_reg_0 = vld1q_f32(&B[(k) * 4]);
  C_reg_0_0 = vfmaq_laneq_f32(C_reg_0_0, A_reg_0, B_reg_0, (0));
  C_reg_1_0 = vfmaq_laneq_f32(C_reg_1_0, A_reg_0, B_reg_0, (1));
  C_reg_2_0 = vfmaq_laneq_f32(C_reg_2_0, A_reg_0, B_reg_0, (2));
  C_reg_3_0 = vfmaq_laneq_f32(C_reg_3_0, A_reg_0, B_reg_0, (3));
  C_reg_0_1 = vfmaq_laneq_f32(C_reg_0_1, A_reg_1, B_reg_0, (0));
  C_reg_1_1 = vfmaq_laneq_f32(C_reg_1_1, A_reg_1, B_reg_0, (1));
  C_reg_2_1 = vfmaq_laneq_f32(C_reg_2_1, A_reg_1, B_reg_0, (2));
  C_reg_3_1 = vfmaq_laneq_f32(C_reg_3_1, A_reg_1, B_reg_0, (3));
  C_reg_0_2 = vfmaq_laneq_f32(C_reg_0_2, A_reg_2, B_reg_0, (0));
  C_reg_1_2 = vfmaq_laneq_f32(C_reg_1_2, A_reg_2, B_reg_0, (1));
  C_reg_2_2 = vfmaq_laneq_f32(C_reg_2_2, A_reg_2, B_reg_0, (2));
  C_reg_3_2 = vfmaq_laneq_f32(C_reg_3_2, A_reg_2, B_reg_0, (3));
  C_reg_0_3 = vfmaq_laneq_f32(C_reg_0_3, A_reg_3, B_reg_0, (0));
  C_reg_1_3 = vfmaq_laneq_f32(C_reg_1_3, A_reg_3, B_reg_0, (1));
  C_reg_2_3 = vfmaq_laneq_f32(C_reg_2_3, A_reg_3, B_reg_0, (2));
  C_reg_3_3 = vfmaq_laneq_f32(C_reg_3_3, A_reg_3, B_reg_0, (3));
}
vst1q_f32(&C.data[0], C_reg_0_0);
vst1q_f32(&C.data[(4) * (C.strides[1])], C_reg_0_1);
vst1q_f32(&C.data[(8) * (C.strides[1])], C_reg_0_2);
vst1q_f32(&C.data[(12) * (C.strides[1])], C_reg_0_3);
vst1q_f32(&C.data[C.strides[0]], C_reg_1_0);
vst1q_f32(&C.data[C.strides[0] + (4) * (C.strides[1])], C_reg_1_1);
vst1q_f32(&C.data[C.strides[0] + (8) * (C.strides[1])], C_reg_1_2);
vst1q_f32(&C.data[C.strides[0] + (12) * (C.strides[1])], C_reg_1_3);
vst1q_f32(&C.data[(2) * (C.strides[0])], C_reg_2_0);
vst1q_f32(&C.data[(2) * (C.strides[0]) + (4) * (C.strides[1])], C_reg_2_1);
vst1q_f32(&C.data[(2) * (C.strides[0]) + (8) * (C.strides[1])], C_reg_2_2);
vst1q_f32(&C.data[(2) * (C.strides[0]) + (12) * (C.strides[1])], C_reg_2_3);
vst1q_f32(&C.data[(3) * (C.strides[0])], C_reg_3_0);
vst1q_f32(&C.data[(3) * (C.strides[0]) + (4) * (C.strides[1])], C_reg_3_1);
vst1q_f32(&C.data[(3) * (C.strides[0]) + (8) * (C.strides[1])], C_reg_3_2);
vst1q_f32(&C.data[(3) * (C.strides[0]) + (12) * (C.strides[1])], C_reg_3_3);
}

// gemm_NEON_20x4_beta0(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 20] @DRAM,
//     B : f32[KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][4, 20] @DRAM
// )
void gemm_NEON_20x4_beta0( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C ) {
float32x4_t C_reg_0_0;
float32x4_t C_reg_0_1;
float32x4_t C_reg_0_2;
float32x4_t C_reg_0_3;
float32x4_t C_reg_0_4;
float32x4_t C_reg_1_0;
float32x4_t C_reg_1_1;
float32x4_t C_reg_1_2;
float32x4_t C_reg_1_3;
float32x4_t C_reg_1_4;
float32x4_t C_reg_2_0;
float32x4_t C_reg_2_1;
float32x4_t C_reg_2_2;
float32x4_t C_reg_2_3;
float32x4_t C_reg_2_4;
float32x4_t C_reg_3_0;
float32x4_t C_reg_3_1;
float32x4_t C_reg_3_2;
float32x4_t C_reg_3_3;
float32x4_t C_reg_3_4;
C_reg_0_0 = vmovq_n_f32(0.0f);
C_reg_0_1 = vmovq_n_f32(0.0f);
C_reg_0_2 = vmovq_n_f32(0.0f);
C_reg_0_3 = vmovq_n_f32(0.0f);
C_reg_0_4 = vmovq_n_f32(0.0f);
C_reg_1_0 = vmovq_n_f32(0.0f);
C_reg_1_1 = vmovq_n_f32(0.0f);
C_reg_1_2 = vmovq_n_f32(0.0f);
C_reg_1_3 = vmovq_n_f32(0.0f);
C_reg_1_4 = vmovq_n_f32(0.0f);
C_reg_2_0 = vmovq_n_f32(0.0f);
C_reg_2_1 = vmovq_n_f32(0.0f);
C_reg_2_2 = vmovq_n_f32(0.0f);
C_reg_2_3 = vmovq_n_f32(0.0f);
C_reg_2_4 = vmovq_n_f32(0.0f);
C_reg_3_0 = vmovq_n_f32(0.0f);
C_reg_3_1 = vmovq_n_f32(0.0f);
C_reg_3_2 = vmovq_n_f32(0.0f);
C_reg_3_3 = vmovq_n_f32(0.0f);
C_reg_3_4 = vmovq_n_f32(0.0f);
float32x4_t A_reg_0;
float32x4_t A_reg_1;
float32x4_t A_reg_2;
float32x4_t A_reg_3;
float32x4_t A_reg_4;
float32x4_t B_reg_0;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = vld1q_f32(&A[(k) * (20)]);
  A_reg_1 = vld1q_f32(&A[(k) * (20) + 4]);
  A_reg_2 = vld1q_f32(&A[(k) * (20) + 8]);
  A_reg_3 = vld1q_f32(&A[(k) * (20) + 12]);
  A_reg_4 = vld1q_f32(&A[(k) * (20) + 16]);
  B_reg_0 = vld1q_f32(&B[(k) * 4]);
  C_reg_0_0 = vfmaq_laneq_f32(C_reg_0_0, A_reg_0, B_reg_0, (0));
  C_reg_1_0 = vfmaq_laneq_f32(C_reg_1_0, A_reg_0, B_reg_0, (1));
  C_reg_2_0 = vfmaq_laneq_f32(C_reg_2_0, A_reg_0, B_reg_0, (2));
  C_reg_3_0 = vfmaq_laneq_f32(C_reg_3_0, A_reg_0, B_reg_0, (3));
  C_reg_0_1 = vfmaq_laneq_f32(C_reg_0_1, A_reg_1, B_reg_0, (0));
  C_reg_1_1 = vfmaq_laneq_f32(C_reg_1_1, A_reg_1, B_reg_0, (1));
  C_reg_2_1 = vfmaq_laneq_f32(C_reg_2_1, A_reg_1, B_reg_0, (2));
  C_reg_3_1 = vfmaq_laneq_f32(C_reg_3_1, A_reg_1, B_reg_0, (3));
  C_reg_0_2 = vfmaq_laneq_f32(C_reg_0_2, A_reg_2, B_reg_0, (0));
  C_reg_1_2 = vfmaq_laneq_f32(C_reg_1_2, A_reg_2, B_reg_0, (1));
  C_reg_2_2 = vfmaq_laneq_f32(C_reg_2_2, A_reg_2, B_reg_0, (2));
  C_reg_3_2 = vfmaq_laneq_f32(C_reg_3_2, A_reg_2, B_reg_0, (3));
  C_reg_0_3 = vfmaq_laneq_f32(C_reg_0_3, A_reg_3, B_reg_0, (0));
  C_reg_1_3 = vfmaq_laneq_f32(C_reg_1_3, A_reg_3, B_reg_0, (1));
  C_reg_2_3 = vfmaq_laneq_f32(C_reg_2_3, A_reg_3, B_reg_0, (2));
  C_reg_3_3 = vfmaq_laneq_f32(C_reg_3_3, A_reg_3, B_reg_0, (3));
  C_reg_0_4 = vfmaq_laneq_f32(C_reg_0_4, A_reg_4, B_reg_0, (0));
  C_reg_1_4 = vfmaq_laneq_f32(C_reg_1_4, A_reg_4, B_reg_0, (1));
  C_reg_2_4 = vfmaq_laneq_f32(C_reg_2_4, A_reg_4, B_reg_0, (2));
  C_reg_3_4 = vfmaq_laneq_f32(C_reg_3_4, A_reg_4, B_reg_0, (3));
}
vst1q_f32(&C.data[0], C_reg_0_0);
vst1q_f32(&C.data[(4) * (C.strides[1])], C_reg_0_1);
vst1q_f32(&C.data[(8) * (C.strides[1])], C_reg_0_2);
vst1q_f32(&C.data[(12) * (C.strides[1])], C_reg_0_3);
vst1q_f32(&C.data[(16) * (C.strides[1])], C_reg_0_4);
vst1q_f32(&C.data[C.strides[0]], C_reg_1_0);
vst1q_f32(&C.data[C.strides[0] + (4) * (C.strides[1])], C_reg_1_1);
vst1q_f32(&C.data[C.strides[0] + (8) * (C.strides[1])], C_reg_1_2);
vst1q_f32(&C.data[C.strides[0] + (12) * (C.strides[1])], C_reg_1_3);
vst1q_f32(&C.data[C.strides[0] + (16) * (C.strides[1])], C_reg_1_4);
vst1q_f32(&C.data[(2) * (C.strides[0])], C_reg_2_0);
vst1q_f32(&C.data[(2) * (C.strides[0]) + (4) * (C.strides[1])], C_reg_2_1);
vst1q_f32(&C.data[(2) * (C.strides[0]) + (8) * (C.strides[1])], C_reg_2_2);
vst1q_f32(&C.data[(2) * (C.strides[0]) + (12) * (C.strides[1])], C_reg_2_3);
vst1q_f32(&C.data[(2) * (C.strides[0]) + (16) * (C.strides[1])], C_reg_2_4);
vst1q_f32(&C.data[(3) * (C.strides[0])], C_reg_3_0);
vst1q_f32(&C.data[(3) * (C.strides[0]) + (4) * (C.strides[1])], C_reg_3_1);
vst1q_f32(&C.data[(3) * (C.strides[0]) + (8) * (C.strides[1])], C_reg_3_2);
vst1q_f32(&C.data[(3) * (C.strides[0]) + (12) * (C.strides[1])], C_reg_3_3);
vst1q_f32(&C.data[(3) * (C.strides[0]) + (16) * (C.strides[1])], C_reg_3_4);
}

// gemm_NEON_20x4_beta1(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 20] @DRAM,
//     B : f32[KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][4, 20] @DRAM
// )
void gemm_NEON_20x4_beta1( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C ) {
float *Ba = (float*) malloc(KC * 4 * sizeof(*Ba));
free(Ba);
float *Cb = (float*) malloc(4 * 20 * sizeof(*Cb));
free(Cb);
float32x4_t C_reg_0_0;
float32x4_t C_reg_0_1;
float32x4_t C_reg_0_2;
float32x4_t C_reg_0_3;
float32x4_t C_reg_0_4;
float32x4_t C_reg_1_0;
float32x4_t C_reg_1_1;
float32x4_t C_reg_1_2;
float32x4_t C_reg_1_3;
float32x4_t C_reg_1_4;
float32x4_t C_reg_2_0;
float32x4_t C_reg_2_1;
float32x4_t C_reg_2_2;
float32x4_t C_reg_2_3;
float32x4_t C_reg_2_4;
float32x4_t C_reg_3_0;
float32x4_t C_reg_3_1;
float32x4_t C_reg_3_2;
float32x4_t C_reg_3_3;
float32x4_t C_reg_3_4;
C_reg_0_0 = vld1q_f32(&C.data[0]);
C_reg_0_1 = vld1q_f32(&C.data[(4) * (C.strides[1])]);
C_reg_0_2 = vld1q_f32(&C.data[(8) * (C.strides[1])]);
C_reg_0_3 = vld1q_f32(&C.data[(12) * (C.strides[1])]);
C_reg_0_4 = vld1q_f32(&C.data[(16) * (C.strides[1])]);
C_reg_1_0 = vld1q_f32(&C.data[C.strides[0]]);
C_reg_1_1 = vld1q_f32(&C.data[C.strides[0] + (4) * (C.strides[1])]);
C_reg_1_2 = vld1q_f32(&C.data[C.strides[0] + (8) * (C.strides[1])]);
C_reg_1_3 = vld1q_f32(&C.data[C.strides[0] + (12) * (C.strides[1])]);
C_reg_1_4 = vld1q_f32(&C.data[C.strides[0] + (16) * (C.strides[1])]);
C_reg_2_0 = vld1q_f32(&C.data[(2) * (C.strides[0])]);
C_reg_2_1 = vld1q_f32(&C.data[(2) * (C.strides[0]) + (4) * (C.strides[1])]);
C_reg_2_2 = vld1q_f32(&C.data[(2) * (C.strides[0]) + (8) * (C.strides[1])]);
C_reg_2_3 = vld1q_f32(&C.data[(2) * (C.strides[0]) + (12) * (C.strides[1])]);
C_reg_2_4 = vld1q_f32(&C.data[(2) * (C.strides[0]) + (16) * (C.strides[1])]);
C_reg_3_0 = vld1q_f32(&C.data[(3) * (C.strides[0])]);
C_reg_3_1 = vld1q_f32(&C.data[(3) * (C.strides[0]) + (4) * (C.strides[1])]);
C_reg_3_2 = vld1q_f32(&C.data[(3) * (C.strides[0]) + (8) * (C.strides[1])]);
C_reg_3_3 = vld1q_f32(&C.data[(3) * (C.strides[0]) + (12) * (C.strides[1])]);
C_reg_3_4 = vld1q_f32(&C.data[(3) * (C.strides[0]) + (16) * (C.strides[1])]);
float32x4_t A_reg_0;
float32x4_t A_reg_1;
float32x4_t A_reg_2;
float32x4_t A_reg_3;
float32x4_t A_reg_4;
float32x4_t B_reg_0;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = vld1q_f32(&A[(k) * (20)]);
  A_reg_1 = vld1q_f32(&A[(k) * (20) + 4]);
  A_reg_2 = vld1q_f32(&A[(k) * (20) + 8]);
  A_reg_3 = vld1q_f32(&A[(k) * (20) + 12]);
  A_reg_4 = vld1q_f32(&A[(k) * (20) + 16]);
  B_reg_0 = vld1q_f32(&B[(k) * 4]);
  C_reg_0_0 = vfmaq_laneq_f32(C_reg_0_0, A_reg_0, B_reg_0, (0));
  C_reg_1_0 = vfmaq_laneq_f32(C_reg_1_0, A_reg_0, B_reg_0, (1));
  C_reg_2_0 = vfmaq_laneq_f32(C_reg_2_0, A_reg_0, B_reg_0, (2));
  C_reg_3_0 = vfmaq_laneq_f32(C_reg_3_0, A_reg_0, B_reg_0, (3));
  C_reg_0_1 = vfmaq_laneq_f32(C_reg_0_1, A_reg_1, B_reg_0, (0));
  C_reg_1_1 = vfmaq_laneq_f32(C_reg_1_1, A_reg_1, B_reg_0, (1));
  C_reg_2_1 = vfmaq_laneq_f32(C_reg_2_1, A_reg_1, B_reg_0, (2));
  C_reg_3_1 = vfmaq_laneq_f32(C_reg_3_1, A_reg_1, B_reg_0, (3));
  C_reg_0_2 = vfmaq_laneq_f32(C_reg_0_2, A_reg_2, B_reg_0, (0));
  C_reg_1_2 = vfmaq_laneq_f32(C_reg_1_2, A_reg_2, B_reg_0, (1));
  C_reg_2_2 = vfmaq_laneq_f32(C_reg_2_2, A_reg_2, B_reg_0, (2));
  C_reg_3_2 = vfmaq_laneq_f32(C_reg_3_2, A_reg_2, B_reg_0, (3));
  C_reg_0_3 = vfmaq_laneq_f32(C_reg_0_3, A_reg_3, B_reg_0, (0));
  C_reg_1_3 = vfmaq_laneq_f32(C_reg_1_3, A_reg_3, B_reg_0, (1));
  C_reg_2_3 = vfmaq_laneq_f32(C_reg_2_3, A_reg_3, B_reg_0, (2));
  C_reg_3_3 = vfmaq_laneq_f32(C_reg_3_3, A_reg_3, B_reg_0, (3));
  C_reg_0_4 = vfmaq_laneq_f32(C_reg_0_4, A_reg_4, B_reg_0, (0));
  C_reg_1_4 = vfmaq_laneq_f32(C_reg_1_4, A_reg_4, B_reg_0, (1));
  C_reg_2_4 = vfmaq_laneq_f32(C_reg_2_4, A_reg_4, B_reg_0, (2));
  C_reg_3_4 = vfmaq_laneq_f32(C_reg_3_4, A_reg_4, B_reg_0, (3));
}
vst1q_f32(&C.data[0], C_reg_0_0);
vst1q_f32(&C.data[(4) * (C.strides[1])], C_reg_0_1);
vst1q_f32(&C.data[(8) * (C.strides[1])], C_reg_0_2);
vst1q_f32(&C.data[(12) * (C.strides[1])], C_reg_0_3);
vst1q_f32(&C.data[(16) * (C.strides[1])], C_reg_0_4);
vst1q_f32(&C.data[C.strides[0]], C_reg_1_0);
vst1q_f32(&C.data[C.strides[0] + (4) * (C.strides[1])], C_reg_1_1);
vst1q_f32(&C.data[C.strides[0] + (8) * (C.strides[1])], C_reg_1_2);
vst1q_f32(&C.data[C.strides[0] + (12) * (C.strides[1])], C_reg_1_3);
vst1q_f32(&C.data[C.strides[0] + (16) * (C.strides[1])], C_reg_1_4);
vst1q_f32(&C.data[(2) * (C.strides[0])], C_reg_2_0);
vst1q_f32(&C.data[(2) * (C.strides[0]) + (4) * (C.strides[1])], C_reg_2_1);
vst1q_f32(&C.data[(2) * (C.strides[0]) + (8) * (C.strides[1])], C_reg_2_2);
vst1q_f32(&C.data[(2) * (C.strides[0]) + (12) * (C.strides[1])], C_reg_2_3);
vst1q_f32(&C.data[(2) * (C.strides[0]) + (16) * (C.strides[1])], C_reg_2_4);
vst1q_f32(&C.data[(3) * (C.strides[0])], C_reg_3_0);
vst1q_f32(&C.data[(3) * (C.strides[0]) + (4) * (C.strides[1])], C_reg_3_1);
vst1q_f32(&C.data[(3) * (C.strides[0]) + (8) * (C.strides[1])], C_reg_3_2);
vst1q_f32(&C.data[(3) * (C.strides[0]) + (12) * (C.strides[1])], C_reg_3_3);
vst1q_f32(&C.data[(3) * (C.strides[0]) + (16) * (C.strides[1])], C_reg_3_4);
}

// gemm_NEON_24x4_beta0(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 24] @DRAM,
//     B : f32[KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][4, 24] @DRAM
// )
void gemm_NEON_24x4_beta0( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C ) {
float32x4_t C_reg_0_0;
float32x4_t C_reg_0_1;
float32x4_t C_reg_0_2;
float32x4_t C_reg_0_3;
float32x4_t C_reg_0_4;
float32x4_t C_reg_0_5;
float32x4_t C_reg_1_0;
float32x4_t C_reg_1_1;
float32x4_t C_reg_1_2;
float32x4_t C_reg_1_3;
float32x4_t C_reg_1_4;
float32x4_t C_reg_1_5;
float32x4_t C_reg_2_0;
float32x4_t C_reg_2_1;
float32x4_t C_reg_2_2;
float32x4_t C_reg_2_3;
float32x4_t C_reg_2_4;
float32x4_t C_reg_2_5;
float32x4_t C_reg_3_0;
float32x4_t C_reg_3_1;
float32x4_t C_reg_3_2;
float32x4_t C_reg_3_3;
float32x4_t C_reg_3_4;
float32x4_t C_reg_3_5;
C_reg_0_0 = vmovq_n_f32(0.0f);
C_reg_0_1 = vmovq_n_f32(0.0f);
C_reg_0_2 = vmovq_n_f32(0.0f);
C_reg_0_3 = vmovq_n_f32(0.0f);
C_reg_0_4 = vmovq_n_f32(0.0f);
C_reg_0_5 = vmovq_n_f32(0.0f);
C_reg_1_0 = vmovq_n_f32(0.0f);
C_reg_1_1 = vmovq_n_f32(0.0f);
C_reg_1_2 = vmovq_n_f32(0.0f);
C_reg_1_3 = vmovq_n_f32(0.0f);
C_reg_1_4 = vmovq_n_f32(0.0f);
C_reg_1_5 = vmovq_n_f32(0.0f);
C_reg_2_0 = vmovq_n_f32(0.0f);
C_reg_2_1 = vmovq_n_f32(0.0f);
C_reg_2_2 = vmovq_n_f32(0.0f);
C_reg_2_3 = vmovq_n_f32(0.0f);
C_reg_2_4 = vmovq_n_f32(0.0f);
C_reg_2_5 = vmovq_n_f32(0.0f);
C_reg_3_0 = vmovq_n_f32(0.0f);
C_reg_3_1 = vmovq_n_f32(0.0f);
C_reg_3_2 = vmovq_n_f32(0.0f);
C_reg_3_3 = vmovq_n_f32(0.0f);
C_reg_3_4 = vmovq_n_f32(0.0f);
C_reg_3_5 = vmovq_n_f32(0.0f);
float32x4_t A_reg_0;
float32x4_t A_reg_1;
float32x4_t A_reg_2;
float32x4_t A_reg_3;
float32x4_t A_reg_4;
float32x4_t A_reg_5;
float32x4_t B_reg_0;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = vld1q_f32(&A[(k) * (24)]);
  A_reg_1 = vld1q_f32(&A[(k) * (24) + 4]);
  A_reg_2 = vld1q_f32(&A[(k) * (24) + 8]);
  A_reg_3 = vld1q_f32(&A[(k) * (24) + 12]);
  A_reg_4 = vld1q_f32(&A[(k) * (24) + 16]);
  A_reg_5 = vld1q_f32(&A[(k) * (24) + 20]);
  B_reg_0 = vld1q_f32(&B[(k) * 4]);
  C_reg_0_0 = vfmaq_laneq_f32(C_reg_0_0, A_reg_0, B_reg_0, (0));
  C_reg_1_0 = vfmaq_laneq_f32(C_reg_1_0, A_reg_0, B_reg_0, (1));
  C_reg_2_0 = vfmaq_laneq_f32(C_reg_2_0, A_reg_0, B_reg_0, (2));
  C_reg_3_0 = vfmaq_laneq_f32(C_reg_3_0, A_reg_0, B_reg_0, (3));
  C_reg_0_1 = vfmaq_laneq_f32(C_reg_0_1, A_reg_1, B_reg_0, (0));
  C_reg_1_1 = vfmaq_laneq_f32(C_reg_1_1, A_reg_1, B_reg_0, (1));
  C_reg_2_1 = vfmaq_laneq_f32(C_reg_2_1, A_reg_1, B_reg_0, (2));
  C_reg_3_1 = vfmaq_laneq_f32(C_reg_3_1, A_reg_1, B_reg_0, (3));
  C_reg_0_2 = vfmaq_laneq_f32(C_reg_0_2, A_reg_2, B_reg_0, (0));
  C_reg_1_2 = vfmaq_laneq_f32(C_reg_1_2, A_reg_2, B_reg_0, (1));
  C_reg_2_2 = vfmaq_laneq_f32(C_reg_2_2, A_reg_2, B_reg_0, (2));
  C_reg_3_2 = vfmaq_laneq_f32(C_reg_3_2, A_reg_2, B_reg_0, (3));
  C_reg_0_3 = vfmaq_laneq_f32(C_reg_0_3, A_reg_3, B_reg_0, (0));
  C_reg_1_3 = vfmaq_laneq_f32(C_reg_1_3, A_reg_3, B_reg_0, (1));
  C_reg_2_3 = vfmaq_laneq_f32(C_reg_2_3, A_reg_3, B_reg_0, (2));
  C_reg_3_3 = vfmaq_laneq_f32(C_reg_3_3, A_reg_3, B_reg_0, (3));
  C_reg_0_4 = vfmaq_laneq_f32(C_reg_0_4, A_reg_4, B_reg_0, (0));
  C_reg_1_4 = vfmaq_laneq_f32(C_reg_1_4, A_reg_4, B_reg_0, (1));
  C_reg_2_4 = vfmaq_laneq_f32(C_reg_2_4, A_reg_4, B_reg_0, (2));
  C_reg_3_4 = vfmaq_laneq_f32(C_reg_3_4, A_reg_4, B_reg_0, (3));
  C_reg_0_5 = vfmaq_laneq_f32(C_reg_0_5, A_reg_5, B_reg_0, (0));
  C_reg_1_5 = vfmaq_laneq_f32(C_reg_1_5, A_reg_5, B_reg_0, (1));
  C_reg_2_5 = vfmaq_laneq_f32(C_reg_2_5, A_reg_5, B_reg_0, (2));
  C_reg_3_5 = vfmaq_laneq_f32(C_reg_3_5, A_reg_5, B_reg_0, (3));
}
vst1q_f32(&C.data[0], C_reg_0_0);
vst1q_f32(&C.data[(4) * (C.strides[1])], C_reg_0_1);
vst1q_f32(&C.data[(8) * (C.strides[1])], C_reg_0_2);
vst1q_f32(&C.data[(12) * (C.strides[1])], C_reg_0_3);
vst1q_f32(&C.data[(16) * (C.strides[1])], C_reg_0_4);
vst1q_f32(&C.data[(20) * (C.strides[1])], C_reg_0_5);
vst1q_f32(&C.data[C.strides[0]], C_reg_1_0);
vst1q_f32(&C.data[C.strides[0] + (4) * (C.strides[1])], C_reg_1_1);
vst1q_f32(&C.data[C.strides[0] + (8) * (C.strides[1])], C_reg_1_2);
vst1q_f32(&C.data[C.strides[0] + (12) * (C.strides[1])], C_reg_1_3);
vst1q_f32(&C.data[C.strides[0] + (16) * (C.strides[1])], C_reg_1_4);
vst1q_f32(&C.data[C.strides[0] + (20) * (C.strides[1])], C_reg_1_5);
vst1q_f32(&C.data[(2) * (C.strides[0])], C_reg_2_0);
vst1q_f32(&C.data[(2) * (C.strides[0]) + (4) * (C.strides[1])], C_reg_2_1);
vst1q_f32(&C.data[(2) * (C.strides[0]) + (8) * (C.strides[1])], C_reg_2_2);
vst1q_f32(&C.data[(2) * (C.strides[0]) + (12) * (C.strides[1])], C_reg_2_3);
vst1q_f32(&C.data[(2) * (C.strides[0]) + (16) * (C.strides[1])], C_reg_2_4);
vst1q_f32(&C.data[(2) * (C.strides[0]) + (20) * (C.strides[1])], C_reg_2_5);
vst1q_f32(&C.data[(3) * (C.strides[0])], C_reg_3_0);
vst1q_f32(&C.data[(3) * (C.strides[0]) + (4) * (C.strides[1])], C_reg_3_1);
vst1q_f32(&C.data[(3) * (C.strides[0]) + (8) * (C.strides[1])], C_reg_3_2);
vst1q_f32(&C.data[(3) * (C.strides[0]) + (12) * (C.strides[1])], C_reg_3_3);
vst1q_f32(&C.data[(3) * (C.strides[0]) + (16) * (C.strides[1])], C_reg_3_4);
vst1q_f32(&C.data[(3) * (C.strides[0]) + (20) * (C.strides[1])], C_reg_3_5);
}

// gemm_NEON_24x4_beta1(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 24] @DRAM,
//     B : f32[KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][4, 24] @DRAM
// )
void gemm_NEON_24x4_beta1( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C ) {
float *Ba = (float*) malloc(KC * 4 * sizeof(*Ba));
free(Ba);
float *Cb = (float*) malloc(4 * 24 * sizeof(*Cb));
free(Cb);
float32x4_t C_reg_0_0;
float32x4_t C_reg_0_1;
float32x4_t C_reg_0_2;
float32x4_t C_reg_0_3;
float32x4_t C_reg_0_4;
float32x4_t C_reg_0_5;
float32x4_t C_reg_1_0;
float32x4_t C_reg_1_1;
float32x4_t C_reg_1_2;
float32x4_t C_reg_1_3;
float32x4_t C_reg_1_4;
float32x4_t C_reg_1_5;
float32x4_t C_reg_2_0;
float32x4_t C_reg_2_1;
float32x4_t C_reg_2_2;
float32x4_t C_reg_2_3;
float32x4_t C_reg_2_4;
float32x4_t C_reg_2_5;
float32x4_t C_reg_3_0;
float32x4_t C_reg_3_1;
float32x4_t C_reg_3_2;
float32x4_t C_reg_3_3;
float32x4_t C_reg_3_4;
float32x4_t C_reg_3_5;
C_reg_0_0 = vld1q_f32(&C.data[0]);
C_reg_0_1 = vld1q_f32(&C.data[(4) * (C.strides[1])]);
C_reg_0_2 = vld1q_f32(&C.data[(8) * (C.strides[1])]);
C_reg_0_3 = vld1q_f32(&C.data[(12) * (C.strides[1])]);
C_reg_0_4 = vld1q_f32(&C.data[(16) * (C.strides[1])]);
C_reg_0_5 = vld1q_f32(&C.data[(20) * (C.strides[1])]);
C_reg_1_0 = vld1q_f32(&C.data[C.strides[0]]);
C_reg_1_1 = vld1q_f32(&C.data[C.strides[0] + (4) * (C.strides[1])]);
C_reg_1_2 = vld1q_f32(&C.data[C.strides[0] + (8) * (C.strides[1])]);
C_reg_1_3 = vld1q_f32(&C.data[C.strides[0] + (12) * (C.strides[1])]);
C_reg_1_4 = vld1q_f32(&C.data[C.strides[0] + (16) * (C.strides[1])]);
C_reg_1_5 = vld1q_f32(&C.data[C.strides[0] + (20) * (C.strides[1])]);
C_reg_2_0 = vld1q_f32(&C.data[(2) * (C.strides[0])]);
C_reg_2_1 = vld1q_f32(&C.data[(2) * (C.strides[0]) + (4) * (C.strides[1])]);
C_reg_2_2 = vld1q_f32(&C.data[(2) * (C.strides[0]) + (8) * (C.strides[1])]);
C_reg_2_3 = vld1q_f32(&C.data[(2) * (C.strides[0]) + (12) * (C.strides[1])]);
C_reg_2_4 = vld1q_f32(&C.data[(2) * (C.strides[0]) + (16) * (C.strides[1])]);
C_reg_2_5 = vld1q_f32(&C.data[(2) * (C.strides[0]) + (20) * (C.strides[1])]);
C_reg_3_0 = vld1q_f32(&C.data[(3) * (C.strides[0])]);
C_reg_3_1 = vld1q_f32(&C.data[(3) * (C.strides[0]) + (4) * (C.strides[1])]);
C_reg_3_2 = vld1q_f32(&C.data[(3) * (C.strides[0]) + (8) * (C.strides[1])]);
C_reg_3_3 = vld1q_f32(&C.data[(3) * (C.strides[0]) + (12) * (C.strides[1])]);
C_reg_3_4 = vld1q_f32(&C.data[(3) * (C.strides[0]) + (16) * (C.strides[1])]);
C_reg_3_5 = vld1q_f32(&C.data[(3) * (C.strides[0]) + (20) * (C.strides[1])]);
float32x4_t A_reg_0;
float32x4_t A_reg_1;
float32x4_t A_reg_2;
float32x4_t A_reg_3;
float32x4_t A_reg_4;
float32x4_t A_reg_5;
float32x4_t B_reg_0;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = vld1q_f32(&A[(k) * (24)]);
  A_reg_1 = vld1q_f32(&A[(k) * (24) + 4]);
  A_reg_2 = vld1q_f32(&A[(k) * (24) + 8]);
  A_reg_3 = vld1q_f32(&A[(k) * (24) + 12]);
  A_reg_4 = vld1q_f32(&A[(k) * (24) + 16]);
  A_reg_5 = vld1q_f32(&A[(k) * (24) + 20]);
  B_reg_0 = vld1q_f32(&B[(k) * 4]);
  C_reg_0_0 = vfmaq_laneq_f32(C_reg_0_0, A_reg_0, B_reg_0, (0));
  C_reg_1_0 = vfmaq_laneq_f32(C_reg_1_0, A_reg_0, B_reg_0, (1));
  C_reg_2_0 = vfmaq_laneq_f32(C_reg_2_0, A_reg_0, B_reg_0, (2));
  C_reg_3_0 = vfmaq_laneq_f32(C_reg_3_0, A_reg_0, B_reg_0, (3));
  C_reg_0_1 = vfmaq_laneq_f32(C_reg_0_1, A_reg_1, B_reg_0, (0));
  C_reg_1_1 = vfmaq_laneq_f32(C_reg_1_1, A_reg_1, B_reg_0, (1));
  C_reg_2_1 = vfmaq_laneq_f32(C_reg_2_1, A_reg_1, B_reg_0, (2));
  C_reg_3_1 = vfmaq_laneq_f32(C_reg_3_1, A_reg_1, B_reg_0, (3));
  C_reg_0_2 = vfmaq_laneq_f32(C_reg_0_2, A_reg_2, B_reg_0, (0));
  C_reg_1_2 = vfmaq_laneq_f32(C_reg_1_2, A_reg_2, B_reg_0, (1));
  C_reg_2_2 = vfmaq_laneq_f32(C_reg_2_2, A_reg_2, B_reg_0, (2));
  C_reg_3_2 = vfmaq_laneq_f32(C_reg_3_2, A_reg_2, B_reg_0, (3));
  C_reg_0_3 = vfmaq_laneq_f32(C_reg_0_3, A_reg_3, B_reg_0, (0));
  C_reg_1_3 = vfmaq_laneq_f32(C_reg_1_3, A_reg_3, B_reg_0, (1));
  C_reg_2_3 = vfmaq_laneq_f32(C_reg_2_3, A_reg_3, B_reg_0, (2));
  C_reg_3_3 = vfmaq_laneq_f32(C_reg_3_3, A_reg_3, B_reg_0, (3));
  C_reg_0_4 = vfmaq_laneq_f32(C_reg_0_4, A_reg_4, B_reg_0, (0));
  C_reg_1_4 = vfmaq_laneq_f32(C_reg_1_4, A_reg_4, B_reg_0, (1));
  C_reg_2_4 = vfmaq_laneq_f32(C_reg_2_4, A_reg_4, B_reg_0, (2));
  C_reg_3_4 = vfmaq_laneq_f32(C_reg_3_4, A_reg_4, B_reg_0, (3));
  C_reg_0_5 = vfmaq_laneq_f32(C_reg_0_5, A_reg_5, B_reg_0, (0));
  C_reg_1_5 = vfmaq_laneq_f32(C_reg_1_5, A_reg_5, B_reg_0, (1));
  C_reg_2_5 = vfmaq_laneq_f32(C_reg_2_5, A_reg_5, B_reg_0, (2));
  C_reg_3_5 = vfmaq_laneq_f32(C_reg_3_5, A_reg_5, B_reg_0, (3));
}
vst1q_f32(&C.data[0], C_reg_0_0);
vst1q_f32(&C.data[(4) * (C.strides[1])], C_reg_0_1);
vst1q_f32(&C.data[(8) * (C.strides[1])], C_reg_0_2);
vst1q_f32(&C.data[(12) * (C.strides[1])], C_reg_0_3);
vst1q_f32(&C.data[(16) * (C.strides[1])], C_reg_0_4);
vst1q_f32(&C.data[(20) * (C.strides[1])], C_reg_0_5);
vst1q_f32(&C.data[C.strides[0]], C_reg_1_0);
vst1q_f32(&C.data[C.strides[0] + (4) * (C.strides[1])], C_reg_1_1);
vst1q_f32(&C.data[C.strides[0] + (8) * (C.strides[1])], C_reg_1_2);
vst1q_f32(&C.data[C.strides[0] + (12) * (C.strides[1])], C_reg_1_3);
vst1q_f32(&C.data[C.strides[0] + (16) * (C.strides[1])], C_reg_1_4);
vst1q_f32(&C.data[C.strides[0] + (20) * (C.strides[1])], C_reg_1_5);
vst1q_f32(&C.data[(2) * (C.strides[0])], C_reg_2_0);
vst1q_f32(&C.data[(2) * (C.strides[0]) + (4) * (C.strides[1])], C_reg_2_1);
vst1q_f32(&C.data[(2) * (C.strides[0]) + (8) * (C.strides[1])], C_reg_2_2);
vst1q_f32(&C.data[(2) * (C.strides[0]) + (12) * (C.strides[1])], C_reg_2_3);
vst1q_f32(&C.data[(2) * (C.strides[0]) + (16) * (C.strides[1])], C_reg_2_4);
vst1q_f32(&C.data[(2) * (C.strides[0]) + (20) * (C.strides[1])], C_reg_2_5);
vst1q_f32(&C.data[(3) * (C.strides[0])], C_reg_3_0);
vst1q_f32(&C.data[(3) * (C.strides[0]) + (4) * (C.strides[1])], C_reg_3_1);
vst1q_f32(&C.data[(3) * (C.strides[0]) + (8) * (C.strides[1])], C_reg_3_2);
vst1q_f32(&C.data[(3) * (C.strides[0]) + (12) * (C.strides[1])], C_reg_3_3);
vst1q_f32(&C.data[(3) * (C.strides[0]) + (16) * (C.strides[1])], C_reg_3_4);
vst1q_f32(&C.data[(3) * (C.strides[0]) + (20) * (C.strides[1])], C_reg_3_5);
}

// gemm_NEON_4x12_beta0(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 4] @DRAM,
//     B : f32[KC, 12] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][12, 4] @DRAM
// )
void gemm_NEON_4x12_beta0( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C ) {
float32x4_t C_reg_0_0;
float32x4_t C_reg_1_0;
float32x4_t C_reg_2_0;
float32x4_t C_reg_3_0;
float32x4_t C_reg_4_0;
float32x4_t C_reg_5_0;
float32x4_t C_reg_6_0;
float32x4_t C_reg_7_0;
float32x4_t C_reg_8_0;
float32x4_t C_reg_9_0;
float32x4_t C_reg_10_0;
float32x4_t C_reg_11_0;
C_reg_0_0 = vmovq_n_f32(0.0f);
C_reg_1_0 = vmovq_n_f32(0.0f);
C_reg_2_0 = vmovq_n_f32(0.0f);
C_reg_3_0 = vmovq_n_f32(0.0f);
C_reg_4_0 = vmovq_n_f32(0.0f);
C_reg_5_0 = vmovq_n_f32(0.0f);
C_reg_6_0 = vmovq_n_f32(0.0f);
C_reg_7_0 = vmovq_n_f32(0.0f);
C_reg_8_0 = vmovq_n_f32(0.0f);
C_reg_9_0 = vmovq_n_f32(0.0f);
C_reg_10_0 = vmovq_n_f32(0.0f);
C_reg_11_0 = vmovq_n_f32(0.0f);
float32x4_t A_reg_0;
float32x4_t B_reg_0;
float32x4_t B_reg_1;
float32x4_t B_reg_2;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = vld1q_f32(&A[(k) * 4]);
  B_reg_0 = vld1q_f32(&B[(k) * (12)]);
  B_reg_1 = vld1q_f32(&B[(k) * (12) + 4]);
  B_reg_2 = vld1q_f32(&B[(k) * (12) + 8]);
  C_reg_0_0 = vfmaq_laneq_f32(C_reg_0_0, A_reg_0, B_reg_0, (0));
  C_reg_1_0 = vfmaq_laneq_f32(C_reg_1_0, A_reg_0, B_reg_0, (1));
  C_reg_2_0 = vfmaq_laneq_f32(C_reg_2_0, A_reg_0, B_reg_0, (2));
  C_reg_3_0 = vfmaq_laneq_f32(C_reg_3_0, A_reg_0, B_reg_0, (3));
  C_reg_4_0 = vfmaq_laneq_f32(C_reg_4_0, A_reg_0, B_reg_1, (0));
  C_reg_5_0 = vfmaq_laneq_f32(C_reg_5_0, A_reg_0, B_reg_1, (1));
  C_reg_6_0 = vfmaq_laneq_f32(C_reg_6_0, A_reg_0, B_reg_1, (2));
  C_reg_7_0 = vfmaq_laneq_f32(C_reg_7_0, A_reg_0, B_reg_1, (3));
  C_reg_8_0 = vfmaq_laneq_f32(C_reg_8_0, A_reg_0, B_reg_2, (0));
  C_reg_9_0 = vfmaq_laneq_f32(C_reg_9_0, A_reg_0, B_reg_2, (1));
  C_reg_10_0 = vfmaq_laneq_f32(C_reg_10_0, A_reg_0, B_reg_2, (2));
  C_reg_11_0 = vfmaq_laneq_f32(C_reg_11_0, A_reg_0, B_reg_2, (3));
}
vst1q_f32(&C.data[0], C_reg_0_0);
vst1q_f32(&C.data[C.strides[0]], C_reg_1_0);
vst1q_f32(&C.data[(2) * (C.strides[0])], C_reg_2_0);
vst1q_f32(&C.data[(3) * (C.strides[0])], C_reg_3_0);
vst1q_f32(&C.data[(4) * (C.strides[0])], C_reg_4_0);
vst1q_f32(&C.data[(5) * (C.strides[0])], C_reg_5_0);
vst1q_f32(&C.data[(6) * (C.strides[0])], C_reg_6_0);
vst1q_f32(&C.data[(7) * (C.strides[0])], C_reg_7_0);
vst1q_f32(&C.data[(8) * (C.strides[0])], C_reg_8_0);
vst1q_f32(&C.data[(9) * (C.strides[0])], C_reg_9_0);
vst1q_f32(&C.data[(10) * (C.strides[0])], C_reg_10_0);
vst1q_f32(&C.data[(11) * (C.strides[0])], C_reg_11_0);
}

// gemm_NEON_4x12_beta1(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 4] @DRAM,
//     B : f32[KC, 12] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][12, 4] @DRAM
// )
void gemm_NEON_4x12_beta1( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C ) {
float *Ba = (float*) malloc(KC * 12 * sizeof(*Ba));
free(Ba);
float *Cb = (float*) malloc(12 * 4 * sizeof(*Cb));
free(Cb);
float32x4_t C_reg_0_0;
float32x4_t C_reg_1_0;
float32x4_t C_reg_2_0;
float32x4_t C_reg_3_0;
float32x4_t C_reg_4_0;
float32x4_t C_reg_5_0;
float32x4_t C_reg_6_0;
float32x4_t C_reg_7_0;
float32x4_t C_reg_8_0;
float32x4_t C_reg_9_0;
float32x4_t C_reg_10_0;
float32x4_t C_reg_11_0;
C_reg_0_0 = vld1q_f32(&C.data[0]);
C_reg_1_0 = vld1q_f32(&C.data[C.strides[0]]);
C_reg_2_0 = vld1q_f32(&C.data[(2) * (C.strides[0])]);
C_reg_3_0 = vld1q_f32(&C.data[(3) * (C.strides[0])]);
C_reg_4_0 = vld1q_f32(&C.data[(4) * (C.strides[0])]);
C_reg_5_0 = vld1q_f32(&C.data[(5) * (C.strides[0])]);
C_reg_6_0 = vld1q_f32(&C.data[(6) * (C.strides[0])]);
C_reg_7_0 = vld1q_f32(&C.data[(7) * (C.strides[0])]);
C_reg_8_0 = vld1q_f32(&C.data[(8) * (C.strides[0])]);
C_reg_9_0 = vld1q_f32(&C.data[(9) * (C.strides[0])]);
C_reg_10_0 = vld1q_f32(&C.data[(10) * (C.strides[0])]);
C_reg_11_0 = vld1q_f32(&C.data[(11) * (C.strides[0])]);
float32x4_t A_reg_0;
float32x4_t B_reg_0;
float32x4_t B_reg_1;
float32x4_t B_reg_2;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = vld1q_f32(&A[(k) * 4]);
  B_reg_0 = vld1q_f32(&B[(k) * (12)]);
  B_reg_1 = vld1q_f32(&B[(k) * (12) + 4]);
  B_reg_2 = vld1q_f32(&B[(k) * (12) + 8]);
  C_reg_0_0 = vfmaq_laneq_f32(C_reg_0_0, A_reg_0, B_reg_0, (0));
  C_reg_1_0 = vfmaq_laneq_f32(C_reg_1_0, A_reg_0, B_reg_0, (1));
  C_reg_2_0 = vfmaq_laneq_f32(C_reg_2_0, A_reg_0, B_reg_0, (2));
  C_reg_3_0 = vfmaq_laneq_f32(C_reg_3_0, A_reg_0, B_reg_0, (3));
  C_reg_4_0 = vfmaq_laneq_f32(C_reg_4_0, A_reg_0, B_reg_1, (0));
  C_reg_5_0 = vfmaq_laneq_f32(C_reg_5_0, A_reg_0, B_reg_1, (1));
  C_reg_6_0 = vfmaq_laneq_f32(C_reg_6_0, A_reg_0, B_reg_1, (2));
  C_reg_7_0 = vfmaq_laneq_f32(C_reg_7_0, A_reg_0, B_reg_1, (3));
  C_reg_8_0 = vfmaq_laneq_f32(C_reg_8_0, A_reg_0, B_reg_2, (0));
  C_reg_9_0 = vfmaq_laneq_f32(C_reg_9_0, A_reg_0, B_reg_2, (1));
  C_reg_10_0 = vfmaq_laneq_f32(C_reg_10_0, A_reg_0, B_reg_2, (2));
  C_reg_11_0 = vfmaq_laneq_f32(C_reg_11_0, A_reg_0, B_reg_2, (3));
}
vst1q_f32(&C.data[0], C_reg_0_0);
vst1q_f32(&C.data[C.strides[0]], C_reg_1_0);
vst1q_f32(&C.data[(2) * (C.strides[0])], C_reg_2_0);
vst1q_f32(&C.data[(3) * (C.strides[0])], C_reg_3_0);
vst1q_f32(&C.data[(4) * (C.strides[0])], C_reg_4_0);
vst1q_f32(&C.data[(5) * (C.strides[0])], C_reg_5_0);
vst1q_f32(&C.data[(6) * (C.strides[0])], C_reg_6_0);
vst1q_f32(&C.data[(7) * (C.strides[0])], C_reg_7_0);
vst1q_f32(&C.data[(8) * (C.strides[0])], C_reg_8_0);
vst1q_f32(&C.data[(9) * (C.strides[0])], C_reg_9_0);
vst1q_f32(&C.data[(10) * (C.strides[0])], C_reg_10_0);
vst1q_f32(&C.data[(11) * (C.strides[0])], C_reg_11_0);
}

// gemm_NEON_4x16_beta0(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 4] @DRAM,
//     B : f32[KC, 16] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][16, 4] @DRAM
// )
void gemm_NEON_4x16_beta0( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C ) {
float32x4_t C_reg_0_0;
float32x4_t C_reg_1_0;
float32x4_t C_reg_2_0;
float32x4_t C_reg_3_0;
float32x4_t C_reg_4_0;
float32x4_t C_reg_5_0;
float32x4_t C_reg_6_0;
float32x4_t C_reg_7_0;
float32x4_t C_reg_8_0;
float32x4_t C_reg_9_0;
float32x4_t C_reg_10_0;
float32x4_t C_reg_11_0;
float32x4_t C_reg_12_0;
float32x4_t C_reg_13_0;
float32x4_t C_reg_14_0;
float32x4_t C_reg_15_0;
C_reg_0_0 = vmovq_n_f32(0.0f);
C_reg_1_0 = vmovq_n_f32(0.0f);
C_reg_2_0 = vmovq_n_f32(0.0f);
C_reg_3_0 = vmovq_n_f32(0.0f);
C_reg_4_0 = vmovq_n_f32(0.0f);
C_reg_5_0 = vmovq_n_f32(0.0f);
C_reg_6_0 = vmovq_n_f32(0.0f);
C_reg_7_0 = vmovq_n_f32(0.0f);
C_reg_8_0 = vmovq_n_f32(0.0f);
C_reg_9_0 = vmovq_n_f32(0.0f);
C_reg_10_0 = vmovq_n_f32(0.0f);
C_reg_11_0 = vmovq_n_f32(0.0f);
C_reg_12_0 = vmovq_n_f32(0.0f);
C_reg_13_0 = vmovq_n_f32(0.0f);
C_reg_14_0 = vmovq_n_f32(0.0f);
C_reg_15_0 = vmovq_n_f32(0.0f);
float32x4_t A_reg_0;
float32x4_t B_reg_0;
float32x4_t B_reg_1;
float32x4_t B_reg_2;
float32x4_t B_reg_3;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = vld1q_f32(&A[(k) * 4]);
  B_reg_0 = vld1q_f32(&B[(k) * (16)]);
  B_reg_1 = vld1q_f32(&B[(k) * (16) + 4]);
  B_reg_2 = vld1q_f32(&B[(k) * (16) + 8]);
  B_reg_3 = vld1q_f32(&B[(k) * (16) + 12]);
  C_reg_0_0 = vfmaq_laneq_f32(C_reg_0_0, A_reg_0, B_reg_0, (0));
  C_reg_1_0 = vfmaq_laneq_f32(C_reg_1_0, A_reg_0, B_reg_0, (1));
  C_reg_2_0 = vfmaq_laneq_f32(C_reg_2_0, A_reg_0, B_reg_0, (2));
  C_reg_3_0 = vfmaq_laneq_f32(C_reg_3_0, A_reg_0, B_reg_0, (3));
  C_reg_4_0 = vfmaq_laneq_f32(C_reg_4_0, A_reg_0, B_reg_1, (0));
  C_reg_5_0 = vfmaq_laneq_f32(C_reg_5_0, A_reg_0, B_reg_1, (1));
  C_reg_6_0 = vfmaq_laneq_f32(C_reg_6_0, A_reg_0, B_reg_1, (2));
  C_reg_7_0 = vfmaq_laneq_f32(C_reg_7_0, A_reg_0, B_reg_1, (3));
  C_reg_8_0 = vfmaq_laneq_f32(C_reg_8_0, A_reg_0, B_reg_2, (0));
  C_reg_9_0 = vfmaq_laneq_f32(C_reg_9_0, A_reg_0, B_reg_2, (1));
  C_reg_10_0 = vfmaq_laneq_f32(C_reg_10_0, A_reg_0, B_reg_2, (2));
  C_reg_11_0 = vfmaq_laneq_f32(C_reg_11_0, A_reg_0, B_reg_2, (3));
  C_reg_12_0 = vfmaq_laneq_f32(C_reg_12_0, A_reg_0, B_reg_3, (0));
  C_reg_13_0 = vfmaq_laneq_f32(C_reg_13_0, A_reg_0, B_reg_3, (1));
  C_reg_14_0 = vfmaq_laneq_f32(C_reg_14_0, A_reg_0, B_reg_3, (2));
  C_reg_15_0 = vfmaq_laneq_f32(C_reg_15_0, A_reg_0, B_reg_3, (3));
}
vst1q_f32(&C.data[0], C_reg_0_0);
vst1q_f32(&C.data[C.strides[0]], C_reg_1_0);
vst1q_f32(&C.data[(2) * (C.strides[0])], C_reg_2_0);
vst1q_f32(&C.data[(3) * (C.strides[0])], C_reg_3_0);
vst1q_f32(&C.data[(4) * (C.strides[0])], C_reg_4_0);
vst1q_f32(&C.data[(5) * (C.strides[0])], C_reg_5_0);
vst1q_f32(&C.data[(6) * (C.strides[0])], C_reg_6_0);
vst1q_f32(&C.data[(7) * (C.strides[0])], C_reg_7_0);
vst1q_f32(&C.data[(8) * (C.strides[0])], C_reg_8_0);
vst1q_f32(&C.data[(9) * (C.strides[0])], C_reg_9_0);
vst1q_f32(&C.data[(10) * (C.strides[0])], C_reg_10_0);
vst1q_f32(&C.data[(11) * (C.strides[0])], C_reg_11_0);
vst1q_f32(&C.data[(12) * (C.strides[0])], C_reg_12_0);
vst1q_f32(&C.data[(13) * (C.strides[0])], C_reg_13_0);
vst1q_f32(&C.data[(14) * (C.strides[0])], C_reg_14_0);
vst1q_f32(&C.data[(15) * (C.strides[0])], C_reg_15_0);
}

// gemm_NEON_4x16_beta1(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 4] @DRAM,
//     B : f32[KC, 16] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][16, 4] @DRAM
// )
void gemm_NEON_4x16_beta1( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C ) {
float *Ba = (float*) malloc(KC * 16 * sizeof(*Ba));
free(Ba);
float *Cb = (float*) malloc(16 * 4 * sizeof(*Cb));
free(Cb);
float32x4_t C_reg_0_0;
float32x4_t C_reg_1_0;
float32x4_t C_reg_2_0;
float32x4_t C_reg_3_0;
float32x4_t C_reg_4_0;
float32x4_t C_reg_5_0;
float32x4_t C_reg_6_0;
float32x4_t C_reg_7_0;
float32x4_t C_reg_8_0;
float32x4_t C_reg_9_0;
float32x4_t C_reg_10_0;
float32x4_t C_reg_11_0;
float32x4_t C_reg_12_0;
float32x4_t C_reg_13_0;
float32x4_t C_reg_14_0;
float32x4_t C_reg_15_0;
C_reg_0_0 = vld1q_f32(&C.data[0]);
C_reg_1_0 = vld1q_f32(&C.data[C.strides[0]]);
C_reg_2_0 = vld1q_f32(&C.data[(2) * (C.strides[0])]);
C_reg_3_0 = vld1q_f32(&C.data[(3) * (C.strides[0])]);
C_reg_4_0 = vld1q_f32(&C.data[(4) * (C.strides[0])]);
C_reg_5_0 = vld1q_f32(&C.data[(5) * (C.strides[0])]);
C_reg_6_0 = vld1q_f32(&C.data[(6) * (C.strides[0])]);
C_reg_7_0 = vld1q_f32(&C.data[(7) * (C.strides[0])]);
C_reg_8_0 = vld1q_f32(&C.data[(8) * (C.strides[0])]);
C_reg_9_0 = vld1q_f32(&C.data[(9) * (C.strides[0])]);
C_reg_10_0 = vld1q_f32(&C.data[(10) * (C.strides[0])]);
C_reg_11_0 = vld1q_f32(&C.data[(11) * (C.strides[0])]);
C_reg_12_0 = vld1q_f32(&C.data[(12) * (C.strides[0])]);
C_reg_13_0 = vld1q_f32(&C.data[(13) * (C.strides[0])]);
C_reg_14_0 = vld1q_f32(&C.data[(14) * (C.strides[0])]);
C_reg_15_0 = vld1q_f32(&C.data[(15) * (C.strides[0])]);
float32x4_t A_reg_0;
float32x4_t B_reg_0;
float32x4_t B_reg_1;
float32x4_t B_reg_2;
float32x4_t B_reg_3;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = vld1q_f32(&A[(k) * 4]);
  B_reg_0 = vld1q_f32(&B[(k) * (16)]);
  B_reg_1 = vld1q_f32(&B[(k) * (16) + 4]);
  B_reg_2 = vld1q_f32(&B[(k) * (16) + 8]);
  B_reg_3 = vld1q_f32(&B[(k) * (16) + 12]);
  C_reg_0_0 = vfmaq_laneq_f32(C_reg_0_0, A_reg_0, B_reg_0, (0));
  C_reg_1_0 = vfmaq_laneq_f32(C_reg_1_0, A_reg_0, B_reg_0, (1));
  C_reg_2_0 = vfmaq_laneq_f32(C_reg_2_0, A_reg_0, B_reg_0, (2));
  C_reg_3_0 = vfmaq_laneq_f32(C_reg_3_0, A_reg_0, B_reg_0, (3));
  C_reg_4_0 = vfmaq_laneq_f32(C_reg_4_0, A_reg_0, B_reg_1, (0));
  C_reg_5_0 = vfmaq_laneq_f32(C_reg_5_0, A_reg_0, B_reg_1, (1));
  C_reg_6_0 = vfmaq_laneq_f32(C_reg_6_0, A_reg_0, B_reg_1, (2));
  C_reg_7_0 = vfmaq_laneq_f32(C_reg_7_0, A_reg_0, B_reg_1, (3));
  C_reg_8_0 = vfmaq_laneq_f32(C_reg_8_0, A_reg_0, B_reg_2, (0));
  C_reg_9_0 = vfmaq_laneq_f32(C_reg_9_0, A_reg_0, B_reg_2, (1));
  C_reg_10_0 = vfmaq_laneq_f32(C_reg_10_0, A_reg_0, B_reg_2, (2));
  C_reg_11_0 = vfmaq_laneq_f32(C_reg_11_0, A_reg_0, B_reg_2, (3));
  C_reg_12_0 = vfmaq_laneq_f32(C_reg_12_0, A_reg_0, B_reg_3, (0));
  C_reg_13_0 = vfmaq_laneq_f32(C_reg_13_0, A_reg_0, B_reg_3, (1));
  C_reg_14_0 = vfmaq_laneq_f32(C_reg_14_0, A_reg_0, B_reg_3, (2));
  C_reg_15_0 = vfmaq_laneq_f32(C_reg_15_0, A_reg_0, B_reg_3, (3));
}
vst1q_f32(&C.data[0], C_reg_0_0);
vst1q_f32(&C.data[C.strides[0]], C_reg_1_0);
vst1q_f32(&C.data[(2) * (C.strides[0])], C_reg_2_0);
vst1q_f32(&C.data[(3) * (C.strides[0])], C_reg_3_0);
vst1q_f32(&C.data[(4) * (C.strides[0])], C_reg_4_0);
vst1q_f32(&C.data[(5) * (C.strides[0])], C_reg_5_0);
vst1q_f32(&C.data[(6) * (C.strides[0])], C_reg_6_0);
vst1q_f32(&C.data[(7) * (C.strides[0])], C_reg_7_0);
vst1q_f32(&C.data[(8) * (C.strides[0])], C_reg_8_0);
vst1q_f32(&C.data[(9) * (C.strides[0])], C_reg_9_0);
vst1q_f32(&C.data[(10) * (C.strides[0])], C_reg_10_0);
vst1q_f32(&C.data[(11) * (C.strides[0])], C_reg_11_0);
vst1q_f32(&C.data[(12) * (C.strides[0])], C_reg_12_0);
vst1q_f32(&C.data[(13) * (C.strides[0])], C_reg_13_0);
vst1q_f32(&C.data[(14) * (C.strides[0])], C_reg_14_0);
vst1q_f32(&C.data[(15) * (C.strides[0])], C_reg_15_0);
}

// gemm_NEON_4x24_beta0(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 4] @DRAM,
//     B : f32[KC, 24] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][24, 4] @DRAM
// )
void gemm_NEON_4x24_beta0( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C ) {
float32x4_t C_reg_0_0;
float32x4_t C_reg_1_0;
float32x4_t C_reg_2_0;
float32x4_t C_reg_3_0;
float32x4_t C_reg_4_0;
float32x4_t C_reg_5_0;
float32x4_t C_reg_6_0;
float32x4_t C_reg_7_0;
float32x4_t C_reg_8_0;
float32x4_t C_reg_9_0;
float32x4_t C_reg_10_0;
float32x4_t C_reg_11_0;
float32x4_t C_reg_12_0;
float32x4_t C_reg_13_0;
float32x4_t C_reg_14_0;
float32x4_t C_reg_15_0;
float32x4_t C_reg_16_0;
float32x4_t C_reg_17_0;
float32x4_t C_reg_18_0;
float32x4_t C_reg_19_0;
float32x4_t C_reg_20_0;
float32x4_t C_reg_21_0;
float32x4_t C_reg_22_0;
float32x4_t C_reg_23_0;
C_reg_0_0 = vmovq_n_f32(0.0f);
C_reg_1_0 = vmovq_n_f32(0.0f);
C_reg_2_0 = vmovq_n_f32(0.0f);
C_reg_3_0 = vmovq_n_f32(0.0f);
C_reg_4_0 = vmovq_n_f32(0.0f);
C_reg_5_0 = vmovq_n_f32(0.0f);
C_reg_6_0 = vmovq_n_f32(0.0f);
C_reg_7_0 = vmovq_n_f32(0.0f);
C_reg_8_0 = vmovq_n_f32(0.0f);
C_reg_9_0 = vmovq_n_f32(0.0f);
C_reg_10_0 = vmovq_n_f32(0.0f);
C_reg_11_0 = vmovq_n_f32(0.0f);
C_reg_12_0 = vmovq_n_f32(0.0f);
C_reg_13_0 = vmovq_n_f32(0.0f);
C_reg_14_0 = vmovq_n_f32(0.0f);
C_reg_15_0 = vmovq_n_f32(0.0f);
C_reg_16_0 = vmovq_n_f32(0.0f);
C_reg_17_0 = vmovq_n_f32(0.0f);
C_reg_18_0 = vmovq_n_f32(0.0f);
C_reg_19_0 = vmovq_n_f32(0.0f);
C_reg_20_0 = vmovq_n_f32(0.0f);
C_reg_21_0 = vmovq_n_f32(0.0f);
C_reg_22_0 = vmovq_n_f32(0.0f);
C_reg_23_0 = vmovq_n_f32(0.0f);
float32x4_t A_reg_0;
float32x4_t B_reg_0;
float32x4_t B_reg_1;
float32x4_t B_reg_2;
float32x4_t B_reg_3;
float32x4_t B_reg_4;
float32x4_t B_reg_5;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = vld1q_f32(&A[(k) * 4]);
  B_reg_0 = vld1q_f32(&B[(k) * (24)]);
  B_reg_1 = vld1q_f32(&B[(k) * (24) + 4]);
  B_reg_2 = vld1q_f32(&B[(k) * (24) + 8]);
  B_reg_3 = vld1q_f32(&B[(k) * (24) + 12]);
  B_reg_4 = vld1q_f32(&B[(k) * (24) + 16]);
  B_reg_5 = vld1q_f32(&B[(k) * (24) + 20]);
  C_reg_0_0 = vfmaq_laneq_f32(C_reg_0_0, A_reg_0, B_reg_0, (0));
  C_reg_1_0 = vfmaq_laneq_f32(C_reg_1_0, A_reg_0, B_reg_0, (1));
  C_reg_2_0 = vfmaq_laneq_f32(C_reg_2_0, A_reg_0, B_reg_0, (2));
  C_reg_3_0 = vfmaq_laneq_f32(C_reg_3_0, A_reg_0, B_reg_0, (3));
  C_reg_4_0 = vfmaq_laneq_f32(C_reg_4_0, A_reg_0, B_reg_1, (0));
  C_reg_5_0 = vfmaq_laneq_f32(C_reg_5_0, A_reg_0, B_reg_1, (1));
  C_reg_6_0 = vfmaq_laneq_f32(C_reg_6_0, A_reg_0, B_reg_1, (2));
  C_reg_7_0 = vfmaq_laneq_f32(C_reg_7_0, A_reg_0, B_reg_1, (3));
  C_reg_8_0 = vfmaq_laneq_f32(C_reg_8_0, A_reg_0, B_reg_2, (0));
  C_reg_9_0 = vfmaq_laneq_f32(C_reg_9_0, A_reg_0, B_reg_2, (1));
  C_reg_10_0 = vfmaq_laneq_f32(C_reg_10_0, A_reg_0, B_reg_2, (2));
  C_reg_11_0 = vfmaq_laneq_f32(C_reg_11_0, A_reg_0, B_reg_2, (3));
  C_reg_12_0 = vfmaq_laneq_f32(C_reg_12_0, A_reg_0, B_reg_3, (0));
  C_reg_13_0 = vfmaq_laneq_f32(C_reg_13_0, A_reg_0, B_reg_3, (1));
  C_reg_14_0 = vfmaq_laneq_f32(C_reg_14_0, A_reg_0, B_reg_3, (2));
  C_reg_15_0 = vfmaq_laneq_f32(C_reg_15_0, A_reg_0, B_reg_3, (3));
  C_reg_16_0 = vfmaq_laneq_f32(C_reg_16_0, A_reg_0, B_reg_4, (0));
  C_reg_17_0 = vfmaq_laneq_f32(C_reg_17_0, A_reg_0, B_reg_4, (1));
  C_reg_18_0 = vfmaq_laneq_f32(C_reg_18_0, A_reg_0, B_reg_4, (2));
  C_reg_19_0 = vfmaq_laneq_f32(C_reg_19_0, A_reg_0, B_reg_4, (3));
  C_reg_20_0 = vfmaq_laneq_f32(C_reg_20_0, A_reg_0, B_reg_5, (0));
  C_reg_21_0 = vfmaq_laneq_f32(C_reg_21_0, A_reg_0, B_reg_5, (1));
  C_reg_22_0 = vfmaq_laneq_f32(C_reg_22_0, A_reg_0, B_reg_5, (2));
  C_reg_23_0 = vfmaq_laneq_f32(C_reg_23_0, A_reg_0, B_reg_5, (3));
}
vst1q_f32(&C.data[0], C_reg_0_0);
vst1q_f32(&C.data[C.strides[0]], C_reg_1_0);
vst1q_f32(&C.data[(2) * (C.strides[0])], C_reg_2_0);
vst1q_f32(&C.data[(3) * (C.strides[0])], C_reg_3_0);
vst1q_f32(&C.data[(4) * (C.strides[0])], C_reg_4_0);
vst1q_f32(&C.data[(5) * (C.strides[0])], C_reg_5_0);
vst1q_f32(&C.data[(6) * (C.strides[0])], C_reg_6_0);
vst1q_f32(&C.data[(7) * (C.strides[0])], C_reg_7_0);
vst1q_f32(&C.data[(8) * (C.strides[0])], C_reg_8_0);
vst1q_f32(&C.data[(9) * (C.strides[0])], C_reg_9_0);
vst1q_f32(&C.data[(10) * (C.strides[0])], C_reg_10_0);
vst1q_f32(&C.data[(11) * (C.strides[0])], C_reg_11_0);
vst1q_f32(&C.data[(12) * (C.strides[0])], C_reg_12_0);
vst1q_f32(&C.data[(13) * (C.strides[0])], C_reg_13_0);
vst1q_f32(&C.data[(14) * (C.strides[0])], C_reg_14_0);
vst1q_f32(&C.data[(15) * (C.strides[0])], C_reg_15_0);
vst1q_f32(&C.data[(16) * (C.strides[0])], C_reg_16_0);
vst1q_f32(&C.data[(17) * (C.strides[0])], C_reg_17_0);
vst1q_f32(&C.data[(18) * (C.strides[0])], C_reg_18_0);
vst1q_f32(&C.data[(19) * (C.strides[0])], C_reg_19_0);
vst1q_f32(&C.data[(20) * (C.strides[0])], C_reg_20_0);
vst1q_f32(&C.data[(21) * (C.strides[0])], C_reg_21_0);
vst1q_f32(&C.data[(22) * (C.strides[0])], C_reg_22_0);
vst1q_f32(&C.data[(23) * (C.strides[0])], C_reg_23_0);
}

// gemm_NEON_4x24_beta1(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 4] @DRAM,
//     B : f32[KC, 24] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][24, 4] @DRAM
// )
void gemm_NEON_4x24_beta1( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C ) {
float *Ba = (float*) malloc(KC * 24 * sizeof(*Ba));
free(Ba);
float *Cb = (float*) malloc(24 * 4 * sizeof(*Cb));
free(Cb);
float32x4_t C_reg_0_0;
float32x4_t C_reg_1_0;
float32x4_t C_reg_2_0;
float32x4_t C_reg_3_0;
float32x4_t C_reg_4_0;
float32x4_t C_reg_5_0;
float32x4_t C_reg_6_0;
float32x4_t C_reg_7_0;
float32x4_t C_reg_8_0;
float32x4_t C_reg_9_0;
float32x4_t C_reg_10_0;
float32x4_t C_reg_11_0;
float32x4_t C_reg_12_0;
float32x4_t C_reg_13_0;
float32x4_t C_reg_14_0;
float32x4_t C_reg_15_0;
float32x4_t C_reg_16_0;
float32x4_t C_reg_17_0;
float32x4_t C_reg_18_0;
float32x4_t C_reg_19_0;
float32x4_t C_reg_20_0;
float32x4_t C_reg_21_0;
float32x4_t C_reg_22_0;
float32x4_t C_reg_23_0;
C_reg_0_0 = vld1q_f32(&C.data[0]);
C_reg_1_0 = vld1q_f32(&C.data[C.strides[0]]);
C_reg_2_0 = vld1q_f32(&C.data[(2) * (C.strides[0])]);
C_reg_3_0 = vld1q_f32(&C.data[(3) * (C.strides[0])]);
C_reg_4_0 = vld1q_f32(&C.data[(4) * (C.strides[0])]);
C_reg_5_0 = vld1q_f32(&C.data[(5) * (C.strides[0])]);
C_reg_6_0 = vld1q_f32(&C.data[(6) * (C.strides[0])]);
C_reg_7_0 = vld1q_f32(&C.data[(7) * (C.strides[0])]);
C_reg_8_0 = vld1q_f32(&C.data[(8) * (C.strides[0])]);
C_reg_9_0 = vld1q_f32(&C.data[(9) * (C.strides[0])]);
C_reg_10_0 = vld1q_f32(&C.data[(10) * (C.strides[0])]);
C_reg_11_0 = vld1q_f32(&C.data[(11) * (C.strides[0])]);
C_reg_12_0 = vld1q_f32(&C.data[(12) * (C.strides[0])]);
C_reg_13_0 = vld1q_f32(&C.data[(13) * (C.strides[0])]);
C_reg_14_0 = vld1q_f32(&C.data[(14) * (C.strides[0])]);
C_reg_15_0 = vld1q_f32(&C.data[(15) * (C.strides[0])]);
C_reg_16_0 = vld1q_f32(&C.data[(16) * (C.strides[0])]);
C_reg_17_0 = vld1q_f32(&C.data[(17) * (C.strides[0])]);
C_reg_18_0 = vld1q_f32(&C.data[(18) * (C.strides[0])]);
C_reg_19_0 = vld1q_f32(&C.data[(19) * (C.strides[0])]);
C_reg_20_0 = vld1q_f32(&C.data[(20) * (C.strides[0])]);
C_reg_21_0 = vld1q_f32(&C.data[(21) * (C.strides[0])]);
C_reg_22_0 = vld1q_f32(&C.data[(22) * (C.strides[0])]);
C_reg_23_0 = vld1q_f32(&C.data[(23) * (C.strides[0])]);
float32x4_t A_reg_0;
float32x4_t B_reg_0;
float32x4_t B_reg_1;
float32x4_t B_reg_2;
float32x4_t B_reg_3;
float32x4_t B_reg_4;
float32x4_t B_reg_5;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = vld1q_f32(&A[(k) * 4]);
  B_reg_0 = vld1q_f32(&B[(k) * (24)]);
  B_reg_1 = vld1q_f32(&B[(k) * (24) + 4]);
  B_reg_2 = vld1q_f32(&B[(k) * (24) + 8]);
  B_reg_3 = vld1q_f32(&B[(k) * (24) + 12]);
  B_reg_4 = vld1q_f32(&B[(k) * (24) + 16]);
  B_reg_5 = vld1q_f32(&B[(k) * (24) + 20]);
  C_reg_0_0 = vfmaq_laneq_f32(C_reg_0_0, A_reg_0, B_reg_0, (0));
  C_reg_1_0 = vfmaq_laneq_f32(C_reg_1_0, A_reg_0, B_reg_0, (1));
  C_reg_2_0 = vfmaq_laneq_f32(C_reg_2_0, A_reg_0, B_reg_0, (2));
  C_reg_3_0 = vfmaq_laneq_f32(C_reg_3_0, A_reg_0, B_reg_0, (3));
  C_reg_4_0 = vfmaq_laneq_f32(C_reg_4_0, A_reg_0, B_reg_1, (0));
  C_reg_5_0 = vfmaq_laneq_f32(C_reg_5_0, A_reg_0, B_reg_1, (1));
  C_reg_6_0 = vfmaq_laneq_f32(C_reg_6_0, A_reg_0, B_reg_1, (2));
  C_reg_7_0 = vfmaq_laneq_f32(C_reg_7_0, A_reg_0, B_reg_1, (3));
  C_reg_8_0 = vfmaq_laneq_f32(C_reg_8_0, A_reg_0, B_reg_2, (0));
  C_reg_9_0 = vfmaq_laneq_f32(C_reg_9_0, A_reg_0, B_reg_2, (1));
  C_reg_10_0 = vfmaq_laneq_f32(C_reg_10_0, A_reg_0, B_reg_2, (2));
  C_reg_11_0 = vfmaq_laneq_f32(C_reg_11_0, A_reg_0, B_reg_2, (3));
  C_reg_12_0 = vfmaq_laneq_f32(C_reg_12_0, A_reg_0, B_reg_3, (0));
  C_reg_13_0 = vfmaq_laneq_f32(C_reg_13_0, A_reg_0, B_reg_3, (1));
  C_reg_14_0 = vfmaq_laneq_f32(C_reg_14_0, A_reg_0, B_reg_3, (2));
  C_reg_15_0 = vfmaq_laneq_f32(C_reg_15_0, A_reg_0, B_reg_3, (3));
  C_reg_16_0 = vfmaq_laneq_f32(C_reg_16_0, A_reg_0, B_reg_4, (0));
  C_reg_17_0 = vfmaq_laneq_f32(C_reg_17_0, A_reg_0, B_reg_4, (1));
  C_reg_18_0 = vfmaq_laneq_f32(C_reg_18_0, A_reg_0, B_reg_4, (2));
  C_reg_19_0 = vfmaq_laneq_f32(C_reg_19_0, A_reg_0, B_reg_4, (3));
  C_reg_20_0 = vfmaq_laneq_f32(C_reg_20_0, A_reg_0, B_reg_5, (0));
  C_reg_21_0 = vfmaq_laneq_f32(C_reg_21_0, A_reg_0, B_reg_5, (1));
  C_reg_22_0 = vfmaq_laneq_f32(C_reg_22_0, A_reg_0, B_reg_5, (2));
  C_reg_23_0 = vfmaq_laneq_f32(C_reg_23_0, A_reg_0, B_reg_5, (3));
}
vst1q_f32(&C.data[0], C_reg_0_0);
vst1q_f32(&C.data[C.strides[0]], C_reg_1_0);
vst1q_f32(&C.data[(2) * (C.strides[0])], C_reg_2_0);
vst1q_f32(&C.data[(3) * (C.strides[0])], C_reg_3_0);
vst1q_f32(&C.data[(4) * (C.strides[0])], C_reg_4_0);
vst1q_f32(&C.data[(5) * (C.strides[0])], C_reg_5_0);
vst1q_f32(&C.data[(6) * (C.strides[0])], C_reg_6_0);
vst1q_f32(&C.data[(7) * (C.strides[0])], C_reg_7_0);
vst1q_f32(&C.data[(8) * (C.strides[0])], C_reg_8_0);
vst1q_f32(&C.data[(9) * (C.strides[0])], C_reg_9_0);
vst1q_f32(&C.data[(10) * (C.strides[0])], C_reg_10_0);
vst1q_f32(&C.data[(11) * (C.strides[0])], C_reg_11_0);
vst1q_f32(&C.data[(12) * (C.strides[0])], C_reg_12_0);
vst1q_f32(&C.data[(13) * (C.strides[0])], C_reg_13_0);
vst1q_f32(&C.data[(14) * (C.strides[0])], C_reg_14_0);
vst1q_f32(&C.data[(15) * (C.strides[0])], C_reg_15_0);
vst1q_f32(&C.data[(16) * (C.strides[0])], C_reg_16_0);
vst1q_f32(&C.data[(17) * (C.strides[0])], C_reg_17_0);
vst1q_f32(&C.data[(18) * (C.strides[0])], C_reg_18_0);
vst1q_f32(&C.data[(19) * (C.strides[0])], C_reg_19_0);
vst1q_f32(&C.data[(20) * (C.strides[0])], C_reg_20_0);
vst1q_f32(&C.data[(21) * (C.strides[0])], C_reg_21_0);
vst1q_f32(&C.data[(22) * (C.strides[0])], C_reg_22_0);
vst1q_f32(&C.data[(23) * (C.strides[0])], C_reg_23_0);
}

// gemm_NEON_4x4_beta0(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 4] @DRAM,
//     B : f32[KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][4, 4] @DRAM
// )
void gemm_NEON_4x4_beta0( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C ) {
float32x4_t C_reg_0_0;
float32x4_t C_reg_1_0;
float32x4_t C_reg_2_0;
float32x4_t C_reg_3_0;
C_reg_0_0 = vmovq_n_f32(0.0f);
C_reg_1_0 = vmovq_n_f32(0.0f);
C_reg_2_0 = vmovq_n_f32(0.0f);
C_reg_3_0 = vmovq_n_f32(0.0f);
float32x4_t A_reg_0;
float32x4_t B_reg_0;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = vld1q_f32(&A[(k) * 4]);
  B_reg_0 = vld1q_f32(&B[(k) * 4]);
  C_reg_0_0 = vfmaq_laneq_f32(C_reg_0_0, A_reg_0, B_reg_0, (0));
  C_reg_1_0 = vfmaq_laneq_f32(C_reg_1_0, A_reg_0, B_reg_0, (1));
  C_reg_2_0 = vfmaq_laneq_f32(C_reg_2_0, A_reg_0, B_reg_0, (2));
  C_reg_3_0 = vfmaq_laneq_f32(C_reg_3_0, A_reg_0, B_reg_0, (3));
}
vst1q_f32(&C.data[0], C_reg_0_0);
vst1q_f32(&C.data[C.strides[0]], C_reg_1_0);
vst1q_f32(&C.data[(2) * (C.strides[0])], C_reg_2_0);
vst1q_f32(&C.data[(3) * (C.strides[0])], C_reg_3_0);
}

// gemm_NEON_4x4_beta1(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 4] @DRAM,
//     B : f32[KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][4, 4] @DRAM
// )
void gemm_NEON_4x4_beta1( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C ) {
float *Ba = (float*) malloc(KC * 4 * sizeof(*Ba));
free(Ba);
float *Cb = (float*) malloc(4 * 4 * sizeof(*Cb));
free(Cb);
float32x4_t C_reg_0_0;
float32x4_t C_reg_1_0;
float32x4_t C_reg_2_0;
float32x4_t C_reg_3_0;
C_reg_0_0 = vld1q_f32(&C.data[0]);
C_reg_1_0 = vld1q_f32(&C.data[C.strides[0]]);
C_reg_2_0 = vld1q_f32(&C.data[(2) * (C.strides[0])]);
C_reg_3_0 = vld1q_f32(&C.data[(3) * (C.strides[0])]);
float32x4_t A_reg_0;
float32x4_t B_reg_0;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = vld1q_f32(&A[(k) * 4]);
  B_reg_0 = vld1q_f32(&B[(k) * 4]);
  C_reg_0_0 = vfmaq_laneq_f32(C_reg_0_0, A_reg_0, B_reg_0, (0));
  C_reg_1_0 = vfmaq_laneq_f32(C_reg_1_0, A_reg_0, B_reg_0, (1));
  C_reg_2_0 = vfmaq_laneq_f32(C_reg_2_0, A_reg_0, B_reg_0, (2));
  C_reg_3_0 = vfmaq_laneq_f32(C_reg_3_0, A_reg_0, B_reg_0, (3));
}
vst1q_f32(&C.data[0], C_reg_0_0);
vst1q_f32(&C.data[C.strides[0]], C_reg_1_0);
vst1q_f32(&C.data[(2) * (C.strides[0])], C_reg_2_0);
vst1q_f32(&C.data[(3) * (C.strides[0])], C_reg_3_0);
}

// gemm_NEON_4x8_beta0(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 4] @DRAM,
//     B : f32[KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][8, 4] @DRAM
// )
void gemm_NEON_4x8_beta0( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C ) {
float32x4_t C_reg_0_0;
float32x4_t C_reg_1_0;
float32x4_t C_reg_2_0;
float32x4_t C_reg_3_0;
float32x4_t C_reg_4_0;
float32x4_t C_reg_5_0;
float32x4_t C_reg_6_0;
float32x4_t C_reg_7_0;
C_reg_0_0 = vmovq_n_f32(0.0f);
C_reg_1_0 = vmovq_n_f32(0.0f);
C_reg_2_0 = vmovq_n_f32(0.0f);
C_reg_3_0 = vmovq_n_f32(0.0f);
C_reg_4_0 = vmovq_n_f32(0.0f);
C_reg_5_0 = vmovq_n_f32(0.0f);
C_reg_6_0 = vmovq_n_f32(0.0f);
C_reg_7_0 = vmovq_n_f32(0.0f);
float32x4_t A_reg_0;
float32x4_t B_reg_0;
float32x4_t B_reg_1;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = vld1q_f32(&A[(k) * 4]);
  B_reg_0 = vld1q_f32(&B[(k) * 8]);
  B_reg_1 = vld1q_f32(&B[(k) * 8 + 4]);
  C_reg_0_0 = vfmaq_laneq_f32(C_reg_0_0, A_reg_0, B_reg_0 (0));
  C_reg_1_0 = vfmaq_laneq_f32(C_reg_1_0, A_reg_0, B_reg_0, (1));
  C_reg_2_0 = vfmaq_laneq_f32(C_reg_2_0, A_reg_0, B_reg_0, (2));
  C_reg_3_0 = vfmaq_laneq_f32(C_reg_3_0, A_reg_0, B_reg_0, (3));
  C_reg_4_0 = vfmaq_laneq_f32(C_reg_4_0, A_reg_0, B_reg_0, (0));
  C_reg_5_0 = vfmaq_laneq_f32(C_reg_5_0, A_reg_0, B_reg_0, (1));
  C_reg_6_0 = vfmaq_laneq_f32(C_reg_6_0, A_reg_0, B_reg_0, (2));
  C_reg_7_0 = vfmaq_laneq_f32(C_reg_7_0, A_reg_0, B_reg_0, (3));
}
vst1q_f32(&C.data[0], C_reg_0_0);
vst1q_f32(&C.data[C.strides[0]], C_reg_1_0);
vst1q_f32(&C.data[(2) * (C.strides[0])], C_reg_2_0);
vst1q_f32(&C.data[(3) * (C.strides[0])], C_reg_3_0);
vst1q_f32(&C.data[(4) * (C.strides[0])], C_reg_4_0);
vst1q_f32(&C.data[(5) * (C.strides[0])], C_reg_5_0);
vst1q_f32(&C.data[(6) * (C.strides[0])], C_reg_6_0);
vst1q_f32(&C.data[(7) * (C.strides[0])], C_reg_7_0);
}

// gemm_NEON_4x8_beta1(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 4] @DRAM,
//     B : f32[KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][8, 4] @DRAM
// )
void gemm_NEON_4x8_beta1( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C ) {
float *Ba = (float*) malloc(KC * 8 * sizeof(*Ba));
free(Ba);
float *Cb = (float*) malloc(8 * 4 * sizeof(*Cb));
free(Cb);
float32x4_t C_reg_0_0;
float32x4_t C_reg_1_0;
float32x4_t C_reg_2_0;
float32x4_t C_reg_3_0;
float32x4_t C_reg_4_0;
float32x4_t C_reg_5_0;
float32x4_t C_reg_6_0;
float32x4_t C_reg_7_0;
C_reg_0_0 = vld1q_f32(&C.data[0]);
C_reg_1_0 = vld1q_f32(&C.data[C.strides[0]]);
C_reg_2_0 = vld1q_f32(&C.data[(2) * (C.strides[0])]);
C_reg_3_0 = vld1q_f32(&C.data[(3) * (C.strides[0])]);
C_reg_4_0 = vld1q_f32(&C.data[(4) * (C.strides[0])]);
C_reg_5_0 = vld1q_f32(&C.data[(5) * (C.strides[0])]);
C_reg_6_0 = vld1q_f32(&C.data[(6) * (C.strides[0])]);
C_reg_7_0 = vld1q_f32(&C.data[(7) * (C.strides[0])]);
float32x4_t A_reg_0;
float32x4_t B_reg_0;
float32x4_t B_reg_1;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = vld1q_f32(&A[(k) * 4]);
  B_reg_0 = vld1q_f32(&B[(k) * 8]);
  B_reg_1 = vld1q_f32(&B[(k) * 8 + 4]);
  C_reg_0_0 = vfmaq_laneq_f32(C_reg_0_0, A_reg_0, B_reg_0, (0));
  C_reg_1_0 = vfmaq_laneq_f32(C_reg_1_0, A_reg_0, B_reg_0, (1));
  C_reg_2_0 = vfmaq_laneq_f32(C_reg_2_0, A_reg_0, B_reg_0, (2));
  C_reg_3_0 = vfmaq_laneq_f32(C_reg_3_0, A_reg_0, B_reg_0, (3));
  C_reg_4_0 = vfmaq_laneq_f32(C_reg_4_0, A_reg_0, B_reg_1, (0));
  C_reg_5_0 = vfmaq_laneq_f32(C_reg_5_0, A_reg_0, B_reg_1, (1));
  C_reg_6_0 = vfmaq_laneq_f32(C_reg_6_0, A_reg_0, B_reg_1, (2));
  C_reg_7_0 = vfmaq_laneq_f32(C_reg_7_0, A_reg_0, B_reg_1, (3));
}
vst1q_f32(&C.data[0], C_reg_0_0);
vst1q_f32(&C.data[C.strides[0]], C_reg_1_0);
vst1q_f32(&C.data[(2) * (C.strides[0])], C_reg_2_0);
vst1q_f32(&C.data[(3) * (C.strides[0])], C_reg_3_0);
vst1q_f32(&C.data[(4) * (C.strides[0])], C_reg_4_0);
vst1q_f32(&C.data[(5) * (C.strides[0])], C_reg_5_0);
vst1q_f32(&C.data[(6) * (C.strides[0])], C_reg_6_0);
vst1q_f32(&C.data[(7) * (C.strides[0])], C_reg_7_0);
}

// gemm_NEON_8x12_beta0(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 8] @DRAM,
//     B : f32[KC, 12] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][12, 8] @DRAM
// )
void gemm_NEON_8x12_beta0( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C ) {
float32x4_t C_reg_0_0;
float32x4_t C_reg_0_1;
float32x4_t C_reg_1_0;
float32x4_t C_reg_1_1;
float32x4_t C_reg_2_0;
float32x4_t C_reg_2_1;
float32x4_t C_reg_3_0;
float32x4_t C_reg_3_1;
float32x4_t C_reg_4_0;
float32x4_t C_reg_4_1;
float32x4_t C_reg_5_0;
float32x4_t C_reg_5_1;
float32x4_t C_reg_6_0;
float32x4_t C_reg_6_1;
float32x4_t C_reg_7_0;
float32x4_t C_reg_7_1;
float32x4_t C_reg_8_0;
float32x4_t C_reg_8_1;
float32x4_t C_reg_9_0;
float32x4_t C_reg_9_1;
float32x4_t C_reg_10_0;
float32x4_t C_reg_10_1;
float32x4_t C_reg_11_0;
float32x4_t C_reg_11_1;
C_reg_0_0 = vmovq_n_f32(0.0f);
C_reg_0_1 = vmovq_n_f32(0.0f);
C_reg_1_0 = vmovq_n_f32(0.0f);
C_reg_1_1 = vmovq_n_f32(0.0f);
C_reg_2_0 = vmovq_n_f32(0.0f);
C_reg_2_1 = vmovq_n_f32(0.0f);
C_reg_3_0 = vmovq_n_f32(0.0f);
C_reg_3_1 = vmovq_n_f32(0.0f);
C_reg_4_0 = vmovq_n_f32(0.0f);
C_reg_4_1 = vmovq_n_f32(0.0f);
C_reg_5_0 = vmovq_n_f32(0.0f);
C_reg_5_1 = vmovq_n_f32(0.0f);
C_reg_6_0 = vmovq_n_f32(0.0f);
C_reg_6_1 = vmovq_n_f32(0.0f);
C_reg_7_0 = vmovq_n_f32(0.0f);
C_reg_7_1 = vmovq_n_f32(0.0f);
C_reg_8_0 = vmovq_n_f32(0.0f);
C_reg_8_1 = vmovq_n_f32(0.0f);
C_reg_9_0 = vmovq_n_f32(0.0f);
C_reg_9_1 = vmovq_n_f32(0.0f);
C_reg_10_0 = vmovq_n_f32(0.0f);
C_reg_10_1 = vmovq_n_f32(0.0f);
C_reg_11_0 = vmovq_n_f32(0.0f);
C_reg_11_1 = vmovq_n_f32(0.0f);
float32x4_t A_reg_0;
float32x4_t A_reg_1;
float32x4_t B_reg_0;
float32x4_t B_reg_1;
float32x4_t B_reg_2;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = vld1q_f32(&A[(k) * 8]);
  A_reg_1 = vld1q_f32(&A[(k) * 8 + 4]);
  B_reg_0 = vld1q_f32(&B[(k) * (12)]);
  B_reg_1 = vld1q_f32(&B[(k) * (12) + 4]);
  B_reg_2 = vld1q_f32(&B[(k) * (12) + 8]);
  C_reg_0_0 = vfmaq_laneq_f32(C_reg_0_0, A_reg_0, B_reg_0, (0));
  C_reg_1_0 = vfmaq_laneq_f32(C_reg_1_0, A_reg_0, B_reg_0, (1));
  C_reg_2_0 = vfmaq_laneq_f32(C_reg_2_0, A_reg_0, B_reg_0, (2));
  C_reg_3_0 = vfmaq_laneq_f32(C_reg_3_0, A_reg_0, B_reg_0, (3));
  C_reg_0_1 = vfmaq_laneq_f32(C_reg_0_1, A_reg_1, B_reg_0, (0));
  C_reg_1_1 = vfmaq_laneq_f32(C_reg_1_1, A_reg_1, B_reg_0, (1));
  C_reg_2_1 = vfmaq_laneq_f32(C_reg_2_1, A_reg_1, B_reg_0, (2));
  C_reg_3_1 = vfmaq_laneq_f32(C_reg_3_1, A_reg_1, B_reg_0, (3));
  C_reg_4_0 = vfmaq_laneq_f32(C_reg_4_0, A_reg_0, B_reg_1, (0));
  C_reg_5_0 = vfmaq_laneq_f32(C_reg_5_0, A_reg_0, B_reg_1, (1));
  C_reg_6_0 = vfmaq_laneq_f32(C_reg_6_0, A_reg_0, B_reg_1, (2));
  C_reg_7_0 = vfmaq_laneq_f32(C_reg_7_0, A_reg_0, B_reg_1, (3));
  C_reg_4_1 = vfmaq_laneq_f32(C_reg_4_1, A_reg_1, B_reg_1, (0));
  C_reg_5_1 = vfmaq_laneq_f32(C_reg_5_1, A_reg_1, B_reg_1, (1));
  C_reg_6_1 = vfmaq_laneq_f32(C_reg_6_1, A_reg_1, B_reg_1, (2));
  C_reg_7_1 = vfmaq_laneq_f32(C_reg_7_1, A_reg_1, B_reg_1, (3));
  C_reg_8_0 = vfmaq_laneq_f32(C_reg_8_0, A_reg_0, B_reg_2, (0));
  C_reg_9_0 = vfmaq_laneq_f32(C_reg_9_0, A_reg_0, B_reg_2, (1));
  C_reg_10_0 = vfmaq_laneq_f32(C_reg_10_0, A_reg_0, B_reg_2, (2));
  C_reg_11_0 = vfmaq_laneq_f32(C_reg_11_0, A_reg_0, B_reg_2, (3));
  C_reg_8_1 = vfmaq_laneq_f32(C_reg_8_1, A_reg_1, B_reg_2, (0));
  C_reg_9_1 = vfmaq_laneq_f32(C_reg_9_1, A_reg_1, B_reg_2, (1));
  C_reg_10_1 = vfmaq_laneq_f32(C_reg_10_1, A_reg_1, B_reg_2, (2));
  C_reg_11_1 = vfmaq_laneq_f32(C_reg_11_1, A_reg_1, B_reg_2, (3));
}
vst1q_f32(&C.data[0], C_reg_0_0);
vst1q_f32(&C.data[(4) * (C.strides[1])], C_reg_0_1);
vst1q_f32(&C.data[C.strides[0]], C_reg_1_0);
vst1q_f32(&C.data[C.strides[0] + (4) * (C.strides[1])], C_reg_1_1);
vst1q_f32(&C.data[(2) * (C.strides[0])], C_reg_2_0);
vst1q_f32(&C.data[(2) * (C.strides[0]) + (4) * (C.strides[1])], C_reg_2_1);
vst1q_f32(&C.data[(3) * (C.strides[0])], C_reg_3_0);
vst1q_f32(&C.data[(3) * (C.strides[0]) + (4) * (C.strides[1])], C_reg_3_1);
vst1q_f32(&C.data[(4) * (C.strides[0])], C_reg_4_0);
vst1q_f32(&C.data[(4) * (C.strides[0]) + (4) * (C.strides[1])], C_reg_4_1);
vst1q_f32(&C.data[(5) * (C.strides[0])], C_reg_5_0);
vst1q_f32(&C.data[(5) * (C.strides[0]) + (4) * (C.strides[1])], C_reg_5_1);
vst1q_f32(&C.data[(6) * (C.strides[0])], C_reg_6_0);
vst1q_f32(&C.data[(6) * (C.strides[0]) + (4) * (C.strides[1])], C_reg_6_1);
vst1q_f32(&C.data[(7) * (C.strides[0])], C_reg_7_0);
vst1q_f32(&C.data[(7) * (C.strides[0]) + (4) * (C.strides[1])], C_reg_7_1);
vst1q_f32(&C.data[(8) * (C.strides[0])], C_reg_8_0);
vst1q_f32(&C.data[(8) * (C.strides[0]) + (4) * (C.strides[1])], C_reg_8_1);
vst1q_f32(&C.data[(9) * (C.strides[0])], C_reg_9_0);
vst1q_f32(&C.data[(9) * (C.strides[0]) + (4) * (C.strides[1])], C_reg_9_1);
vst1q_f32(&C.data[(10) * (C.strides[0])], C_reg_10_0);
vst1q_f32(&C.data[(10) * (C.strides[0]) + (4) * (C.strides[1])], C_reg_10_1);
vst1q_f32(&C.data[(11) * (C.strides[0])], C_reg_11_0);
vst1q_f32(&C.data[(11) * (C.strides[0]) + (4) * (C.strides[1])], C_reg_11_1);
}

// gemm_NEON_8x12_beta1(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 8] @DRAM,
//     B : f32[KC, 12] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][12, 8] @DRAM
// )
void gemm_NEON_8x12_beta1( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C ) {
float *Ba = (float*) malloc(KC * 12 * sizeof(*Ba));
free(Ba);
float *Cb = (float*) malloc(12 * 8 * sizeof(*Cb));
free(Cb);
float32x4_t C_reg_0_0;
float32x4_t C_reg_0_1;
float32x4_t C_reg_1_0;
float32x4_t C_reg_1_1;
float32x4_t C_reg_2_0;
float32x4_t C_reg_2_1;
float32x4_t C_reg_3_0;
float32x4_t C_reg_3_1;
float32x4_t C_reg_4_0;
float32x4_t C_reg_4_1;
float32x4_t C_reg_5_0;
float32x4_t C_reg_5_1;
float32x4_t C_reg_6_0;
float32x4_t C_reg_6_1;
float32x4_t C_reg_7_0;
float32x4_t C_reg_7_1;
float32x4_t C_reg_8_0;
float32x4_t C_reg_8_1;
float32x4_t C_reg_9_0;
float32x4_t C_reg_9_1;
float32x4_t C_reg_10_0;
float32x4_t C_reg_10_1;
float32x4_t C_reg_11_0;
float32x4_t C_reg_11_1;
C_reg_0_0 = vld1q_f32(&C.data[0]);
C_reg_0_1 = vld1q_f32(&C.data[(4) * (C.strides[1])]);
C_reg_1_0 = vld1q_f32(&C.data[C.strides[0]]);
C_reg_1_1 = vld1q_f32(&C.data[C.strides[0] + (4) * (C.strides[1])]);
C_reg_2_0 = vld1q_f32(&C.data[(2) * (C.strides[0])]);
C_reg_2_1 = vld1q_f32(&C.data[(2) * (C.strides[0]) + (4) * (C.strides[1])]);
C_reg_3_0 = vld1q_f32(&C.data[(3) * (C.strides[0])]);
C_reg_3_1 = vld1q_f32(&C.data[(3) * (C.strides[0]) + (4) * (C.strides[1])]);
C_reg_4_0 = vld1q_f32(&C.data[(4) * (C.strides[0])]);
C_reg_4_1 = vld1q_f32(&C.data[(4) * (C.strides[0]) + (4) * (C.strides[1])]);
C_reg_5_0 = vld1q_f32(&C.data[(5) * (C.strides[0])]);
C_reg_5_1 = vld1q_f32(&C.data[(5) * (C.strides[0]) + (4) * (C.strides[1])]);
C_reg_6_0 = vld1q_f32(&C.data[(6) * (C.strides[0])]);
C_reg_6_1 = vld1q_f32(&C.data[(6) * (C.strides[0]) + (4) * (C.strides[1])]);
C_reg_7_0 = vld1q_f32(&C.data[(7) * (C.strides[0])]);
C_reg_7_1 = vld1q_f32(&C.data[(7) * (C.strides[0]) + (4) * (C.strides[1])]);
C_reg_8_0 = vld1q_f32(&C.data[(8) * (C.strides[0])]);
C_reg_8_1 = vld1q_f32(&C.data[(8) * (C.strides[0]) + (4) * (C.strides[1])]);
C_reg_9_0 = vld1q_f32(&C.data[(9) * (C.strides[0])]);
C_reg_9_1 = vld1q_f32(&C.data[(9) * (C.strides[0]) + (4) * (C.strides[1])]);
C_reg_10_0 = vld1q_f32(&C.data[(10) * (C.strides[0])]);
C_reg_10_1 = vld1q_f32(&C.data[(10) * (C.strides[0]) + (4) * (C.strides[1])]);
C_reg_11_0 = vld1q_f32(&C.data[(11) * (C.strides[0])]);
C_reg_11_1 = vld1q_f32(&C.data[(11) * (C.strides[0]) + (4) * (C.strides[1])]);
float32x4_t A_reg_0;
float32x4_t A_reg_1;
float32x4_t B_reg_0;
float32x4_t B_reg_1;
float32x4_t B_reg_2;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = vld1q_f32(&A[(k) * 8]);
  A_reg_1 = vld1q_f32(&A[(k) * 8 + 4]);
  B_reg_0 = vld1q_f32(&B[(k) * (12)]);
  B_reg_1 = vld1q_f32(&B[(k) * (12) + 4]);
  B_reg_2 = vld1q_f32(&B[(k) * (12) + 8]);
  C_reg_0_0 = vfmaq_laneq_f32(C_reg_0_0, A_reg_0, B_reg_0, (0));
  C_reg_1_0 = vfmaq_laneq_f32(C_reg_1_0, A_reg_0, B_reg_0, (1));
  C_reg_2_0 = vfmaq_laneq_f32(C_reg_2_0, A_reg_0, B_reg_0, (2));
  C_reg_3_0 = vfmaq_laneq_f32(C_reg_3_0, A_reg_0, B_reg_0, (3));
  C_reg_0_1 = vfmaq_laneq_f32(C_reg_0_1, A_reg_1, B_reg_0, (0));
  C_reg_1_1 = vfmaq_laneq_f32(C_reg_1_1, A_reg_1, B_reg_0, (1));
  C_reg_2_1 = vfmaq_laneq_f32(C_reg_2_1, A_reg_1, B_reg_0, (2));
  C_reg_3_1 = vfmaq_laneq_f32(C_reg_3_1, A_reg_1, B_reg_0, (3));
  C_reg_4_0 = vfmaq_laneq_f32(C_reg_4_0, A_reg_0, B_reg_1, (0));
  C_reg_5_0 = vfmaq_laneq_f32(C_reg_5_0, A_reg_0, B_reg_1, (1));
  C_reg_6_0 = vfmaq_laneq_f32(C_reg_6_0, A_reg_0, B_reg_1, (2));
  C_reg_7_0 = vfmaq_laneq_f32(C_reg_7_0, A_reg_0, B_reg_1, (3));
  C_reg_4_1 = vfmaq_laneq_f32(C_reg_4_1, A_reg_1, B_reg_1, (0));
  C_reg_5_1 = vfmaq_laneq_f32(C_reg_5_1, A_reg_1, B_reg_1, (1));
  C_reg_6_1 = vfmaq_laneq_f32(C_reg_6_1, A_reg_1, B_reg_1, (2));
  C_reg_7_1 = vfmaq_laneq_f32(C_reg_7_1, A_reg_1, B_reg_1, (3));
  C_reg_8_0 = vfmaq_laneq_f32(C_reg_8_0, A_reg_0, B_reg_2, (0));
  C_reg_9_0 = vfmaq_laneq_f32(C_reg_9_0, A_reg_0, B_reg_2, (1));
  C_reg_10_0 = vfmaq_laneq_f32(C_reg_10_0, A_reg_0, B_reg_2, (2));
  C_reg_11_0 = vfmaq_laneq_f32(C_reg_11_0, A_reg_0, B_reg_2, (3));
  C_reg_8_1 = vfmaq_laneq_f32(C_reg_8_1, A_reg_1, B_reg_2, (0));
  C_reg_9_1 = vfmaq_laneq_f32(C_reg_9_1, A_reg_1, B_reg_2, (1));
  C_reg_10_1 = vfmaq_laneq_f32(C_reg_10_1, A_reg_1, B_reg_2, (2));
  C_reg_11_1 = vfmaq_laneq_f32(C_reg_11_1, A_reg_1, B_reg_2, (3));
}
vst1q_f32(&C.data[0], C_reg_0_0);
vst1q_f32(&C.data[(4) * (C.strides[1])], C_reg_0_1);
vst1q_f32(&C.data[C.strides[0]], C_reg_1_0);
vst1q_f32(&C.data[C.strides[0] + (4) * (C.strides[1])], C_reg_1_1);
vst1q_f32(&C.data[(2) * (C.strides[0])], C_reg_2_0);
vst1q_f32(&C.data[(2) * (C.strides[0]) + (4) * (C.strides[1])], C_reg_2_1);
vst1q_f32(&C.data[(3) * (C.strides[0])], C_reg_3_0);
vst1q_f32(&C.data[(3) * (C.strides[0]) + (4) * (C.strides[1])], C_reg_3_1);
vst1q_f32(&C.data[(4) * (C.strides[0])], C_reg_4_0);
vst1q_f32(&C.data[(4) * (C.strides[0]) + (4) * (C.strides[1])], C_reg_4_1);
vst1q_f32(&C.data[(5) * (C.strides[0])], C_reg_5_0);
vst1q_f32(&C.data[(5) * (C.strides[0]) + (4) * (C.strides[1])], C_reg_5_1);
vst1q_f32(&C.data[(6) * (C.strides[0])], C_reg_6_0);
vst1q_f32(&C.data[(6) * (C.strides[0]) + (4) * (C.strides[1])], C_reg_6_1);
vst1q_f32(&C.data[(7) * (C.strides[0])], C_reg_7_0);
vst1q_f32(&C.data[(7) * (C.strides[0]) + (4) * (C.strides[1])], C_reg_7_1);
vst1q_f32(&C.data[(8) * (C.strides[0])], C_reg_8_0);
vst1q_f32(&C.data[(8) * (C.strides[0]) + (4) * (C.strides[1])], C_reg_8_1);
vst1q_f32(&C.data[(9) * (C.strides[0])], C_reg_9_0);
vst1q_f32(&C.data[(9) * (C.strides[0]) + (4) * (C.strides[1])], C_reg_9_1);
vst1q_f32(&C.data[(10) * (C.strides[0])], C_reg_10_0);
vst1q_f32(&C.data[(10) * (C.strides[0]) + (4) * (C.strides[1])], C_reg_10_1);
vst1q_f32(&C.data[(11) * (C.strides[0])], C_reg_11_0);
vst1q_f32(&C.data[(11) * (C.strides[0]) + (4) * (C.strides[1])], C_reg_11_1);
}

// gemm_NEON_8x4_beta0(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 8] @DRAM,
//     B : f32[KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][4, 8] @DRAM
// )
void gemm_NEON_8x4_beta0( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C ) {
float32x4_t C_reg_0_0;
float32x4_t C_reg_0_1;
float32x4_t C_reg_1_0;
float32x4_t C_reg_1_1;
float32x4_t C_reg_2_0;
float32x4_t C_reg_2_1;
float32x4_t C_reg_3_0;
float32x4_t C_reg_3_1;
C_reg_0_0 = vmovq_n_f32(0.0f);
C_reg_0_1 = vmovq_n_f32(0.0f);
C_reg_1_0 = vmovq_n_f32(0.0f);
C_reg_1_1 = vmovq_n_f32(0.0f);
C_reg_2_0 = vmovq_n_f32(0.0f);
C_reg_2_1 = vmovq_n_f32(0.0f);
C_reg_3_0 = vmovq_n_f32(0.0f);
C_reg_3_1 = vmovq_n_f32(0.0f);
float32x4_t A_reg_0;
float32x4_t A_reg_1;
float32x4_t B_reg_0;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = vld1q_f32(&A[(k) * 8]);
  A_reg_1 = vld1q_f32(&A[(k) * 8 + 4]);
  B_reg_0 = vld1q_f32(&B[(k) * 4]);
  C_reg_0_0 = vfmaq_laneq_f32(C_reg_0_0, A_reg_0, B_reg_0, (0));
  C_reg_1_0 = vfmaq_laneq_f32(C_reg_1_0, A_reg_0, B_reg_0, (1));
  C_reg_2_0 = vfmaq_laneq_f32(C_reg_2_0, A_reg_0, B_reg_0, (2));
  C_reg_3_0 = vfmaq_laneq_f32(C_reg_3_0, A_reg_0, B_reg_0, (3));
  C_reg_0_1 = vfmaq_laneq_f32(C_reg_0_1, A_reg_1, B_reg_0, (0));
  C_reg_1_1 = vfmaq_laneq_f32(C_reg_1_1, A_reg_1, B_reg_0, (1));
  C_reg_2_1 = vfmaq_laneq_f32(C_reg_2_1, A_reg_1, B_reg_0, (2));
  C_reg_3_1 = vfmaq_laneq_f32(C_reg_3_1, A_reg_1, B_reg_0, (3));
}
vst1q_f32(&C.data[0], C_reg_0_0);
vst1q_f32(&C.data[(4) * (C.strides[1])], C_reg_0_1);
vst1q_f32(&C.data[C.strides[0]], C_reg_1_0);
vst1q_f32(&C.data[C.strides[0] + (4) * (C.strides[1])], C_reg_1_1);
vst1q_f32(&C.data[(2) * (C.strides[0])], C_reg_2_0);
vst1q_f32(&C.data[(2) * (C.strides[0]) + (4) * (C.strides[1])], C_reg_2_1);
vst1q_f32(&C.data[(3) * (C.strides[0])], C_reg_3_0);
vst1q_f32(&C.data[(3) * (C.strides[0]) + (4) * (C.strides[1])], C_reg_3_1);
}

// gemm_NEON_8x4_beta1(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 8] @DRAM,
//     B : f32[KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][4, 8] @DRAM
// )
void gemm_NEON_8x4_beta1( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C ) {
float *Ba = (float*) malloc(KC * 4 * sizeof(*Ba));
free(Ba);
float *Cb = (float*) malloc(4 * 8 * sizeof(*Cb));
free(Cb);
float32x4_t C_reg_0_0;
float32x4_t C_reg_0_1;
float32x4_t C_reg_1_0;
float32x4_t C_reg_1_1;
float32x4_t C_reg_2_0;
float32x4_t C_reg_2_1;
float32x4_t C_reg_3_0;
float32x4_t C_reg_3_1;
C_reg_0_0 = vld1q_f32(&C.data[0]);
C_reg_0_1 = vld1q_f32(&C.data[(4) * (C.strides[1])]);
C_reg_1_0 = vld1q_f32(&C.data[C.strides[0]]);
C_reg_1_1 = vld1q_f32(&C.data[C.strides[0] + (4) * (C.strides[1])]);
C_reg_2_0 = vld1q_f32(&C.data[(2) * (C.strides[0])]);
C_reg_2_1 = vld1q_f32(&C.data[(2) * (C.strides[0]) + (4) * (C.strides[1])]);
C_reg_3_0 = vld1q_f32(&C.data[(3) * (C.strides[0])]);
C_reg_3_1 = vld1q_f32(&C.data[(3) * (C.strides[0]) + (4) * (C.strides[1])]);
float32x4_t A_reg_0;
float32x4_t A_reg_1;
float32x4_t B_reg_0;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = vld1q_f32(&A[(k) * 8]);
  A_reg_1 = vld1q_f32(&A[(k) * 8 + 4]);
  B_reg_0 = vld1q_f32(&B[(k) * 4]);
  C_reg_0_0 = vfmaq_laneq_f32(C_reg_0_0, A_reg_0, B_reg_0, (0));
  C_reg_1_0 = vfmaq_laneq_f32(C_reg_1_0, A_reg_0, B_reg_0, (1));
  C_reg_2_0 = vfmaq_laneq_f32(C_reg_2_0, A_reg_0, B_reg_0, (2));
  C_reg_3_0 = vfmaq_laneq_f32(C_reg_3_0, A_reg_0, B_reg_0, (3));
  C_reg_0_1 = vfmaq_laneq_f32(C_reg_0_1, A_reg_1, B_reg_0, (0));
  C_reg_1_1 = vfmaq_laneq_f32(C_reg_1_1, A_reg_1, B_reg_0, (1));
  C_reg_2_1 = vfmaq_laneq_f32(C_reg_2_1, A_reg_1, B_reg_0, (2));
  C_reg_3_1 = vfmaq_laneq_f32(C_reg_3_1, A_reg_1, B_reg_0, (3));
}
vst1q_f32(&C.data[0], C_reg_0_0);
vst1q_f32(&C.data[(4) * (C.strides[1])], C_reg_0_1);
vst1q_f32(&C.data[C.strides[0]], C_reg_1_0);
vst1q_f32(&C.data[C.strides[0] + (4) * (C.strides[1])], C_reg_1_1);
vst1q_f32(&C.data[(2) * (C.strides[0])], C_reg_2_0);
vst1q_f32(&C.data[(2) * (C.strides[0]) + (4) * (C.strides[1])], C_reg_2_1);
vst1q_f32(&C.data[(3) * (C.strides[0])], C_reg_3_0);
vst1q_f32(&C.data[(3) * (C.strides[0]) + (4) * (C.strides[1])], C_reg_3_1);
}

// gemm_NEON_8x8_beta0(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 8] @DRAM,
//     B : f32[KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][8, 8] @DRAM
// )
void gemm_NEON_8x8_beta0( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C ) {
float32x4_t C_reg_0_0;
float32x4_t C_reg_0_1;
float32x4_t C_reg_1_0;
float32x4_t C_reg_1_1;
float32x4_t C_reg_2_0;
float32x4_t C_reg_2_1;
float32x4_t C_reg_3_0;
float32x4_t C_reg_3_1;
float32x4_t C_reg_4_0;
float32x4_t C_reg_4_1;
float32x4_t C_reg_5_0;
float32x4_t C_reg_5_1;
float32x4_t C_reg_6_0;
float32x4_t C_reg_6_1;
float32x4_t C_reg_7_0;
float32x4_t C_reg_7_1;
C_reg_0_0 = vmovq_n_f32(0.0f);
C_reg_0_1 = vmovq_n_f32(0.0f);
C_reg_1_0 = vmovq_n_f32(0.0f);
C_reg_1_1 = vmovq_n_f32(0.0f);
C_reg_2_0 = vmovq_n_f32(0.0f);
C_reg_2_1 = vmovq_n_f32(0.0f);
C_reg_3_0 = vmovq_n_f32(0.0f);
C_reg_3_1 = vmovq_n_f32(0.0f);
C_reg_4_0 = vmovq_n_f32(0.0f);
C_reg_4_1 = vmovq_n_f32(0.0f);
C_reg_5_0 = vmovq_n_f32(0.0f);
C_reg_5_1 = vmovq_n_f32(0.0f);
C_reg_6_0 = vmovq_n_f32(0.0f);
C_reg_6_1 = vmovq_n_f32(0.0f);
C_reg_7_0 = vmovq_n_f32(0.0f);
C_reg_7_1 = vmovq_n_f32(0.0f);
float32x4_t A_reg_0;
float32x4_t A_reg_1;
float32x4_t B_reg_0;
float32x4_t B_reg_1;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = vld1q_f32(&A[(k) * 8]);
  A_reg_1 = vld1q_f32(&A[(k) * 8 + 4]);
  B_reg_0 = vld1q_f32(&B[(k) * 8]);
  B_reg_1 = vld1q_f32(&B[(k) * 8 + 4]);
  C_reg_0_0 = vfmaq_laneq_f32(C_reg_0_0, A_reg_0, B_reg_0, (0));
  C_reg_1_0 = vfmaq_laneq_f32(C_reg_1_0, A_reg_0, B_reg_0, (1));
  C_reg_2_0 = vfmaq_laneq_f32(C_reg_2_0, A_reg_0, B_reg_0, (2));
  C_reg_3_0 = vfmaq_laneq_f32(C_reg_3_0, A_reg_0, B_reg_0, (3));
  C_reg_0_1 = vfmaq_laneq_f32(C_reg_0_1, A_reg_1, B_reg_0, (0));
  C_reg_1_1 = vfmaq_laneq_f32(C_reg_1_1, A_reg_1, B_reg_0, (1));
  C_reg_2_1 = vfmaq_laneq_f32(C_reg_2_1, A_reg_1, B_reg_0, (2));
  C_reg_3_1 = vfmaq_laneq_f32(C_reg_3_1, A_reg_1, B_reg_0, (3));
  C_reg_4_0 = vfmaq_laneq_f32(C_reg_4_0, A_reg_0, B_reg_1, (0));
  C_reg_5_0 = vfmaq_laneq_f32(C_reg_5_0, A_reg_0, B_reg_1, (1));
  C_reg_6_0 = vfmaq_laneq_f32(C_reg_6_0, A_reg_0, B_reg_1, (2));
  C_reg_7_0 = vfmaq_laneq_f32(C_reg_7_0, A_reg_0, B_reg_1, (3));
  C_reg_4_1 = vfmaq_laneq_f32(C_reg_4_1, A_reg_1, B_reg_1, (0));
  C_reg_5_1 = vfmaq_laneq_f32(C_reg_5_1, A_reg_1, B_reg_1, (1));
  C_reg_6_1 = vfmaq_laneq_f32(C_reg_6_1, A_reg_1, B_reg_1, (2));
  C_reg_7_1 = vfmaq_laneq_f32(C_reg_7_1, A_reg_1, B_reg_1, (3));
}
vst1q_f32(&C.data[0], C_reg_0_0);
vst1q_f32(&C.data[(4) * (C.strides[1])], C_reg_0_1);
vst1q_f32(&C.data[C.strides[0]], C_reg_1_0);
vst1q_f32(&C.data[C.strides[0] + (4) * (C.strides[1])], C_reg_1_1);
vst1q_f32(&C.data[(2) * (C.strides[0])], C_reg_2_0);
vst1q_f32(&C.data[(2) * (C.strides[0]) + (4) * (C.strides[1])], C_reg_2_1);
vst1q_f32(&C.data[(3) * (C.strides[0])], C_reg_3_0);
vst1q_f32(&C.data[(3) * (C.strides[0]) + (4) * (C.strides[1])], C_reg_3_1);
vst1q_f32(&C.data[(4) * (C.strides[0])], C_reg_4_0);
vst1q_f32(&C.data[(4) * (C.strides[0]) + (4) * (C.strides[1])], C_reg_4_1);
vst1q_f32(&C.data[(5) * (C.strides[0])], C_reg_5_0);
vst1q_f32(&C.data[(5) * (C.strides[0]) + (4) * (C.strides[1])], C_reg_5_1);
vst1q_f32(&C.data[(6) * (C.strides[0])], C_reg_6_0);
vst1q_f32(&C.data[(6) * (C.strides[0]) + (4) * (C.strides[1])], C_reg_6_1);
vst1q_f32(&C.data[(7) * (C.strides[0])], C_reg_7_0);
vst1q_f32(&C.data[(7) * (C.strides[0]) + (4) * (C.strides[1])], C_reg_7_1);
}

// gemm_NEON_8x8_beta1(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 8] @DRAM,
//     B : f32[KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][8, 8] @DRAM
// )
void gemm_NEON_8x8_beta1( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, struct exo_win_2f32 C ) {
float *Ba = (float*) malloc(KC * 8 * sizeof(*Ba));
free(Ba);
float *Cb = (float*) malloc(8 * 8 * sizeof(*Cb));
free(Cb);
float32x4_t C_reg_0_0;
float32x4_t C_reg_0_1;
float32x4_t C_reg_1_0;
float32x4_t C_reg_1_1;
float32x4_t C_reg_2_0;
float32x4_t C_reg_2_1;
float32x4_t C_reg_3_0;
float32x4_t C_reg_3_1;
float32x4_t C_reg_4_0;
float32x4_t C_reg_4_1;
float32x4_t C_reg_5_0;
float32x4_t C_reg_5_1;
float32x4_t C_reg_6_0;
float32x4_t C_reg_6_1;
float32x4_t C_reg_7_0;
float32x4_t C_reg_7_1;
C_reg_0_0 = vld1q_f32(&C.data[0]);
C_reg_0_1 = vld1q_f32(&C.data[(4) * (C.strides[1])]);
C_reg_1_0 = vld1q_f32(&C.data[C.strides[0]]);
C_reg_1_1 = vld1q_f32(&C.data[C.strides[0] + (4) * (C.strides[1])]);
C_reg_2_0 = vld1q_f32(&C.data[(2) * (C.strides[0])]);
C_reg_2_1 = vld1q_f32(&C.data[(2) * (C.strides[0]) + (4) * (C.strides[1])]);
C_reg_3_0 = vld1q_f32(&C.data[(3) * (C.strides[0])]);
C_reg_3_1 = vld1q_f32(&C.data[(3) * (C.strides[0]) + (4) * (C.strides[1])]);
C_reg_4_0 = vld1q_f32(&C.data[(4) * (C.strides[0])]);
C_reg_4_1 = vld1q_f32(&C.data[(4) * (C.strides[0]) + (4) * (C.strides[1])]);
C_reg_5_0 = vld1q_f32(&C.data[(5) * (C.strides[0])]);
C_reg_5_1 = vld1q_f32(&C.data[(5) * (C.strides[0]) + (4) * (C.strides[1])]);
C_reg_6_0 = vld1q_f32(&C.data[(6) * (C.strides[0])]);
C_reg_6_1 = vld1q_f32(&C.data[(6) * (C.strides[0]) + (4) * (C.strides[1])]);
C_reg_7_0 = vld1q_f32(&C.data[(7) * (C.strides[0])]);
C_reg_7_1 = vld1q_f32(&C.data[(7) * (C.strides[0]) + (4) * (C.strides[1])]);
float32x4_t A_reg_0;
float32x4_t A_reg_1;
float32x4_t B_reg_0;
float32x4_t B_reg_1;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = vld1q_f32(&A[(k) * 8]);
  A_reg_1 = vld1q_f32(&A[(k) * 8 + 4]);
  B_reg_0 = vld1q_f32(&B[(k) * 8]);
  B_reg_1 = vld1q_f32(&B[(k) * 8 + 4]);
  C_reg_0_0 = vfmaq_laneq_f32(C_reg_0_0, A_reg_0, B_reg_0, (0));
  C_reg_1_0 = vfmaq_laneq_f32(C_reg_1_0, A_reg_0, B_reg_0, (1));
  C_reg_2_0 = vfmaq_laneq_f32(C_reg_2_0, A_reg_0, B_reg_0, (2));
  C_reg_3_0 = vfmaq_laneq_f32(C_reg_3_0, A_reg_0, B_reg_0, (3));
  C_reg_0_1 = vfmaq_laneq_f32(C_reg_0_1, A_reg_1, B_reg_0, (0));
  C_reg_1_1 = vfmaq_laneq_f32(C_reg_1_1, A_reg_1, B_reg_0, (1));
  C_reg_2_1 = vfmaq_laneq_f32(C_reg_2_1, A_reg_1, B_reg_0, (2));
  C_reg_3_1 = vfmaq_laneq_f32(C_reg_3_1, A_reg_1, B_reg_0, (3));
  C_reg_4_0 = vfmaq_laneq_f32(C_reg_4_0, A_reg_0, B_reg_1, (0));
  C_reg_5_0 = vfmaq_laneq_f32(C_reg_5_0, A_reg_0, B_reg_1, (1));
  C_reg_6_0 = vfmaq_laneq_f32(C_reg_6_0, A_reg_0, B_reg_1, (2));
  C_reg_7_0 = vfmaq_laneq_f32(C_reg_7_0, A_reg_0, B_reg_1, (3));
  C_reg_4_1 = vfmaq_laneq_f32(C_reg_4_1, A_reg_1, B_reg_1, (0));
  C_reg_5_1 = vfmaq_laneq_f32(C_reg_5_1, A_reg_1, B_reg_1, (1));
  C_reg_6_1 = vfmaq_laneq_f32(C_reg_6_1, A_reg_1, B_reg_1, (2));
  C_reg_7_1 = vfmaq_laneq_f32(C_reg_7_1, A_reg_1, B_reg_1, (3));
}
vst1q_f32(&C.data[0], C_reg_0_0);
vst1q_f32(&C.data[(4) * (C.strides[1])], C_reg_0_1);
vst1q_f32(&C.data[C.strides[0]], C_reg_1_0);
vst1q_f32(&C.data[C.strides[0] + (4) * (C.strides[1])], C_reg_1_1);
vst1q_f32(&C.data[(2) * (C.strides[0])], C_reg_2_0);
vst1q_f32(&C.data[(2) * (C.strides[0]) + (4) * (C.strides[1])], C_reg_2_1);
vst1q_f32(&C.data[(3) * (C.strides[0])], C_reg_3_0);
vst1q_f32(&C.data[(3) * (C.strides[0]) + (4) * (C.strides[1])], C_reg_3_1);
vst1q_f32(&C.data[(4) * (C.strides[0])], C_reg_4_0);
vst1q_f32(&C.data[(4) * (C.strides[0]) + (4) * (C.strides[1])], C_reg_4_1);
vst1q_f32(&C.data[(5) * (C.strides[0])], C_reg_5_0);
vst1q_f32(&C.data[(5) * (C.strides[0]) + (4) * (C.strides[1])], C_reg_5_1);
vst1q_f32(&C.data[(6) * (C.strides[0])], C_reg_6_0);
vst1q_f32(&C.data[(6) * (C.strides[0]) + (4) * (C.strides[1])], C_reg_6_1);
vst1q_f32(&C.data[(7) * (C.strides[0])], C_reg_7_0);
vst1q_f32(&C.data[(7) * (C.strides[0]) + (4) * (C.strides[1])], C_reg_7_1);
}


/* relying on the following instruction..."
neon_vfmladrian_4xf32_4xf32(dst,lhs,rhs,s)
{dst_data} = vfmaq_laneq_f32({dst_data}, {lhs_data}, {rhs_data}, {s});
*/

/* relying on the following instruction..."
neon_vld_4xf32(dst,src)
{dst_data} = vld1q_f32(&{src_data});
*/

/* relying on the following instruction..."
neon_vst_4xf32(dst,src)
vst1q_f32(&{dst_data}, {src_data});
*/

/* relying on the following instruction..."
neon_zero_4xf32(dst)
{dst_data} = vmovq_n_f32(0.0f);
*/
