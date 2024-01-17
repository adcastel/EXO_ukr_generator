from __future__ import annotations
from exo import proc
from exo.platforms.neon import *
from exo.stdlib.scheduling import *

#@proc
def ukernel_get_ref(
        MR: size,
        NR: size,
        alpha1: bool,
        beta1: bool,
        
        ):
    def ukernel_beta1(
            KC: size,
            alpha: f32[1],
            A: f32[KC, MR] @ DRAM,
            B: f32[KC, NR] @ DRAM,
            beta: f32[1],
            C: f32[NR, MR] @ DRAM,
            ):
        
        Ba: f32[KC,NR] @ DRAM
        Cb: f32[NR,MR] @ DRAM
        
        if beta1 == False:
            for cj in seq(0, NR):
                for ci in seq(0, MR):
                    Cb[cj,ci] = C[cj,ci] * beta[0]
        
        if alpha1 == False:
            for bk in seq(0, KC):
                for bj in seq(0, NR):
                    B[bk,bj] = B[bk,bj] * alpha[0]
                    
        for k in seq(0, KC):
            for j in seq(0, NR):
                for i in seq(0, MR):
                    if beta1 == False:
                        Cb[j, i] += A[k,i] * B[k,j]
                    else:
                        C[j, i] += A[k,i] * B[k,j]
        
        if beta1 == False:
            for cj in seq(0, NR):
                for ci in seq(0, MR):
                    C[cj,ci] = Cb[cj,ci]
    

    def ukernel_beta0(
            KC: size,
            alpha: f32[1],
            A: f32[KC, MR] @ DRAM,
            B: f32[KC, NR] @ DRAM,
            beta: f32[1],
            C: f32[NR, MR] @ DRAM,
            ):
        
        for j in seq(0, NR):
            for i in seq(0, MR):
                C[j,i] = 0.0
                    
        for k in seq(0, KC):
            for j in seq(0, NR):
                for i in seq(0, MR):
                    C[j, i] += A[k,i] * B[k,j]

    if (beta1 == True):
        return proc(ukernel_beta1)
    else:
        return proc(ukernel_beta0)

