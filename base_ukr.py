# example.py
from __future__ import annotations
#from exo import *
from exo import proc
from exo.platforms.x86 import *
from exo.platforms.neon import *

from exo.stdlib.scheduling import *



#@proc
def ukernel_get_ref(
    MR: size,
    NR: size,
    alpha1: bool,
    beta1: bool,
    
):
    def ukernel_ref(
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
    
    return proc(ukernel_ref)


def generate_original_ukr(MR, NR, KC, alpha1, beta1):
      p = simplify(ukernel_get_ref(MR,NR, alpha1, beta1))
      p = rename(p, "example_sgemm_a1{}_b1{}".format(alpha1,beta1))
      return p


def split_loop(p, loop, LANE):
    return divide_loop(p,loop, LANE, ['{}t'.format(loop),'{}tt'.format(loop)], perfect=True)

def vectorial_memory(p, var, mem):
    p = set_memory(p, var, mem)
    return p

def from_X_to_Xreg(p, Buf, loop, F, up1, up2, LANE, data):
    Xreg='{}_reg'.format(Buf)
    p = bind_expr(p, '{}[_]'.format(Buf),Xreg)
    if F % LANE == 0:
        p = expand_dim(p, Xreg , LANE, '{}tt'.format(loop), unsafe_disable_checks=True)
        p = expand_dim(p, Xreg, F//LANE, '{}t'.format(loop), unsafe_disable_checks=True)
        p = lift_alloc(p, Xreg, n_lifts=up1)
        p = autofission(p, p.find('{}[_] = _'.format(Xreg)).after(),n_lifts=up2)
        if data == "f32":
            p = replace(p, 'for {}tt in _: _ #0'.format(loop), neon_vld_4xf32)
            p = set_memory(p, Xreg, Neon)
        elif data == "f16":
            p = set_precision(p, Xreg, data)
            p = replace(p, 'for {}tt in _: _ #0'.format(loop), neon_vld_8xf16)
            p = set_memory(p, Xreg, Neon8f)
    else:
        p = expand_dim(p, Xreg , LANE, 'jtt'.format(loop), unsafe_disable_checks=True)
        p = expand_dim(p, Xreg , F, '{}'.format(loop), unsafe_disable_checks=True)
        p = lift_alloc(p, Xreg, n_lifts=up1)
        p = autofission(p, p.find('{}[_] = _'.format(Xreg)).after(),n_lifts=up2)
        print('DEBUG',p)
        if data == "f32":
            p = replace(p, 'for jtt in _: _ #0', neon_broadcast_4xf32)
            p = set_memory(p, Xreg, Neon)
        elif data == "f16":
            p = set_precision(p, Xreg, data)
            p = replace(p, 'for jtt in _: _ #0', neon_broadcast_8xf16)
            p = set_memory(p, Xreg, Neon8f)
    return p


def from_C_to_Creg_2d(p, MR, NR, beta1, LANE, data):
      #we need to tackle the initialization of C to 0
      name='C'
      if beta1 == False:
          name = name+'b'
      name_reg=name+'_reg'
      if beta1 == False:
          name_reg = 'tmp'
      Cp = '{}[{} * jt + jtt, {} * it + itt]'.format(name,LANE,LANE)
      p = stage_mem(p, '{}[_] += _'.format(name), Cp, name_reg)
      p = expand_dim(p, name_reg, LANE, 'itt', unsafe_disable_checks=True)
      p = expand_dim(p, name_reg, MR//LANE, 'it', unsafe_disable_checks=True)
      p = expand_dim(p, name_reg, NR, 'jt*{}+jtt'.format(LANE), unsafe_disable_checks=True)
      if beta1 == False:
          p = reuse_buffer(p, "Cb_reg:_", 'tmp')
          name_reg = 'Cb_reg'
      if beta1 == True:
           p = lift_alloc(p, name_reg, n_lifts=5)
      p = autofission(p, p.find('{}[_] = _'.format(name_reg)).after(), n_lifts=5)
      p = autofission(p, p.find('{}[_] = _'.format(name)).before(), n_lifts=5)
      
      #TODO: read instructions and type from file
      if data == "f32": 
          p = replace(p, 'for itt in _: _ #0', neon_vld_4xf32)
          p = replace(p, 'for itt in _: _ #1', neon_vst_4xf32)
          p = vectorial_memory(p, name_reg, Neon)
      elif data == "f16": 
          p = replace(p, 'for itt in _: _ #0', neon_vld_8xf16)
          p = replace(p, 'for itt in _: _ #1', neon_vst_8xf16)
          p = vectorial_memory(p, name_reg, Neon8f)
      p = unroll_loop(p,'it')
      p = unroll_loop(p,'jtt')
      p = unroll_loop(p,'jt')
      return  simplify(p)

def bcast_scalar(p,scal,loop,u1,u2,LANE, data):
    scr = '{}_reg'.format(scal)
    p = bind_expr(p,scal,scr)
    p = expand_dim(p, scr, LANE, loop, unsafe_disable_checks=True)
    p = lift_alloc(p, scr, n_lifts=u1)
    p = autofission(p, p.find('{}[_] = _'.format(scr)).after(), n_lifts=u2)
    if data == "f32":
        p = replace(p, 'for {} in _: _ '.format(loop), neon_broadcast_4xf32)
        p = set_memory(p, scr, Neon)
    elif data == "f16":
        p = replace(p, 'for {} in _: _ '.format(loop), neon_broadcast_8xf16)
        p = set_memory(p, scr, Neon8f)
    return p

def manage_C_init(p,MR,NR,LANE, data):
    name = 'Cb'
    namer = name+'_reg'
    p = split_loop(p, 'ci', LANE)
    exp = '{}[cj, {} * cit + citt]'.format(name,LANE)
    p = stage_mem(p, '{}[_] = _'.format(name), exp, namer)
    p = expand_dim(p, namer, LANE, 'citt', unsafe_disable_checks=True)
    p = expand_dim(p, namer, MR//LANE, 'cit', unsafe_disable_checks=True)
    p = expand_dim(p, namer, NR, 'cj', unsafe_disable_checks=True)
    p = lift_alloc(p, namer, n_lifts=3)
    p = autofission(p, p.find('{}[_] = _'.format(namer)).after(), n_lifts=3)
    if data == "f32":
        p = set_memory(p, namer, Neon)
    elif data == "f16":
        p = set_memory(p, namer, Neon8f)
    p = bcast_scalar(p,'beta','citt',3,3,LANE, data)
    p = from_X_to_Xreg(p,'C','ci',MR,3,2,LANE, data)
    if data == "f32":
        p = replace(p, 'for citt in _: _ #0', neon_vmul_4xf32)
        p = replace(p, 'for citt in _: _ #0', neon_vst_4xf32)
    elif data == "f16":
        p = replace(p, 'for citt in _: _ #0', neon_vmul_8xf16)
        p = replace(p, 'for citt in _: _ #0', neon_vst_8xf16)
    return simplify(p)

def specialize_microkernel(p, precision, alpha1, beta1):
    args = ["A", "B", "C", "alpha", "beta"]
    for arg in args:
        p = set_precision(p, arg, precision)
    if beta1 == False:
        p = set_precision(p, "Cb", precision)

    return p

def set_windowing(p):
    p = set_window(p, "C", True)
    return p

def from_C_to_Creg_1d(p, MR, NR, beta1, LANE, data):
      #we need to tackle the initialization of C to 0
      name='C'
      if beta1 == False:
          name = name+'b'
      name_reg=name+'_reg'
      if beta1 == False:
          name_reg = 'tmp'
      Cp = '{}[{} * jt + jtt, i]'.format(name,LANE)
      p = stage_mem(p, '{}[_] += _'.format(name), Cp, name_reg)
      p = expand_dim(p, name_reg, LANE, 'jtt', unsafe_disable_checks=True)
      p = expand_dim(p, name_reg, NR//LANE, 'jt'.format(LANE), unsafe_disable_checks=True)
      p = expand_dim(p, name_reg, MR, 'i', unsafe_disable_checks=True)
      if beta1 == False:
          p = reuse_buffer(p, "Cb_reg:_", 'tmp')
          name_reg = 'Cb_reg'
      if beta1 == True:
           p = lift_alloc(p, name_reg, n_lifts=4)
      p = autofission(p, p.find('{}[_] = _'.format(name_reg)).after(), n_lifts=4)
      p = autofission(p, p.find('{}[_] = _'.format(name)).before(), n_lifts=4)
      print(p) 
      
      #TODO: read instructions and type from file
      if data == "f32": 
          p = replace(p, 'for jtt in _: _ #0', neon_vld_4xf32)
          p = replace(p, 'for jtt in _: _ #1', neon_vst_4xf32)
          p = vectorial_memory(p, name_reg, Neon)
      elif data == "f16": 
          p = replace(p, 'for jtt in _: _ #0', neon_vld_8xf16)
          p = replace(p, 'for jtt in _: _ #1', neon_vst_8xf16)
          p = vectorial_memory(p, name_reg, Neon8f)
      p = unroll_loop(p,'jt')
      p = unroll_loop(p,'i')
      return  simplify(p)

def generate_non_multiple(p, MR,NR,KC,alpha1,beta1,LANE,windowing, data):
    p = reorder_loops(p, 'j i')
    p = split_loop(p, 'j', LANE)
    p = simplify(p)
    print("LOOPS\n",p)
    print("Warning! We are working on it so the result may not be the desired one :)")
    # WOR-IN-PROGRESS
    # THIS IS NOT READY FOR PRODUCTION
    return p
    # C 
    p = from_C_to_Creg_1d(p, MR, NR, beta1, LANE, data)
    p = simplify(p)
    print("C\n",p)
    # A
    p = from_X_to_Xreg(p, 'A', 'i' , MR, 4, 3, LANE, data)
    p = simplify(p)
    print("A\n",p)
    # B
    p = from_X_to_Xreg(p, 'B', 'j' , NR,4, 3, LANE, data)
    p = simplify(p)
    print("B\n",p)
    if data == "f32":
      p = replace(p, 'for jtt in _: _ #0', neon_vfmadd_4xf32_4xf32)
    else:
      p = replace(p, 'for jtt in _: _ #0', neon_vfmadd_8xf16_8xf16)
    p = simplify(p)
    print("FMLA\n",p)
    return p

def generate_optimized_ukr(MR, NR, KC, alpha1, beta1, LANE, windowing = 0, data="f32"):
      p = simplify(ukernel_get_ref(MR,NR, alpha1, beta1))

      if data != "f32":
          p = specialize_microkernel(p,data,alpha1,beta1)
      if windowing:
          p = rename(p, "uk_wind_{}x{}_a1{}_b1{}".format(MR,NR,alpha1,beta1))
          p = set_windowing(p)
      else:
          p = rename(p, "uk_{}x{}_a1{}_b1{}".format(MR,NR,alpha1,beta1))
      #loops partition
      #manage C initialization
      if MR % LANE != 0:
          return generate_non_multiple(p, MR,NR,KC,alpha1,beta1,LANE,windowing, data)
      
      if beta1 == False:
          p = manage_C_init(p, MR,NR, LANE, data)
      
      #main loop
      p = split_loop(p, 'i', LANE)
      p = split_loop(p, 'j', LANE)
      p = simplify(p)
      print("LOOPS\n",p)
      # C 
      p = from_C_to_Creg_2d(p, MR, NR, beta1, LANE, data)
      p = simplify(p)
      print("C\n",p)
      # A
      p = from_X_to_Xreg(p, 'A', 'i' , MR, 5, 4, LANE, data)
      p = simplify(p)
      print("A\n",p)
      # B
      p = from_X_to_Xreg(p, 'B', 'j' , NR,5, 4, LANE, data)
      p = simplify(p)
      print("B\n",p)
      
      # fmla
      p = reorder_loops(p,'jtt it')
      if data == "f32":
          p = replace(p, 'for itt in _: _ #0', neon_vfmla_4xf32_4xf32)
      else:
          p = replace(p, 'for itt in _: _ #0', neon_vfmla_8xf16_8xf16)
      p = simplify(p)
      print("FMLA\n",p)
      #unroll A and B loads
      p = unroll_loop(p,'it')
      p = unroll_loop(p,'jt')
      
      #unroll fmla
      p = unroll_loop(p,'jtt')
      p = unroll_loop(p,'it')
      p = unroll_loop(p,'jt')
      
      #unroll C store
      p = unroll_loop(p,'it')
      p = unroll_loop(p,'jtt')
      p = unroll_loop(p,'jt')
      print("FINAL\n",p)
      
      if beta1 == False:
          p = manage_C_end(p, MR,NR, LANE)
      
      return p


