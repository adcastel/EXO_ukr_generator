# example.py
from __future__ import annotations
#from exo import *
import exo
from exo import proc
from exo.platforms.neon import *
from exo.stdlib.scheduling import *
from kernels.gemm.ukgemm import *


@exo.instr("{dst_data} = vfmaq_laneq_f32({dst_data}, {lhs_data}, {rhs_data}, {lane});")
def neon_vfmladrian_4xf32_4xf32(dst: [f32][4] @ Neon, lhs: [f32][4] @ Neon, rhs: [f32][4]@ Neon, lane: index):
    assert stride(dst, 0) == 1
    assert stride(lhs, 0) == 1
    assert stride(rhs, 0) == 1
    assert lane >= 0
    assert lane < 4
    
    for i in seq(0, 4):
        dst[i] += lhs[i] * rhs[i]



def generate_original_ukr(MR, NR, KC, alpha1, beta1):
      p = simplify(ukernel_get_ref(MR,NR, alpha1, beta1))
      p = rename(p, "example_sgemm_a1{}_b1{}".format(alpha1,beta1))
      return p

######## LOOP ZONE
def split_loop(p, loop, LANE):
    return divide_loop(p,loop, LANE, ['{}t'.format(loop),'{}tt'.format(loop)], perfect=True)

def loop_split(p, loop, size):
    for l in loop:
        p = split_loop(p, l, size)
    return p

def loop_unroll(p, loop):
    for l in loop:
        p = unroll_loop(p, l)
    return p

#########  MEMORY ZONE

def vectorial_memory(p, var, mem):
    p = set_memory(p, var, mem)
    return p

def from_X_to_Xreg(p, Buf, loop, F, up1, up2, LANE, intrinsics, data):
    Xreg='{}_reg'.format(Buf)
    p = bind_expr(p, '{}[_]'.format(Buf),Xreg)
    p = expand_dim(p, Xreg , LANE, '{}tt'.format(loop), unsafe_disable_checks=True)
    p = expand_dim(p, Xreg, F//LANE, '{}t'.format(loop), unsafe_disable_checks=True)
    p = lift_alloc(p, Xreg, n_lifts=up1)
    p = autofission(p, p.find('{}[_] = _'.format(Xreg)).after(),n_lifts=up2)
    if data == "f16":
        p = set_precision(p, Xreg, data)
    p = replace(p, 'for {}tt in _: _ #0'.format(loop), intrinsics['load'])
    p = set_memory(p, Xreg, Neon)
    return p


def from_C_to_Creg_2d(p, MR, NR, beta1, LANE, intrinsics, data):
      #we need to tackle the initialization of C to 0
      name='C'
      #if beta1 == False:
      #    name = name+'b'
      name_reg=name+'_reg'
      #if beta1 == False:
      #    name_reg = 'tmp'
      if beta1 == True:
          Cp = '{}[{} * jt + jtt, {} * it + itt]'.format(name,LANE,LANE)
          p = stage_mem(p, '{}[_] += _'.format(name), Cp, name_reg)
      else:
          Cp = '{}[jtt + {} * jt, itt + {} * it]'.format(name,LANE,LANE)
          name_reg=name+'_reg0'
          p = stage_mem(p, '{}[_] += _'.format(name), Cp, name_reg)
      p = expand_dim(p, name_reg, LANE, 'itt', unsafe_disable_checks=True)
      p = expand_dim(p, name_reg, MR//LANE, 'it', unsafe_disable_checks=True)
      p = expand_dim(p, name_reg, NR, 'jt*{}+jtt'.format(LANE), unsafe_disable_checks=True)
      if beta1 == False:
          p = reuse_buffer(p, "C_reg:_", 'C_reg0')
      #    p = reuse_buffer(p, "Cb_reg:_", 'tmp')
          name_reg = 'C_reg'
      if beta1 == True:
          p = lift_alloc(p, name_reg, n_lifts=5)
      if beta1 == False:
          p = fuse(p,'for jtt in _: _ #0','for jtt in _: _ #1')
          p = fuse(p,'for it in _: _ #0','for it in _: _ #1')
          #p = fuse(p,'for itt in _: _ #0','for itt in _: _ #1')
          #p = reorder_stmts(p, 'C_reg[jtt + 4 * jt, it, itt] = 0.0; C[jtt + 4 * jt, itt + 4 * it] = C_reg[jtt + 4 * jt, it, itt]') 
      
      if beta1 == True:
          p = autofission(p, p.find('{}[_] = _'.format(name_reg)).after(), n_lifts=5)
          p = autofission(p, p.find('{}[_] = _'.format(name)).before(), n_lifts=5)
      else:
          p = autofission(p, p.find('{}[_] = _ #1'.format(name_reg)).after(), n_lifts=5)
          p = autofission(p, p.find('{}[_] = _ #1'.format(name)).before(), n_lifts=5)
          p = fuse(p,'for jt in _: _ #0','for jt in _: _ #1')
          p = fuse(p,'for jtt in _: _ #0','for jtt in _: _ #1')
          p = fuse(p,'for it in _: _ #0','for it in _: _ #1')
      
      #TODO: read instructions and type from file
      if beta1 == True:
          p = replace(p, 'for itt in _: _ #0', intrinsics['load'])#neon_vld_4xf32)
          p = replace(p, 'for itt in _: _ #1', intrinsics['store'])
      else:
          p = replace(p, 'for itt in _: _ #0', intrinsics['zeros'])#neon_vld_4xf32)
          p = replace(p, 'for itt in _: _ #0', intrinsics['store'])#neon_vld_4xf32)
          p = replace(p, 'for itt in _: _ #0', intrinsics['load'])
          p = replace(p, 'for itt in _: _ #1', intrinsics['store'])

      p = vectorial_memory(p, name_reg, Neon)
      
      p = loop_unroll(p,['it','jtt','jt'])
      return  simplify(p)

def buffer_unrolling(p,name_reg, dims):
    p = unroll_buffer(p, '{}:_'.format(name_reg),0)
    for i in range(dims):
        ll='{}_{}:_'.format(name_reg,i)
        p = unroll_buffer(p, ll, 0)
    return p

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
        p = set_memory(p, scr,Neon)
    return p

def manage_C_init(p,MR,NR,LANE, intrinsics, data):
    name = 'C'
    name_reg = name+'_reg'
    p = loop_split(p, ['i','j'], LANE)
    p = simplify(p)
    exp = '{}[jtt + {} * jt, itt + {} * it]'.format(name,LANE,LANE)
    p = stage_mem(p, '{}[_] = _'.format(name), exp, name_reg)
    
    p = expand_dim(p, name_reg, LANE, 'itt', unsafe_disable_checks=True)
    p = expand_dim(p, name_reg, MR//LANE, 'it', unsafe_disable_checks=True)
    p = expand_dim(p, name_reg, NR, 'jt*{}+jtt'.format(LANE), unsafe_disable_checks=True)
    
    p = lift_alloc(p, name_reg, n_lifts=4)
    p = autofission(p, p.find('{}[_] = _'.format(name_reg)).after(), n_lifts=3)
    #if data == "f32":
    p = set_memory(p, name_reg, Neon)
    #p = replace(p, 'for itt in _: _ #0', neon_zero_4xf32)
    
    #elif data == "f16":
    #    p = set_memory(p, namer, Neon)
    #p = bcast_scalar(p,'beta','citt',3,3,LANE, data)
    #p = from_X_to_Xreg(p,'C','ci',MR,3,2,LANE, data)
    #if data == "f32":
    #    p = replace(p, 'for citt in _: _ #0', neon_vmul_4xf32)
    #    p = replace(p, 'for citt in _: _ #0', neon_vst_4xf32)
    #elif data == "f16":
    #    p = replace(p, 'for citt in _: _ #0', neon_vmul_8xf16)
    #    p = replace(p, 'for citt in _: _ #0', neon_vst_8xf16)
    return simplify(p)

def specialize_microkernel(p, precision, alpha1, beta1):
    args = ["A", "B", "C", "alpha", "beta"]
    for arg in args:
        p = set_precision(p, arg, precision)
    #if beta1 == False:
    #    p = set_precision(p, "Cb", precision)

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
      
      #TODO: read instructions and type from file
      if data == "f32": 
          p = replace(p, 'for jtt in _: _ #0', neon_vld_4xf32)
          p = replace(p, 'for jtt in _: _ #1', neon_vst_4xf32)
          p = vectorial_memory(p, name_reg, Neon)
      elif data == "f16": 
          p = replace(p, 'for jtt in _: _ #0', neon_vld_8xf16)
          p = replace(p, 'for jtt in _: _ #1', neon_vst_8xf16)
          p = vectorial_memory(p, name_reg,Neon)
      p = unroll_loop(p,'jt')
      p = unroll_loop(p,'i')
      return  simplify(p)



def generate_optimized_ukr(MR, NR, KC, alpha1, beta1, arch, LANE, intrinsics, windowing = 0, data="f32"):
      p = simplify(ukernel_get_ref(MR,NR, alpha1, beta1))
      p = specialize_microkernel(p,data,alpha1,beta1)
      if windowing:
          p = rename(p, "gemm_{}_{}x{}_beta{}_{}".format(arch, MR,NR,1 if beta1 else 0,data))
          p = set_windowing(p)
      else:
          p = rename(p, "gemm_{}_{}x{}_beta{}_{}".format(arch, MR,NR,1 if beta1 else 0, data))
      
      
      if beta1 == False:
          p = manage_C_init(p, MR,NR, LANE, intrinsics, data)
          p = simplify(p)
      
      #main loop
      p = loop_split(p, ['i','j'], LANE)
      p = simplify(p)
      # C 
      p = from_C_to_Creg_2d(p, MR, NR, beta1, LANE, intrinsics, data)
      p = simplify(p)
      # A
      p = from_X_to_Xreg(p, 'A', 'i' , MR, 5, 4, LANE, intrinsics, data)
      p = simplify(p)
      # B
      p = from_X_to_Xreg(p, 'B', 'j' , NR,5, 4, LANE, intrinsics, data)
      p = simplify(p)
      
      # fmla
      p = reorder_loops(p,'jtt it')
      print(p)
      #p = replace(p, 'for itt in _: _ #0', intrinsics['fmla'])
      p = simplify(p)
      p = replace(p, 'for itt in _: _ #0', neon_vfmladrian_4xf32_4xf32)
      p = simplify(p)
      
      #unroll A and B loads
      p = loop_unroll(p,['it','jt'])
      p = simplify(p)
      
      #unroll fmla
      p = loop_unroll(p,['jtt','it','jt'])
      p = simplify(p)
      
      #unroll C store
      p = loop_unroll(p,['it','jtt','jt'])
      p = simplify(p)
      p = buffer_unrolling(p,'C_reg', NR)
      p = simplify(p)
      p = buffer_unrolling(p,'A_reg', 0)
      print(p)
      try:
          p = buffer_unrolling(p,'B_reg', 0)
      except:
          print("WARNING with {}x{}!!".format(MR,NR))
      return p


