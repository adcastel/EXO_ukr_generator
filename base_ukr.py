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
):
    def ukernel_ref(
        KC: size,
        C: f32[NR, MR] @ DRAM,
        A: f32[KC, MR] @ DRAM,
        B: f32[KC, NR] @ DRAM,
    ):
        for k in seq(0, KC):
            for j in seq(0, NR):
                for i in seq(0, MR):
                    C[j, i] += A[k,i] * B[k,j] 


    return proc(ukernel_ref)



def generate_original_ukr(MR, NR, KC):
      p = simplify(ukernel_get_ref(MR,NR))
      p = rename(p, "example_sgemm")
      return p



def set_windowing(p):
    p = set_window(p, "A", True)
    p = set_window(p, "B", True)
    p = set_window(p, "C", True)
    return p

def split_loop(p, loop, LANE):
    return divide_loop(p,loop, LANE, ['{}t'.format(loop),'{}tt'.format(loop)], perfect=True)

def vectorial_memory(p, var, mem):
    p = set_memory(p, var, mem)
    return p

def from_X_to_Xreg(p, Buf, loop, F, LANE):
    Xreg='{}_reg'.format(Buf)
    p = bind_expr(p, '{}[_]'.format(Buf),Xreg)
    p = expand_dim(p, Xreg , LANE, '{}tt'.format(loop), unsafe_disable_checks=True)
    p = expand_dim(p, Xreg, F//LANE, '{}t'.format(loop), unsafe_disable_checks=True)
    p = lift_alloc(p, Xreg, n_lifts=5)
    p = autofission(p, p.find('{}[_] = _'.format(Xreg)).after(),n_lifts=4)
    p = replace(p, 'for {}tt in _: _ #0'.format(loop), neon_vld_4xf32)
    p = set_memory(p, Xreg, Neon)
    return p
def from_C_to_Creg_2d(p, MR, NR, LANE):
      Cp = 'C[{} * jt + jtt, {} * it + itt]'.format(LANE,LANE)
      p = stage_mem(p, 'C[_] += _', Cp, 'C_reg')
      p = expand_dim(p, 'C_reg', LANE, 'itt', unsafe_disable_checks=True)
      p = expand_dim(p, 'C_reg', MR//LANE, 'it', unsafe_disable_checks=True)
      p = expand_dim(p, 'C_reg', NR, 'jt*4+jtt', unsafe_disable_checks=True)
      p = lift_alloc(p, 'C_reg', n_lifts=5)
      p = autofission(p, p.find('C_reg[_] = _').after(), n_lifts=5)
      p = autofission(p, p.find('C[_] = _').before(), n_lifts=5)
      #TODO: read instructions and type from file
      p = replace(p, 'for itt in _: _ #0', neon_vld_4xf32)
      p = replace(p, 'for itt in _: _ #1', neon_vst_4xf32)
      p = vectorial_memory(p, 'C_reg', Neon)
      p = unroll_loop(p,'it')
      p = unroll_loop(p,'jtt')
      p = unroll_loop(p,'jt')
      return  p

def generate_optimized_ukr(MR, NR, KC, LANE, windowing = 0):
      p = simplify(ukernel_get_ref(MR,NR))
      if windowing:
          p = rename(p, "uk_wind_{}x{}".format(MR,NR))
      else:
          p = rename(p, "uk_{}x{}".format(MR,NR))
      #loops partition
      p = split_loop(p, 'i', LANE)
      p = split_loop(p, 'j', LANE)

      # C 
      p = from_C_to_Creg_2d(p, MR, NR, LANE)
      # A
      p = from_X_to_Xreg(p, 'A', 'i' , MR,LANE)
      # B
      p = from_X_to_Xreg(p, 'B', 'j' , NR,LANE)
      return p



"""

buf='A'
p = bind_expr(p, f'{buf}[_]', f'{buf}_vec')
p = expand_dim(p, f'{buf}_vec', LANE, 'itt', unsafe_disable_checks=True)
p = expand_dim(p, f'{buf}_vec', AFAC, 'it', unsafe_disable_checks=True)
p = lift_alloc(p, f'{buf}_vec', n_lifts=5)
p = replace(p, 'for itt in _: _ #0', neon_vld_4xf32)
p = set_memory(p, f'{buf}_vec', Neon4f)
#p = divide_loop(p,'i #1', LANE, ['ii','iii'], perfect=True)

print("A matrix\n",p)

buf='B'
p = bind_expr(p, f'{buf}[_]', f'{buf}_vec')
p = expand_dim(p, f'{buf}_vec', LANE, 'jtt', unsafe_disable_checks=True)
p = expand_dim(p, f'{buf}_vec', NO, 'jt', unsafe_disable_checks=True)
p = lift_alloc(p, f'{buf}_vec', n_lifts=5)
p = autofission(p, p.find(f'{buf}_vec[_] = _').after(),n_lifts=4)
p = replace(p, 'for jtt in _: _ #1', neon_vld_4xf32)
p = set_memory(p, f'{buf}_vec', Neon4f)
print("B matrix\n",p);
#p = unroll_loop(p,'it #0')
p = unroll_loop(p,'jtt #0')
#p = unroll_loop(p,'jt #0')

#p = unroll_loop(p,'it #0')
#p = unroll_loop(p,'jt #0')
#print("Pre pre Unroll\n",p); 
#p = reorder_loops(p,'jtt it')
#print("Pre Unroll\n",p); 
p = replace(p, 'for itt in _: _ #0', neon_vfmla_4xf32_4xf32)
#print("Unroll\n",p); 
#p = unroll_loop(p,'jtt #0')
#p = unroll_loop(p,'it #0')
#p = unroll_loop(p,'jt #0')
#p = unroll_loop(p,'it #0')
#p = unroll_loop(p,'jtt #0')
#p = unroll_loop(p,'jt #0')
p=simplify(p)
print("Final\n",p)
"""

