from base_ukr import generate_optimized_ukr, generate_original_ukr
from exo.platforms.neon import *




MR = 8
NR = 12
KC = 512
LANE = 4
alpha1=True
beta1=False
beta1=True
intrinsics32 = []
intrinsics16 = []
arch = "NEON"
if arch == "NEON":
    intrinsics32 = {'load': neon_vld_4xf32, 'store': neon_vst_4xf32, 'fmla':  neon_vfmla_4xf32_4xf32, 
            'bcast': neon_broadcast_4xf32, 'vmul':neon_vmul_4xf32, 'zeros': neon_zero_4xf32}
    LANE = 4
else:
    print("Not supported hardware: {}".format(arch))
lM = [4,8,12,16,20,24]
lN = [4,8,12,16,24]

lM = [8]
lN = [12]

def registers(M,N,LANE):
    C = M * N//LANE
    A = M//LANE
    B = N//LANE
    return C+A+B

# Version of the paper CG0 2024
for i in lM:
    for j in lN:
        r = registers(i,j,LANE)
        if r > 32:
            print("Skipping {}x{} because it uses {} registers".format(i,j,r))
            continue
        locals()['uk_{0}x{1}'.format(i,j)] = generate_optimized_ukr(i, j, KC, alpha1,  beta1, arch, LANE,intrinsics32, windowing=1) 
        #locals()['uk_{0}x{1}_b'.format(i,j)] = generate_optimized_ukr(i, j, KC, alpha1,  False, arch, LANE,intrinsics32, windowing=1) 
"""
NR = 8
## MR = 8 NR = 8
y = generate_optimized_ukr(MR, NR, KC, alpha1, beta1, LANE,windowing=1) #windowing version
print("VFINAL\n",y)
NR = 4
## MR = 8 NR = 4
z = generate_optimized_ukr(MR, NR, KC, alpha1, beta1, LANE,windowing=1) #windowing version
print("VFINAL\n",z)

MR = 4
NR = 12
z412 = generate_optimized_ukr(MR, NR, KC, alpha1, beta1, LANE,windowing=1) #windowing version
print("VFINAL\n",z412)


NR = 8
z48 = generate_optimized_ukr(MR, NR, KC, alpha1, beta1, LANE,windowing=1) #windowing version
print("VFINAL\n",z48)

NR = 4
z44 = generate_optimized_ukr(MR, NR, KC, alpha1, beta1, LANE,windowing=1) #windowing version
print("VFINAL\n",z44)
#Just for check
# if we want use fp16 
#w = generate_optimized_ukr(MR, NR, KC, alpha1, beta1, LANE,windowing=1, data="f16") #windowing version
#print(w)
"""
