from base_ukr import generate_optimized_ukr, generate_original_ukr


MR = 8
NR = 12
KC = 512
LANE = 4
alpha1=True
beta1=True

# Version of the paper CG0 2024
q = generate_optimized_ukr(MR, NR, KC, alpha1, beta1, LANE)
print(q)

# if we want to use windowed feature for C
#x = generate_optimized_ukr(MR, NR, KC, alpha1, beta1, LANE,windowing=1) #windowing version
#print(x)

# if we want use fp16 
#w = generate_optimized_ukr(MR, NR, KC, alpha1, beta1, LANE,windowing=1, data="f16") #windowing version
#print(w)
