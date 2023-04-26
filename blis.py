from base_ukr import generate_optimized_ukr, generate_original_ukr


MR = 8
NR = 12
KC = 512
LANE = 4
alpha1=True
beta1=True
s = generate_original_ukr(MR, NR, KC, True, True)
p = generate_original_ukr(MR, NR, KC, True, False)
q = generate_optimized_ukr(MR, NR, KC, alpha1, beta1, LANE)
print(q)

beta1=False
w = generate_optimized_ukr(MR, NR, KC, alpha1, beta1, LANE) #windowing version
print(w)
