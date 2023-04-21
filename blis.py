from base_ukr import generate_optimized_ukr, generate_original_ukr


MR = 8
NR = 12
KC = 512
LANE = 4
p = generate_original_ukr(MR, NR, KC)
q = generate_optimized_ukr(MR, NR, KC, LANE)
w = generate_optimized_ukr(MR, NR, KC, LANE, 1) #windowing version
print(q)
