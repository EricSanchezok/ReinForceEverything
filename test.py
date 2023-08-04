import numpy as np

td_error_buffer = [-1, 0.2, 0.3, 0.4, 100]


probs = np.array(td_error_buffer, dtype=np.float32)
# 计算最小值和范围
min_val = np.min(probs)
max_val = np.max(probs)
range_val = max_val - min_val

# 使用 Min-Max 归一化
probs = (probs - min_val) / range_val

probs = np.exp(probs)
probs /= np.sum(probs)

print(probs)

# a = (1, 2)
# b = (3, 4)
# c = (5, 6)

# d = np.eye(500)[1]

# combined = np.concatenate((a, b, c), axis=0, dtype=np.float32)

# next_state = np.concatenate((d, combined), axis=0, dtype=np.float32)

# print(combined.shape)
# print(d.shape)
# print(next_state.shape)