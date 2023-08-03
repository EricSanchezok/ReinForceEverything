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