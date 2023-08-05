import numpy as np
import torch

# class ReplayBuffer:
#     def __init__(self, capacity, device='cpu'):
#         self.buffer = [] 
#         self.device = device
#         self.capacity = capacity

#     def add(self, state, action, reward, next_state, done, agent_model=None): 

#         sample = SingleSample(state, action, reward, next_state, done, device=self.device) 

#         self.buffer.append(sample)
    
#         # 如果缓冲区超过容量，删除最老的样本
#         if len(self.buffer) > self.capacity:
#             self.buffer.pop(0)
#             self.td_error_buffer.pop(0)

#     def sample(self, batch_size): 
#         # 从优先级队列中根据概率选择样本
#         selected_indices = np.random.choice(len(self.buffer), batch_size)
#         selected_samples = [self.buffer[idx] for idx in selected_indices]
        
#         grouped_samples = list(zip(*[(sample.state, sample.action, sample.next_state, sample.reward, sample.done) for sample in selected_samples]))

#         output_samples = []
#         for i in range(5):
#             output_samples.append(torch.stack(grouped_samples[i], dim=0).squeeze(1))

        
#         return output_samples[0], output_samples[1], output_samples[2], output_samples[3], output_samples[4]
        

#     def size(self): 
#         return len(self.buffer)
    

class SingleSample:
    def __init__(self, state, action, reward, next_state, done, device='cpu'):
        self.device = device
        self.state = torch.tensor(np.array(state), dtype=torch.float32).to(self.device).unsqueeze(0)
        self.action = torch.tensor(np.array(action), dtype=torch.float32).to(self.device).unsqueeze(0)
        self.next_state = torch.tensor(np.array(next_state), dtype=torch.float32).to(self.device).unsqueeze(0)
        self.reward = torch.tensor(np.array(reward), dtype=torch.float32).to(self.device).view(-1, 1)
        self.done = torch.tensor(np.array(done), dtype=torch.float32).to(self.device).view(-1, 1)
        self.td_error = None


    def update_TD_error(self, td_error):
        self.td_error = td_error

    def __eq__(self, __value: object) -> bool:
        return torch.all(torch.eq(self.state, __value.state)) and torch.all(torch.eq(self.action, __value.action)) \
            and torch.all(torch.eq(self.reward, __value.reward)) and torch.all(torch.eq(self.next_state, __value.next_state)) \
            and torch.all(torch.eq(self.done, __value.done))
    
    def __lt__(self, other):
        return self.td_error < other.td_error

    
class ReplayBuffer:
    def __init__(self, capacity, device='cpu'):
        self.buffer = []
        self.td_error_buffer = []
        self.capacity = capacity
        self.device = device

    def add(self, state, action, reward, next_state, done, agent_model=None):

        sample = SingleSample(state, action, reward, next_state, done, device=self.device) 

        with torch.no_grad():
            next_q_value = agent_model.target_critic(sample.next_state, agent_model.target_actor(sample.next_state))
            q_target = sample.reward + agent_model.gamma * next_q_value * (1 - sample.done)
            q_value = agent_model.critic(sample.state, sample.action)
            td_error = torch.abs(q_target - q_value).cpu().numpy()[0][0]
            # sample.update_TD_error(td_error)

        self.buffer.append(sample)
        self.td_error_buffer.append(td_error)
    
        # 如果缓冲区超过容量，删除最老的样本
        if len(self.buffer) > self.capacity:
            self.buffer.pop(0)
            self.td_error_buffer.pop(0)

        
    def sample(self, batch_size): 

        probs = np.array(self.td_error_buffer, dtype=np.float32)
        # 计算最小值和范围
        min_val, max_val = np.min(probs), np.max(probs)
        range_val = max_val - min_val

        # 使用 Min-Max 归一化
        probs = (probs - min_val) / range_val

        probs = np.exp(probs)
        probs /= np.sum(probs)

        # 从优先级队列中根据概率选择样本
        selected_indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        selected_samples = [self.buffer[idx] for idx in selected_indices]
        
        grouped_samples = list(zip(*[(sample.state, sample.action, sample.next_state, sample.reward, sample.done) for sample in selected_samples]))

        output_samples = []
        for i in range(5):
            output_samples.append(torch.stack(grouped_samples[i], dim=0).squeeze(1))

        
        return output_samples[0], output_samples[1], output_samples[2], output_samples[3], output_samples[4]
        

    def size(self): 
        return len(self.buffer)
