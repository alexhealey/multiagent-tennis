import collections
import numpy as np
import torch

Experience = collections.namedtuple("Experience", \
    ["start_state", "action", "reward", "next_state", "done"])

class SimpleBuffer:

    def __init__(self, device, seed, hyperparams):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.memory = [None] * hyperparams["buffer_size"]
        self.total_experiences = 0
        self.buffer_size = hyperparams["buffer_size"]
        self.batch_size = hyperparams["batch_size"]
        self.device = device
        np.random.seed(seed)

    def add(self, experience):
        """Add a new experience to memory."""
        self.memory[self.total_experiences % self.buffer_size] = experience
        self.total_experiences = self.total_experiences + 1

    def ready_to_sample(self):
        return len(self) >= self.batch_size

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        indices = np.random.choice(len(self), size=self.batch_size)

        states = torch.from_numpy(np.stack([self.memory[idx].start_state \
            for idx in indices if self.memory[idx] is not None])).float().to(self.device)
        actions = torch.from_numpy(np.stack([self.memory[idx].action \
            for idx in indices if self.memory[idx] is not None])).float().to(self.device)
        if isinstance(self.memory[0].reward, (list, tuple, np.ndarray)):
            rewards = torch.from_numpy(np.stack([self.memory[idx].reward \
                for idx in indices if self.memory[idx] is not None])).float().to(self.device)
        else:
            rewards = torch.from_numpy(np.vstack([self.memory[idx].reward \
                for idx in indices if self.memory[idx] is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.stack([self.memory[idx].next_state \
            if self.memory[idx].next_state is not None else self.memory[idx].start_state \
            for idx in indices if self.memory[idx] is not None])).float().to(self.device)
        if isinstance(self.memory[0].done, (list, tuple, np.ndarray)):
            dones = torch.from_numpy(np.stack([self.memory[idx].done \
                for idx in indices if self.memory[idx] is not None]).astype(np.uint8)).float().to(self.device)
        else:
            dones = torch.from_numpy(np.vstack([self.memory[idx].done \
                for idx in indices if self.memory[idx] is not None]).astype(np.uint8)).float().to(self.device)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        """Return the current size of internal memory."""
        return min(self.total_experiences, self.buffer_size)
