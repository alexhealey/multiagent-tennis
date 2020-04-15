import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.normal import Normal
import numpy as np
import importlib
import model
importlib.reload(model)
import buffer
importlib.reload(buffer)

class SACAgent:

    def __init__(self, device, env, hyperparams):

        self.device = device
        self.env = env
        brain_name = env.brain_names[0]
        brain = env.brains[brain_name]
        self.action_size = brain.vector_action_space_size
        env_info = env.reset(train_mode=True)[brain_name]
        states = env_info.vector_observations
        self.state_size = states.shape[1]

        # hyperparameters
        self.gamma = hyperparams["gamma"]
        self.tau = hyperparams["tau"]
        self.update_step = hyperparams.get("update_step", 0)
        self.delay_step = hyperparams.get("delay_step", 2)

        # initialize networks
        self.q_net1 = model.QNetwork(self.state_size, self.action_size, hyperparams).to(self.device)
        self.q_net2 = model.QNetwork(self.state_size, self.action_size, hyperparams).to(self.device)
        self.target_q_net1 = model.QNetwork(self.state_size, self.action_size, hyperparams).to(self.device)
        self.target_q_net2 = model.QNetwork(self.state_size, self.action_size, hyperparams).to(self.device)
        self.policy_net = model. GaussianPolicyNetwork(self.state_size, self.action_size, hyperparams).to(self.device)

        # copy params to target param
        for target_param, param in zip(self.target_q_net1.parameters(), self.q_net1.parameters()):
            target_param.data.copy_(param)

        for target_param, param in zip(self.target_q_net2.parameters(), self.q_net2.parameters()):
            target_param.data.copy_(param)

        # initialize optimizers
        q_learn_rate = hyperparams["q_learn_rate"]
        policy_learn_rate = hyperparams["policy_learn_rate"]
        self.q1_optimizer = optim.Adam(self.q_net1.parameters(), lr=q_learn_rate)
        self.q2_optimizer = optim.Adam(self.q_net2.parameters(), lr=q_learn_rate)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=policy_learn_rate)

        # entropy temperature
        self.alpha = hyperparams["alpha"]
        a_learn_rate = hyperparams["a_learn_rate"]
        self.target_entropy = -brain.vector_action_space_size
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optim = optim.Adam([self.log_alpha], lr=a_learn_rate)

        self.replay_buffer = buffer.SimpleBuffer(self.device, 0, hyperparams)

    def learn_experience(self, experience):
        self.replay_buffer.add(experience)
        if self.replay_buffer.ready_to_sample():
            self.update()

    def get_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        mean, log_std = self.policy_net.forward(state)
        std = log_std.exp()

        normal = Normal(mean, std)
        z = normal.sample()
        action = torch.tanh(z)
        action = action.cpu().detach().squeeze(0).numpy()

        return action

    def update(self):
        states, actions, rewards, next_states, dones = self.replay_buffer.sample()
        dones = dones.view(dones.size(0), -1)

        next_actions, next_log_pi = self.policy_net.sample(next_states)
        next_q1 = self.target_q_net1(next_states, next_actions)
        next_q2 = self.target_q_net2(next_states, next_actions)
        next_q_target = torch.min(next_q1, next_q2) - self.alpha * next_log_pi
        expected_q = rewards + (1 - dones) * self.gamma * next_q_target

        # q loss
        curr_q1 = self.q_net1.forward(states, actions)
        curr_q2 = self.q_net2.forward(states, actions)
        q1_loss = F.mse_loss(curr_q1, expected_q.detach())
        q2_loss = F.mse_loss(curr_q2, expected_q.detach())

        # update q networks
        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        self.q1_optimizer.step()

        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        self.q2_optimizer.step()

        # delayed update for policy network and target q networks
        new_actions, log_pi = self.policy_net.sample(states)
        if self.update_step % self.delay_step == 0:
            min_q = torch.min(
                self.q_net1.forward(states, new_actions),
                self.q_net2.forward(states, new_actions)
            )
            policy_loss = (self.alpha * log_pi - min_q).mean()

            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()

            # target networks
            for target_param, param in zip(self.target_q_net1.parameters(), self.q_net1.parameters()):
                target_param.data.copy_(self.tau * param + (1 - self.tau) * target_param)

            for target_param, param in zip(self.target_q_net2.parameters(), self.q_net2.parameters()):
                target_param.data.copy_(self.tau * param + (1 - self.tau) * target_param)

        # update temperature
        alpha_loss = (self.log_alpha * (-log_pi - self.target_entropy).detach()).mean()

        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()
        self.alpha = self.log_alpha.exp()

        self.update_step += 1

    def create_frozen(self, hyperparams):
        return FrozenSACAgent(self, hyperparams)

    def get_policy_state_dict(self):
        return self.policy_net.state_dict()

class FrozenSACAgent:

    def __init__(self, training_agent, hyperparams):
        self.device = training_agent.device
        self.policy_net = model.GaussianPolicyNetwork(training_agent.state_size, training_agent.action_size, hyperparams).to(training_agent.device)
        self.policy_net.load_state_dict(training_agent.policy_net.state_dict())

    def copy_and_freeze(self, agent):
        self.policy_net.load_state_dict(agent.policy_net.state_dict())

    def get_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        mean, log_std = self.policy_net.forward(state)
        std = log_std.exp()

        normal = Normal(mean, std)
        z = normal.sample()
        action = torch.tanh(z)
        action = action.cpu().detach().squeeze(0).numpy()

        return action

class DDPGAgent:

    def __init__(self, device, env, hyperparams):

        self.device = device
        self.env = env
        brain_name = env.brain_names[0]
        brain = env.brains[brain_name]
        self.action_size = brain.vector_action_space_size
        env_info = env.reset(train_mode=True)[brain_name]
        states = env_info.vector_observations
        self.state_size = states.shape[1]

        # hyperparameters
        self.gamma = hyperparams["gamma"]
        self.tau = hyperparams["tau"]
        self.update_step = hyperparams.get("update_step", 0)
        self.delay_step = hyperparams.get("delay_step", 2)

        # initialize networks
        self.q_net = model.QNetwork(self.state_size, self.action_size, hyperparams).to(self.device)
        self.target_q_net = model.QNetwork(self.state_size, self.action_size, hyperparams).to(self.device)
        self.policy_net = model.DeterministicPolicyNetwork(self.state_size, self.action_size, hyperparams).to(self.device)
        self.target_policy_net = model.DeterministicPolicyNetwork(self.state_size, self.action_size, hyperparams).to(self.device)

        # copy params to target param
        for target_param, param in zip(self.target_q_net.parameters(), self.q_net.parameters()):
            target_param.data.copy_(param)

        for target_param, param in zip(self.target_policy_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(param)

        # initialize optimizers
        q_learn_rate = hyperparams["q_learn_rate"]
        policy_learn_rate = hyperparams["policy_learn_rate"]
        self.q_optimizer = optim.Adam(self.q_net.parameters(), lr=q_learn_rate)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=policy_learn_rate)

        self.replay_buffer = buffer.SimpleBuffer(self.device, 0, hyperparams)

    def learn_experience(self, experience):
        self.replay_buffer.add(experience)
        if self.replay_buffer.ready_to_sample():
            self.update()

    def get_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        return self.policy_net.sample(state).squeeze()

    def update(self):
        states, actions, rewards, next_states, dones = self.replay_buffer.sample()
        dones = dones.view(dones.size(0), -1)

        next_actions = self.target_policy_net.forward(next_states)
        next_q_target = self.target_q_net(next_states, next_actions)
        expected_q = rewards + (1 - dones) * self.gamma * next_q_target

        # q loss
        curr_q = self.q_net.forward(states, actions)
        q_loss = F.mse_loss(curr_q, expected_q.detach())

        # update q network
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        # delayed update for policy network and target networks
        new_actions = self.policy_net.forward(states)
        if self.update_step % self.delay_step == 0:
            q_vals = self.q_net.forward(states, new_actions)
            policy_loss = (-q_vals).mean()

            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()

            # target networks
            for target_param, param in zip(self.target_q_net.parameters(), self.q_net.parameters()):
                target_param.data.copy_(self.tau * param + (1 - self.tau) * target_param)

            for target_param, param in zip(self.target_policy_net.parameters(), self.policy_net.parameters()):
                target_param.data.copy_(self.tau * param + (1 - self.tau) * target_param)

        self.update_step += 1

    def get_policy_state_dict(self):
        return self.policy_net.state_dict()

    def create_frozen(self, hyperparams):
            return FrozenDDPGAgent(self, hyperparams)

class FrozenDDPGAgent:

    def __init__(self, training_agent, hyperparams):
        self.device = training_agent.device
        self.policy_net = model.DeterministicPolicyNetwork(training_agent.state_size, training_agent.action_size, hyperparams).to(training_agent.device)
        self.policy_net.load_state_dict(training_agent.policy_net.state_dict())

    def copy_and_freeze(self, agent):
        self.policy_net.load_state_dict(agent.policy_net.state_dict())

    def get_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        return self.policy_net.sample(state).squeeze()
