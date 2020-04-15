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

class SACAgentPair:

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
        self.q_net1 = model.QNetwork(2 * self.state_size, 2 * self.action_size, hyperparams).to(self.device)
        self.q_net2 = model.QNetwork(2 * self.state_size, 2 * self.action_size, hyperparams).to(self.device)
        self.target_q_net1 = model.QNetwork(2 * self.state_size, 2 * self.action_size, hyperparams).to(self.device)
        self.target_q_net2 = model.QNetwork(2 * self.state_size, 2 * self.action_size, hyperparams).to(self.device)
        self.policy_net_a = model.GaussianPolicyNetwork(self.state_size, self.action_size, hyperparams).to(self.device)
        self.policy_net_b = model.GaussianPolicyNetwork(self.state_size, self.action_size, hyperparams).to(self.device)

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
        self.policy_optimizer = optim.Adam(list(self.policy_net_a.parameters()) + list(self.policy_net_b.parameters()), lr=policy_learn_rate)

        # entropy temperature
        self.alpha = hyperparams["alpha"]
        a_learn_rate = hyperparams["a_learn_rate"]
        self.target_entropy = -brain.vector_action_space_size
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optim = optim.Adam([self.log_alpha], lr=a_learn_rate)

        self.replay_buffer = buffer.SimpleBuffer(self.device, 0, hyperparams)

    def get_actions(self, state):
        state_a = torch.FloatTensor(state[0,:]).unsqueeze(0).to(self.device)
        state_b = torch.FloatTensor(state[1,:]).unsqueeze(0).to(self.device)
        actions_a = self.sample_action(self.policy_net_a, state_a)
        actions_b = self.sample_action(self.policy_net_b, state_b)
        return [actions_a, actions_b]

    def sample_action(self, policy_net, state):
        mean, log_std = policy_net.forward(state)
        std = log_std.exp()

        normal = Normal(mean, std)
        z = normal.sample()
        action = torch.tanh(z)
        action = action.cpu().detach().squeeze(0).numpy()

        return action

    def learn_experience(self, experience):
        self.replay_buffer.add(experience)
        if self.replay_buffer.ready_to_sample():
            self.update()

    def update(self):
        states, actions, rewards, next_states, dones = self.replay_buffer.sample()
        dones = dones.view(dones.size(0), -1)

        next_actions_a, next_log_pi_a = self.policy_net_a.sample(next_states[:,0,:])
        next_actions_b, next_log_pi_b = self.policy_net_b.sample(next_states[:,1,:])
        next_actions = torch.cat([next_actions_a, next_actions_b], 1)
        next_log_pi = next_log_pi_a + next_log_pi_b
        flattened_next_states = torch.reshape(next_states, (next_states.shape[0],-1))

        next_q1 = self.target_q_net1(flattened_next_states, next_actions)
        next_q2 = self.target_q_net2(flattened_next_states, next_actions)
        next_q_target = torch.min(next_q1, next_q2) - self.alpha * next_log_pi
        expected_q = (rewards[:,0].unsqueeze(1) + rewards[:,1].unsqueeze(1))+ (1 - dones[:,0].unsqueeze(1)) * self.gamma * next_q_target

        # q loss
        flattened_states = torch.reshape(states, (states.shape[0],-1))
        flattened_actions = torch.reshape(actions, (actions.shape[0],-1))
        curr_q1 = self.q_net1.forward(flattened_states, flattened_actions)
        curr_q2 = self.q_net2.forward(flattened_states, flattened_actions)
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
        new_actions_a, log_pi_a = self.policy_net_a.sample(states[:,0,:])
        new_actions_b, log_pi_b = self.policy_net_b.sample(states[:,1,:])
        new_actions = torch.cat([new_actions_a, new_actions_b], 1)
        if self.update_step % self.delay_step == 0:

            min_q = torch.min(
                self.q_net1.forward(flattened_states, new_actions),
                self.q_net2.forward(flattened_states, new_actions)
            )

            policy_loss = (self.alpha * (log_pi_a + log_pi_b) - min_q).mean()
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()

            # target networks
            for target_param, param in zip(self.target_q_net1.parameters(), self.q_net1.parameters()):
                target_param.data.copy_(self.tau * param + (1 - self.tau) * target_param)

            for target_param, param in zip(self.target_q_net2.parameters(), self.q_net2.parameters()):
                target_param.data.copy_(self.tau * param + (1 - self.tau) * target_param)

        # update temperature
        alpha_loss = (self.log_alpha * (-(log_pi_a + log_pi_b) - self.target_entropy).detach()).mean()

        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()
        self.alpha = self.log_alpha.exp()

        self.update_step += 1

    def get_trained_state_dicts(self):
        return { "policy_net_a" : self.policy_net_a.state_dict(),
                 "policy_net_b" : self.policy_net_b.state_dict() }
