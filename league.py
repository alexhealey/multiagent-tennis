import collections
import importlib
import agent
importlib.reload(agent)
import buffer
import random
import numpy as np

class IndependentPlayLeague:

    def __init__(self, device, env, agent1, agent2, hyperparams):
        self.env = env
        self.max_steps = hyperparams.get("max_steps", 1000)
        self.hyperparams = hyperparams
        self.agent1 = agent1
        self.agent2 = agent2

    def arrange_match(self):
        brain_name = self.env.brain_names[0]
        env_info = self.env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations

        episode_reward1 = 0
        episode_reward2 = 0

        for step in range(self.max_steps):
            actions = [self.agent1.get_action(state[0]), self.agent2.get_action(state[1])]
            env_info = self.env.step(actions)[brain_name]
            reward = env_info.rewards
            next_state = env_info.vector_observations
            done = env_info.local_done
            self.agent1.learn_experience(buffer.Experience(state[0], actions[0], reward[0], next_state[0], done[0]))
            self.agent2.learn_experience(buffer.Experience(state[1], actions[1], reward[1], next_state[1], done[1]))

            episode_reward1 += reward[0]
            episode_reward2 += reward[1]

            if done[0] or done[1] or step == self.max_steps-1:
                break

            state = next_state

        return max(episode_reward1, episode_reward2), step

    def get_trained_state_dicts(self):
        return { "agent1_policy_net" : self.agent1.get_policy_state_dict(),
                "agent2_policy_net" : self.agent2.get_policy_state_dict()}


class SelfPlayLeague:

    def __init__(self, device, env, agent, hyperparams):
        self.env = env
        self.max_steps = hyperparams.get("max_steps", 1000)
        self.hyperparams = hyperparams
        self.agent = agent
        self.episode_count = 0

    def arrange_match(self):
        self.episode_count = self.episode_count + 1
        brain_name = self.env.brain_names[0]
        env_info = self.env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations

        episode_reward1 = 0
        episode_reward2 = 0

        for step in range(self.max_steps):
            actions = [self.agent.get_action(state[0]), self.agent.get_action(state[1])]
            env_info = self.env.step(actions)[brain_name]
            reward = env_info.rewards
            next_state = env_info.vector_observations
            done = env_info.local_done
            self.agent.learn_experience(buffer.Experience(state[0], actions[0], reward[0], next_state[0], done[0]))
            self.agent.learn_experience(buffer.Experience(state[1], actions[1], reward[1], next_state[1], done[1]))

            episode_reward1 += reward[0]
            episode_reward2 += reward[1]

            if done[0] or done[1] or step == self.max_steps-1:
                break

            state = next_state

        return max(episode_reward1, episode_reward2), step

    def get_trained_state_dicts(self):
        return { "agent_policy_net" : self.agent.get_policy_state_dict()}


class SelfPlayFrozenLeague:

    def __init__(self, device, env, agent, hyperparams):
        self.env = env
        self.max_steps = hyperparams.get("max_steps", 1000)
        self.freeze_steps = hyperparams.get("freeze_steps", 5000)
        self.hyperparams = hyperparams
        self.agent = agent
        self.episode_count = 0
        self.next_freeze_agent = 0
        self.steps_till_freeze = self.freeze_steps
        self.self_play_probability = hyperparams.get("self_play_probability", 0.5)
        self.frozen_agents = [agent.create_frozen(hyperparams) for i in range(hyperparams.get("frozen_agents", 5))]

    def arrange_match(self):
        self.episode_count = self.episode_count + 1
        if self.steps_till_freeze < 0:
            self.steps_till_freeze = self.freeze_steps
            self.frozen_agents[self.next_freeze_agent].copy_and_freeze(self.agent)
            self.next_freeze_agent = (self.next_freeze_agent + 1) % len(self.frozen_agents)

        brain_name = self.env.brain_names[0]
        env_info = self.env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations

        episode_reward1 = 0
        episode_reward2 = 0

        if random.random() < self.self_play_probability:
            opponent_agent = self.agent
        else:
            opponent_agent = random.choice(self.frozen_agents)

        for step in range(self.max_steps):
            actions = [self.agent.get_action(state[0]), opponent_agent.get_action(state[1])]
            env_infos = self.env.step(actions)
            reward = env_infos[brain_name].rewards
            next_state = env_infos[brain_name].vector_observations
            done = env_infos[brain_name].local_done
            self.agent.learn_experience(buffer.Experience(state[0], actions[0], reward[0], next_state[0], done[0]))
            self.agent.learn_experience(buffer.Experience(state[1], actions[1], reward[1], next_state[1], done[1]))

            episode_reward1 += reward[0]
            episode_reward2 += reward[1]

            if done[0] or done[1] or step == self.max_steps-1:
                break

            self.steps_till_freeze -= 1
            state = next_state

        return max(episode_reward1, episode_reward2), step

    def get_trained_state_dicts(self):
        return { "policy_net" : self.agent.get_policy_state_dict() }

class SharedCriticLeague:

    def __init__(self, device, env, agent, hyperparams):
        self.env = env
        self.max_steps = hyperparams.get("max_steps", 1000)
        self.hyperparams = hyperparams
        self.agent = agent
        self.episode_count = 0

    def arrange_match(self):
        self.episode_count = self.episode_count + 1
        return self.play_episode()

    def play_episode(self):
        brain_name = self.env.brain_names[0]
        env_info = self.env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations

        episode_reward1 = 0
        episode_reward2 = 0

        for step in range(self.max_steps):
            actions = self.agent.get_actions(state)
            env_info = self.env.step(actions)[brain_name]
            reward = env_info.rewards
            next_state = env_info.vector_observations
            done = env_info.local_done
            self.agent.learn_experience(buffer.Experience(state, actions, reward, next_state, done))

            episode_reward1 += reward[0]
            episode_reward2 += reward[1]

            if done[0] or done[1] or step == self.max_steps-1:
                break

            state = next_state

        return max(episode_reward1, episode_reward2), step

    def get_trained_state_dicts(self):
        return self.agent.get_trained_state_dicts()
