import torch as T
import torch.nn.functional as F
from agent import Agent
T.autograd.set_detect_anomaly(True)
class MADDPG:
    def __init__(self, actor_dims, critic_dims, n_agents, n_actions, 
                 scenario='simple',  alpha=0.01, beta=0.01, fc1=64, 
                 fc2=64, gamma=0.99, tau=0.01, chkpt_dir='tmp/maddpg/'):
        self.agents = []
        self.n_agents = n_agents
        self.n_actions = n_actions
        chkpt_dir += scenario 
        for agent_idx in range(self.n_agents):
            self.agents.append(Agent(actor_dims[agent_idx], critic_dims,  
                            n_actions, n_agents, agent_idx, alpha=alpha, beta=beta,
                            chkpt_dir=chkpt_dir))


    def save_checkpoint(self):
        print('... saving checkpoint ...')
        for agent in self.agents:
            agent.save_models()

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        for agent in self.agents:
            agent.load_models()

    def choose_action(self, raw_obs):
        actions = {}
        agents = ['adversary_0','agent_0','agent_1']
        for agent_idx, agent in enumerate(self.agents):
            arg = agents[agent_idx]
            actions[agents[agent_idx]] = agent.choose_action(raw_obs[agent_idx])
            #actions.append(action)
        return actions

    def learn(self, memory):
        if not memory.ready():
            return

        actor_states, states, actions, rewards, \
        actor_new_states, states_, dones = memory.sample_buffer()

        device = self.agents[0].actor.device

        states = T.tensor(states, dtype=T.float).to(device)
        actions = T.tensor(actions, dtype=T.float).to(device)
        rewards = T.tensor(rewards, dtype=T.float).to(device)
        states_ = T.tensor(states_, dtype=T.float).to(device)
        dones = T.tensor(dones).to(device)

        all_agents_new_actions = []
        all_agents_new_mu_actions = []
        old_agents_actions = []

        for agent_idx, agent in enumerate(self.agents):
            new_states = T.tensor(actor_new_states[agent_idx], 
                                 dtype=T.float).to(device)

            new_pi = agent.target_actor.forward(new_states)

            all_agents_new_actions.append(new_pi)
            mu_states = T.tensor(actor_states[agent_idx], 
                                 dtype=T.float).to(device)
            pi = agent.actor.forward(mu_states)
            all_agents_new_mu_actions.append(pi)
            old_agents_actions.append(actions[agent_idx])

        new_actions = T.cat([acts for acts in all_agents_new_actions], dim=1)
        mu = T.cat([acts for acts in all_agents_new_mu_actions], dim=1)
        old_actions = T.cat([acts for acts in old_agents_actions],dim=1)

        agents = []
        for agent_idx, agent in enumerate(self.agents):
            agents.append(agent)

        ##### primer agente
        critic_value_0_ = agents[0].target_critic.forward(states_, new_actions).flatten().float()
        
        critic_value_0_[dones[:,0]] = 0.0
        critic_value_0_ = agents[0].critic.forward(states,old_actions).flatten()

        target_0 = rewards[:, 0] + agents[0].gamma*critic_value_0_
        critic_loss_0 = F.mse_loss(target_0, critic_value_0_)
        agents[0].critic.optimizer.zero_grad()
        critic_loss_0.backward(retain_graph=True)
        agents[0].critic.optimizer.step()
        agents[0].update_network_parameters()

        #### Segundo agente
        critic_value_1_ = agents[1].target_critic.forward(states_, new_actions).flatten().float()
        
        critic_value_1_[dones[:,0]] = 0.0
        critic_value_1_ = agents[1].critic.forward(states,old_actions).flatten()

        target_1 = rewards[:, 0] + agents[1].gamma*critic_value_1_
        critic_loss_1 = F.mse_loss(target_1, critic_value_1_)
        agents[1].critic.optimizer.zero_grad()
        critic_loss_1.backward(retain_graph=True)
        agents[1].critic.optimizer.step()
        agents[1].update_network_parameters()

        #### Tercer agente
        critic_value_2_ = agents[2].target_critic.forward(states_, new_actions).flatten().float()
        
        critic_value_2_[dones[:,0]] = 0.0
        critic_value_2_ = agents[1].critic.forward(states,old_actions).flatten()

        target_2 = rewards[:, 0] + agents[2].gamma*critic_value_2_
        critic_loss_2 = F.mse_loss(target_2, critic_value_2_)
        agents[2].critic.optimizer.zero_grad()
        critic_loss_2.backward(retain_graph=True)
        agents[2].critic.optimizer.step()
        agents[2].update_network_parameters()



        
