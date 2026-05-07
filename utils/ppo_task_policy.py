import torch
import torch.nn as nn
import torch.nn.functional as F


class TaskSelectionPolicy(nn.Module):
    """
    Actor-Critic network.
    πθ  : fc1 → fc2 → action_head  (shared backbone θ)
    Vφ  : fc1 → fc2 → value_head   (shared backbone φ)
    """
    def __init__(self, state_dim, num_tasks):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.action_head = nn.Linear(128, num_tasks)
        self.value_head  = nn.Linear(128, 1)

    def forward(self, state, valid_tasks=None):
        x      = F.relu(self.fc1(state))
        x      = F.relu(self.fc2(x))
        logits = self.action_head(x)                        # (B, num_tasks)

        # Step 2: mask invalid (future) tasks before softmax
        if valid_tasks is not None:
            mask           = torch.full_like(logits, float('-inf'))
            mask[:, :valid_tasks] = logits[:, :valid_tasks]
            logits         = mask

        probs = F.softmax(logits, dim=-1)
        value = self.value_head(x)
        return probs, value

    def act(self, state, valid_tasks=None):
        """Step 2: a ~ Categorical(πθ(s)), return (a, ℓ, V)."""
        probs, value = self.forward(state, valid_tasks)
        dist         = torch.distributions.Categorical(probs)
        action       = dist.sample()
        log_prob     = dist.log_prob(action)
        return action, log_prob, value.squeeze(-1)


class PPOBuffer:
    """Step 4: B.store(s, a, ℓ, V, r)"""
    def __init__(self):
        self.states    = []
        self.actions   = []
        self.log_probs = []
        self.rewards   = []
        self.values    = []

    def store(self, state, action, log_prob, value, reward):
        self.states.append(state.detach())
        self.actions.append(action.detach())
        self.log_probs.append(log_prob.detach())
        self.values.append(value.detach())
        self.rewards.append(
            reward.detach() if isinstance(reward, torch.Tensor)
            else torch.tensor(reward, dtype=torch.float32, device=state.device)
        )

    def clear(self):
        """Step 7: B.clear()"""
        self.states    = []
        self.actions   = []
        self.log_probs = []
        self.rewards   = []
        self.values    = []


def ppo_update(policy, policy_optimizer, value_optimizer, buffer,
               clip_eps=0.2, epochs=4, value_coef=0.5):
    """
    Phase II — Policy Optimization (Post-Task Update).

    Step 5 : Â_i = r_i - V_old_i, normalize across B
    Step 6 : for k=1..K:
               ρ_i      = exp(log π_new - log π_old)          Eq.(7)
               L_CLIP   = -E[min(ρÂ, clip(ρ,1±ε)Â)]          Eq.(8)
               L_VF     = MSE(V, r)                            Eq.(9)
               L_PPO    = L_CLIP + c·L_VF                      Eq.(10)
               θ ← θ - α_π · Adam(∇_θ L_CLIP)
               φ ← φ - α_v · Adam(∇_φ L_VF)
    Step 7 : B.clear()
    """
    if len(buffer.states) == 0:
        return

    states       = torch.stack(buffer.states)
    actions      = torch.stack(buffer.actions)
    old_log_probs = torch.stack(buffer.log_probs)
    rewards      = torch.stack(buffer.rewards)
    old_values   = torch.stack(buffer.values)

    # Step 5: Advantage Estimation
    advantages = rewards - old_values
    if advantages.std() > 1e-8:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # Step 6: Iterative Policy Refinement
    for _ in range(epochs):
        probs, values    = policy(states)
        dist             = torch.distributions.Categorical(probs)
        new_log_probs    = dist.log_prob(actions)

        # Eq.(7): Importance ratio ρ
        ratio = torch.exp(new_log_probs - old_log_probs)

        # Eq.(8): L_CLIP(θ)
        surr1        = ratio * advantages
        surr2        = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
        policy_loss  = -torch.min(surr1, surr2).mean()

        # Eq.(9): L_VF(φ)
        value_loss   = F.mse_loss(values.squeeze(-1), rewards)

        # Eq.(10): L_PPO = L_CLIP + c·L_VF  (aggregate joint objective)
        total_loss   = policy_loss + value_coef * value_loss  # noqa: logged only

        # θ ← θ - α_π · Adam(∇_θ L_CLIP)  [Step 26]
        policy_optimizer.zero_grad()
        policy_loss.backward(retain_graph=True)
        policy_optimizer.step()

        # φ ← φ - α_v · Adam(∇_φ L_VF)   [Step 27]
        value_optimizer.zero_grad()
        value_loss.backward()
        value_optimizer.step()
