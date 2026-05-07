
import copy
import math
import torch
import torch.nn as nn

from utils.utils import print_rank_0
from utils.ppo_task_policy import TaskSelectionPolicy, PPOBuffer, ppo_update

PPO_MAX_TASKS = 64  # fixed state_dim = PPO_MAX_TASKS * lora_depth, avoids policy reset


# -------------------------------------------------
# KD TREE NODE
# -------------------------------------------------
class KDTreeNode:
    def __init__(self, task_indices, depth, grads_tensor, lora_depth):

        self.task_indices = task_indices
        self.depth = depth
        self.left = None
        self.right = None
        self.is_leaf = False
        self.lora_depth = lora_depth
        self.mean_vector = None
        self.median_similarity = None
        self.num_of_selected = None

        self.build_node(grads_tensor)

    def build_node(self, grads_tensor):

        if self.depth >= self.lora_depth or len(self.task_indices) <= 1:
            self.is_leaf = True
            return

        current_grads = grads_tensor[self.task_indices, self.depth, :]
        self.mean_vector = current_grads.mean(dim=0)

        similarities = torch.mv(current_grads, self.mean_vector)
        self.median_similarity = torch.median(similarities).item()

        left_indices = [
            self.task_indices[i]
            for i in range(len(self.task_indices))
            if similarities[i].item() >= self.median_similarity
        ]
        right_indices = [
            self.task_indices[i]
            for i in range(len(self.task_indices))
            if similarities[i].item() < self.median_similarity
        ]

        if len(left_indices) == 0 or len(right_indices) == 0:
            median = len(self.task_indices) // 2
            left_indices = self.task_indices[:median]
            right_indices = self.task_indices[median:]

        self.left = KDTreeNode(left_indices, self.depth + 1, grads_tensor, self.lora_depth)
        self.right = KDTreeNode(right_indices, self.depth + 1, grads_tensor, self.lora_depth)


# -------------------------------------------------
# REGULARIZATION LOSS
# -------------------------------------------------
def tree_lora_loss(current_grad, all_grad, task_id, prev_id_matrix):

    reg_loss = None

    for depth_id, prev_task_id in enumerate(prev_id_matrix):
        # Ensure prev_task_id is within bounds
        prev_task_id = min(prev_task_id.item() if hasattr(prev_task_id, 'item') else prev_task_id, all_grad.shape[0] - 1)
        term = -(current_grad[depth_id] * all_grad[prev_task_id][depth_id]).sum()
        reg_loss = term if reg_loss is None else reg_loss + term

    return reg_loss


# -------------------------------------------------
# KD LoRA TREE + PPO
# -------------------------------------------------
class KD_LoRA_Tree:

    def __init__(self, args):

        self.args = args
        self.current_grad = None
        self.all_accumulate_grads = [None] * args.num_tasks
        self.all_grad_device = None
        self.sim = None
        self.num_of_selected = None
        self.kd_tree_root = None

        self.use_ppo = getattr(args, "use_ppo", False)
        self._ppo_state_dim = None

        if self.use_ppo:
            self.policy = None
            self.policy_optimizer = None
            self.value_optimizer = None
            self.buffer = PPOBuffer()

    # -------------------------------------------------

    def new_epoch_init(self, train_dataloader_len):

        self.current_grad = None
        self.all_grad = None
        # NOTE: do NOT reset all_grad_device here — it holds previous task grads
        self.tmp_rounds = -1
        self.total_rounds = train_dataloader_len
        self.sim = None

    # -------------------------------------------------

    def step(self):

        self.tmp_rounds += 1
        self.tmp_reg = self.args.reg * self.tmp_rounds / self.total_rounds

    # -------------------------------------------------

    def insert_grad(self, grad):

        if self.current_grad is None:
            self.current_grad = grad.detach() / self.total_rounds
        else:
            self.current_grad += grad.detach() / self.total_rounds

    # -------------------------------------------------

    def _build_sim_matrix(self, task_id, device):
        """
        Step 1 — Eq.(5): Sim[i, d] = -L1(g_t[d], g_i[d])  for i < t.
        Accumulated incrementally via _update_similarity; here we
        initialise on first call using full L1 over stored grads.
        """
        lora_depth = self.all_grad_device.shape[1]
        sim = torch.zeros((task_id, lora_depth), device=device)
        if self.current_grad is not None:
            for i in range(task_id):
                for d in range(lora_depth):
                    sim[i, d] = -torch.sum(
                        torch.abs(self.current_grad[d] - self.all_grad_device[i, d])
                    )
        return sim

    def tree_search(self, task_id, device):

        if self.all_grad_device is None:
            self.all_grad_device = torch.stack(
                self.all_accumulate_grads[:task_id], dim=0
            ).to(device)

        # Step 1: build Sim from L1 distance on first call each epoch
        if self.sim is None:
            self.sim = self._build_sim_matrix(task_id, device)
            self.num_of_selected = torch.zeros(
                (self.args.num_tasks, self.all_grad_device.shape[1]),
                device=device
            )

        sim = self.sim.clone()   # shape: (task_id, lora_depth)

        if self.use_ppo:
            # PPO replaces LCB: normalize raw sim as state
            sim_avg = sim.clone()
            sim_avg = -sim_avg
            sim_avg += torch.min(sim_avg)
            sim_avg = sim_avg / (torch.max(sim_avg) - torch.min(sim_avg) + 1e-5)
        else:
            # LCB exploration bonus (original TreeLoRA)
            valid_mask = self.num_of_selected[:task_id, :] > 0
            sim_avg = sim.clone()
            sim_avg[valid_mask] = sim_avg[valid_mask] / self.num_of_selected[:task_id, :][valid_mask]
            lcb_bonus = (1.0 / torch.sqrt(2 * self.num_of_selected[:task_id, :] + 1e-5)
                         * math.sqrt(math.log(2 * self.total_rounds * (self.tmp_rounds + 1) * (self.tmp_rounds + 2))))
            sim_avg = sim_avg - lcb_bonus
            sim_avg = -sim_avg
            sim_avg += torch.min(sim_avg)
            sim_avg = sim_avg / (torch.max(sim_avg) - torch.min(sim_avg) + 1e-5)

            # Apply KD-tree structure bias
            if hasattr(self, 'kd_tree_root') and self.kd_tree_root is not None and self.kd_tree_root.left is not None:
                first_idx = torch.multinomial(
                    torch.softmax(torch.sum(sim_avg, dim=1), dim=0), num_samples=1
                ).item()
                if first_idx in self.kd_tree_root.left.task_indices:
                    node = self.kd_tree_root.left
                else:
                    node = self.kd_tree_root.right
                scale = min(node.median_similarity, 1.5) if node.median_similarity is not None else 1.0
                sim_avg[node.task_indices] *= scale
                sim_avg = sim_avg / (torch.max(sim_avg) - torch.min(sim_avg) + 1e-5)

        # ---------------- PPO SELECTION ----------------
        if self.use_ppo:

            # Step 1 / Step 6 — s = Flatten(Sim), zero-pad to fixed dim
            raw_state = sim_avg.flatten()
            fixed_dim = PPO_MAX_TASKS * sim_avg.shape[1] if sim_avg.dim() > 1 else PPO_MAX_TASKS
            raw_state = raw_state[:fixed_dim]  # truncate if task_id > PPO_MAX_TASKS
            if raw_state.shape[0] < fixed_dim:
                raw_state = torch.cat([raw_state, torch.zeros(fixed_dim - raw_state.shape[0], device=device)])
            state     = raw_state.unsqueeze(0)   # (1, fixed_dim)
            state_dim = fixed_dim

            # Initialize policy once — fixed dims, no reset across tasks
            if self.policy is None:
                self._ppo_state_dim = state_dim
                self.policy = TaskSelectionPolicy(state_dim, PPO_MAX_TASKS).to(device)
                ppo_lr = getattr(self.args, 'ppo_lr', 3e-4)
                # θ: shared backbone + action head
                self.policy_optimizer = torch.optim.Adam(
                    list(self.policy.fc1.parameters()) +
                    list(self.policy.fc2.parameters()) +
                    list(self.policy.action_head.parameters()),
                    lr=ppo_lr
                )
                # φ: value head only
                self.value_optimizer = torch.optim.Adam(
                    self.policy.value_head.parameters(),
                    lr=ppo_lr
                )

            # Step 2: a ~ Categorical(πθ(s)), mask future tasks
            action, log_prob, value = self.policy.act(state, valid_tasks=task_id)

            prev_id_matrix = action.repeat(sim.shape[1])

            # Step 3+4: hold (s,a,ℓ,V) — r filled in get_loss() then stored
            self._ppo_pending = (state.squeeze(0), action, log_prob, value)

        else:
            sim_soft = torch.softmax(sim_avg, dim=0)
            prev_id_matrix = torch.multinomial(
                sim_soft.T, num_samples=1
            ).reshape(-1)

        # Update selection counts and similarity
        if self.num_of_selected is not None:
            self.num_of_selected[prev_id_matrix, torch.arange(sim.shape[1], device=device)] += 1
        self._update_similarity(prev_id_matrix)

        return prev_id_matrix

    def _update_similarity(self, prev_id_matrix):
        """Incrementally update Sim[i,d] with latest L1 distance (Eq.5)."""
        if self.sim is None or self.current_grad is None:
            return
        for depth_idx, prev_id in enumerate(prev_id_matrix):
            pid = prev_id.item() if hasattr(prev_id, 'item') else prev_id
            self.sim[pid, depth_idx] = -torch.sum(
                torch.abs(self.current_grad[depth_idx] - self.all_grad_device[pid, depth_idx])
            ).item()

    # -------------------------------------------------

    def get_loss(self, grad_current, loss, task_id, prev_id_matrix):

        # Check for NaN in input loss
        if torch.isnan(loss):
            if self.use_ppo and hasattr(self, '_ppo_pending'):
                s, a, lp, v = self._ppo_pending
                self.buffer.store(s, a, lp, v, torch.tensor(0.0, device=loss.device))
                del self._ppo_pending
            return torch.tensor(0.0, device=loss.device, requires_grad=True)

        reg_loss = tree_lora_loss(
            grad_current,
            self.all_grad_device,
            task_id,
            prev_id_matrix
        )

        # Check for NaN in reg_loss
        if torch.isnan(reg_loss):
            if self.use_ppo and hasattr(self, '_ppo_pending'):
                s, a, lp, v = self._ppo_pending
                self.buffer.store(s, a, lp, v, torch.tensor(0.0, device=loss.device))
                del self._ppo_pending
            return torch.tensor(0.0, device=loss.device, requires_grad=True)

        # Normalize reg_loss with better numerical stability
        reg_loss_abs = torch.abs(reg_loss.detach())
        if reg_loss_abs > 1e-8:
            reg_loss = (
                reg_loss / (reg_loss_abs + 1e-8)
                * loss.detach()
                * self.tmp_reg
            )
        else:
            reg_loss = torch.tensor(0.0, device=loss.device, requires_grad=True)

        # Step 3+4: B.store(s, a, ℓ, V, r) — reward = -Lreg as per algorithm
        if self.use_ppo and hasattr(self, '_ppo_pending'):
            s, a, lp, v = self._ppo_pending
            reward = -reg_loss.detach()
            self.buffer.store(s, a, lp, v, reward)
            del self._ppo_pending

        return reg_loss

    # -------------------------------------------------

    def end_task(self, task_id):

        if self.args.reg > 0:
            self.all_accumulate_grads[task_id] = self.current_grad

        # Rebuild KD-tree with all tasks so far
        valid_grads = [
            (i, g) for i, g in enumerate(self.all_accumulate_grads[:task_id + 1])
            if g is not None
        ]
        if len(valid_grads) > 1:
            task_ids = [i for i, _ in valid_grads]
            grads_tensor = torch.stack([g for _, g in valid_grads])
            # difference encoding (same as original)
            for i in range(grads_tensor.shape[0] - 1, 0, -1):
                grads_tensor[i] = grads_tensor[i] - grads_tensor[i - 1]
            lora_depth = grads_tensor.shape[1]
            self.kd_tree_root = KDTreeNode(
                task_indices=task_ids, depth=0,
                grads_tensor=grads_tensor, lora_depth=lora_depth
            )
            print_rank_0(f"KD Tree updated for task {task_id}.", self.args.global_rank)

        # Reset all_grad_device so it's rebuilt next task
        self.all_grad_device = None

        # Phase II: Policy Optimization with separate θ and φ updates
        if self.use_ppo and len(self.buffer.states) > 0:
            ppo_update(self.policy, self.policy_optimizer, self.value_optimizer, self.buffer)
            self.buffer.clear()
