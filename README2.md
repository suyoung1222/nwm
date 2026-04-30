export RESULTS_FOLDER=/home/suyoung/Documents/Git/nwm/logs/results
Checkpoints: in ./logs/nwm_cdit_xl/checkpoints
Dataset folders: /home/suyoung/mydata/NWM

테스트데이터 쓸만한거 recon, scand

(gt preparation, one-time)
python isolated_nwm_infer.py     --exp config/nwm_cdit_xl.yaml     --datasets limo1     --batch_size 96     --num_workers 12     --eval_type time     --output_dir ${RESULTS_FOLDER}     --gt 1

test
python isolated_nwm_infer.py \
    --exp config/nwm_cdit_xl.yaml \
    --ckp cdit_xl_ego4d_200000 \
    --datasets limo1 \
    --batch_size 16 \
    --num_workers 12 \
    --eval_type rollout \
    --output_dir ${RESULTS_FOLDER}

test custom dataset 
/home/suyoung/Documents/limo/agilex_open_class/limo/limo_gazebo_sim/scripts/dataset_mocap/realsense_1/nwm_dataset/limo2_bag
torchrun --nproc_per_node=1 isolated_nwm_infer.py   --exp config/nwm_cdit_xl.yaml   --datasets limo2   --eval_type rollout   --output_dir ./logs/results/limo2
python isolated_nwm_infer.py \
    --exp config/nwm_cdit_xl.yaml \
    --ckp cdit_xl_ego4d_200000 \
    --datasets limo2 \
    --batch_size 16 \
    --num_workers 12 \
    --eval_type time \
    --output_dir ./logs/results/limo2

eval metric result (after gt generation, test)
python isolated_nwm_eval.py \
        --datasets limo2 \
        --gt_dir ${RESULTS_FOLDER}/gt \
        --exp_dir ${RESULTS_FOLDER}/nwm_cdit_xl_cdit_xl_ego4d_200000 \
        --eval_types rollout

결론:  Recon데이터셋만갖고 one step prediction, trajectory eval, planning eval을 분석하자
cuda 메모리 이슈때문에 더 적은 배치사이즈로 실행시켜야함.
그렇게 하고 트레이닝 데이터가 아닌 주행 데이터 결과가 어떤지 비교해보자 (내가 과거에 쓴 가제보나 아웃도어 이미지 등)

rollout limo2 돌린거 gt랑 비교해서 metric결과 확인해보기
똑같은짓 gazebo(limo1,limo5)에서도 돌려보기 -여기까지 완! (사실 도는중이니까 집가서 확인 꼭)
gazebo결과 보고 시뮬레이션 정해서 (gazebo, phoenix, habitat등) 골라서 planning 코드짜기
active mapping베이스라인 코드 정해야할텐ㄷ...
왜 context window size is 4? why not longer?
깃푸씨


# Navigation World Models (NWM) — A Complete Technical Guide

**Paper:** Bar, Zhou, Tran, Darrell, LeCun. *Navigation World Models.* CVPR 2025 Oral (arXiv:2412.03572).
**Code:** `github.com/facebookresearch/nwm`


## 1. What NWM is — and what it is not

**NoMaD is a policy** — it maps observations directly to actions. It cannot imagine or evaluate trajectories; it just produces them.

**NWM is a world model** — it maps observations + *candidate* actions to *predicted future observations*. It cannot emit an action on its own. To navigate, NWM has to be paired with an action-search procedure (CEM) or an external policy (NoMaD) that proposes candidate action sequences — then NWM scores each one by generating the video that would result and comparing the final frame to the goal image.

Conceptually:
- NoMaD: $a \sim \pi(a \mid o_t, o_g)$ — one shot, no imagination.
- NWM: $\hat o_{t+1} \sim F_\theta(\hat o_{t+1} \mid o_{t-m:t}, a_t)$ — imagine the future, then planning happens around this.

The payoff: NWM decouples *world dynamics* from *policy*. You can now (a) plan from scratch with no policy, (b) rank any external policy's samples, (c) enforce hard constraints like "no left turns" that a trained policy cannot honour, and (d) train on *unlabeled* navigation video (Ego4D) to improve generalization. None of this is natural for a behavior-cloning policy like NoMaD.

The price: you are running a diffusion video model at planning time — 250 denoising steps × 8 rollout frames × 120 candidate trajectories. This is *expensive*, and the paper spends an entire section on how to make it real-time.

---

## 2. Mathematical formulation

### 2.1 Data and notation

A trajectory is a sequence of (image, action) pairs:
$$
\mathcal{D} = \{(x_0, a_0, x_1, a_1, \dots, x_T, a_T)\}_i
$$
with $x_i \in \mathbb{R}^{H \times W \times 3}$ (RGB) and $a_i = (u_i, \phi_i)$ where $u_i \in \mathbb{R}^2$ is planar translation and $\phi_i \in \mathbb{R}$ is yaw rotation. This is **3-DoF** (flat floor, no pitch/roll), which matches the wheeled-robot datasets (RECON / SCAND / TartanDrive / HuRoN / GoStanford).

Images are encoded to latents with a **frozen Stable-Diffusion VAE**:
$$
s_i = \text{enc}_\theta(x_i) \in \mathbb{R}^{4 \times H/8 \times W/8}
$$
and the model operates in latent space throughout; only the final output is decoded back to pixels for visualization/scoring. For $224 \times 224$ images this gives $28 \times 28$ spatial latents; patched at patch size 2, that's $14 \times 14 = 196$ tokens per frame.

### 2.2 The world-model objective (Eq. 1 in the paper)

Learn a stochastic map
$$
s_{\tau+1} \sim F_\theta(s_{\tau+1} \mid \mathbf{s}_\tau, a_\tau)
$$
where $\mathbf{s}_\tau = (s_\tau, s_{\tau-1}, \dots, s_{\tau-m+1})$ is a **context** of $m$ past latents.

### 2.3 The crucial extension — time shift $k$ (Eq. 2)

A vanilla next-frame model predicts $s_{\tau+1}$ from $s_\tau + a_\tau$. NWM extends this by adding a **temporal shift** $k \in [T_\text{min}, T_\text{max}]$ to the action:
$$
a_\tau = (u, \phi, k).
$$
The model is asked to predict the frame *$k$ steps into the future* (or past). The action $(u, \phi)$ is treated as the integrated motion over that window:
$$
u_{\tau \to \tau+k-1} = \sum_{t=\tau}^{\tau+k-1} u_t, \qquad
\phi_{\tau \to \tau+k-1} = \Bigl(\sum_{t=\tau}^{\tau+k-1} \phi_t\Bigr) \bmod 2\pi.
$$
Why this matters:
- One model, any lookahead. You train once and get a range of temporal resolutions for free at inference.
- Planning doesn't have to roll out frame-by-frame — a coarse "8-step plan at $k=4$" is dramatically cheaper than 32 one-step predictions and often more accurate (less error accumulation).
- It lets you train on **action-unlabeled video** like Ego4D by setting $u = \phi = 0$ and only using $k$.

In practice the paper caps $|k| \le 16$ s.

### 2.4 Diffusion process (DDPM)

**Forward:** $s^{(t)}_{\tau+1} = \sqrt{\bar\alpha_t}\,s_{\tau+1} + \sqrt{1-\bar\alpha_t}\,\epsilon$, $\epsilon \sim \mathcal{N}(0,I)$, with a **linear** $\beta$ schedule and $T = 1000$ training steps (DiT defaults).

**Reverse** (parameterized by the network):
$$
p_\theta\big(s^{(t-1)}_{\tau+1} \mid s^{(t)}_{\tau+1}, \mathbf{s}_\tau, a_\tau, t\big).
$$
**Training loss** (Eq. "$\mathcal{L}_\text{simple}$" in the paper):
$$
\mathcal{L}_\text{simple} = \mathbb{E}_{s_{\tau+1}, a_\tau, \mathbf{s}_\tau, \epsilon, t}\big\|s_{\tau+1} - F_\theta(s^{(t)}_{\tau+1} \mid \mathbf{s}_\tau, a_\tau, t)\big\|_2^2
$$

Following DiT, the network also predicts log-variance range with an additional **variational lower-bound** term $\mathcal{L}_\text{vlb}$ (Nichol & Dhariwal 2021), so the total per-sample loss is $\mathcal{L}_\text{simple} + \mathcal{L}_\text{vlb}$. This is why the CDiT output channels are `in_channels × 2 = 8` when `learn_sigma=True`.

### 2.5 Planning with a learned world model (Eq. 4–5)

Given context $s_0$ and goal latent $s^*$, find actions $(a_0, \dots, a_{T-1})$ that minimize
$$
\mathcal{E}(s_0, a_{0:T-1}, s_T) = -\mathcal{S}(s_T, s^*) + \sum_{\tau} \mathbb{I}[a_\tau \notin \mathcal{A}_\text{valid}] + \sum_{\tau} \mathbb{I}[s_\tau \notin \mathcal{S}_\text{safe}]
$$
where $s_T$ is the last state of the NWM rollout, $\mathcal{S}(\cdot, \cdot)$ is LPIPS perceptual similarity (in pixel space after VAE decode), and the indicator terms add infinite penalties for constraint violations (e.g., no-left-turns, no going off a cliff).

The optimization is solved by the **Cross-Entropy Method** (CEM) — a derivative-free population-based optimizer — because the expectation is over a stochastic rollout and the LPIPS score is non-differentiable as wired up. See §5.2 for the full algorithm.

---

## 3. Architecture — Conditional Diffusion Transformer (CDiT)

File: `models.py`. The entry-point class is `CDiT`; the key innovation is a custom transformer block `CDiTBlock`.

### 3.1 Why CDiT instead of DiT

A standard DiT feeds all tokens from all context frames + the target frame into a single self-attention stack. With $m$ frames and $n$ tokens per frame, attention cost is $O(m^2 n^2 d)$ — **quadratic in context length**. For 4 context frames at 196 tokens that's $5^2 \cdot 196^2 d = 4.8\text{M} \cdot d$ per attention call.

CDiT's move: **never let context tokens attend to each other**. The three-attention-layer pattern in each block is:
1. **Self-attention** — only over the target frame's 196 tokens. Cost: $O(n^2 d)$.
2. **Cross-attention** — target tokens (queries) attending to context tokens (keys/values). Cost: $O(m n^2 d)$ — linear in $m$.
3. **MLP** — as usual.

Total per block: $O(m n^2 d)$ — linear in context. Paper Fig. 5 shows CDiT-L at 1 TFLOP matches DiT-XL at 4 TFLOPs (4× speedup) and CDiT-XL at 1B params outperforms DiT-XL on LPIPS. This matters because for planning you run the model hundreds of times per scene — linear scaling is the difference between "feasible" and "not".

### 3.2 The CDiTBlock, annotated

```python
class CDiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)  # self-attn, TARGET ONLY
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm_cond = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.cttn = nn.MultiheadAttention(hidden_size, num_heads=num_heads,
                                          add_bias_kv=True, bias=True, batch_first=True)  # cross-attn
        # adaLN produces 11 modulation vectors per block:
        #   (shift,scale,gate) for self-attn       -> 3
        #   (shift,scale) for context side of cross-attn -> 2
        #   (shift,scale,gate) for target side of cross-attn -> 3
        #   (shift,scale,gate) for MLP            -> 3
        # total = 11
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 11 * hidden_size, bias=True)
        )
        self.norm3 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp = Mlp(in_features=hidden_size,
                       hidden_features=int(hidden_size*mlp_ratio),
                       act_layer=lambda: nn.GELU(approximate="tanh"), drop=0)

    def forward(self, x, c, x_cond):
        # x       : (B, n, d)         - target frame tokens (being denoised)
        # c       : (B, d)            - conditioning vector (action + rel_t + diff. timestep)
        # x_cond  : (B, m*n, d)       - flattened context tokens

        (shift_msa, scale_msa, gate_msa,
         shift_ca_xcond, scale_ca_xcond,
         shift_ca_x,     scale_ca_x,  gate_ca_x,
         shift_mlp, scale_mlp, gate_mlp) = self.adaLN_modulation(c).chunk(11, dim=1)

        # (1) Self-attention over the target frame's tokens only
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))

        # (2) Cross-attention: target (Q) attends to context (K, V)
        x_cond_norm = modulate(self.norm_cond(x_cond), shift_ca_xcond, scale_ca_xcond)
        x = x + gate_ca_x.unsqueeze(1) * self.cttn(
                query=modulate(self.norm2(x), shift_ca_x, scale_ca_x),
                key=x_cond_norm, value=x_cond_norm, need_weights=False)[0]

        # (3) MLP
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm3(x), shift_mlp, scale_mlp))
        return x
```

Two non-obvious details worth flagging:
- `modulate(h, shift, scale) = h * (1 + scale) + shift` — AdaLN-Zero from DiT. Initialized to zero so every block starts as an identity map and learns its contribution from scratch.
- The context gets its **own** shift/scale (`shift_ca_xcond`, `scale_ca_xcond`) before being used as keys/values. The target side of the cross-attn gets a different (shift, scale, gate). This lets $c$ modulate "what to pay attention to in the past" *independently* of "how to update the target". A vanilla cross-attn implementation wouldn't do this.

### 3.3 Conditioning vector $c$ (Eq. 3)

```python
class ActionEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        hsize = hidden_size // 3
        self.x_emb     = TimestepEmbedder(hsize, frequency_embedding_size)
        self.y_emb     = TimestepEmbedder(hsize, frequency_embedding_size)
        self.angle_emb = TimestepEmbedder(hidden_size - 2*hsize, frequency_embedding_size)
    def forward(self, xya):
        return torch.cat([self.x_emb(xya[...,0:1]),
                          self.y_emb(xya[...,1:2]),
                          self.angle_emb(xya[...,2:3])], dim=-1)
```
Each of $(\Delta x, \Delta y, \Delta \phi)$ is embedded **separately** via sinusoidal features + a 2-layer MLP, then concatenated. Splitting the embedding dim 3 ways (roughly $d/3$ each, with `angle_emb` taking the slack) prevents the model from confusing translation magnitude with angle magnitude.

In `CDiT.forward`:
```python
t = self.t_embedder(t[..., None])            # diffusion timestep -> ψ_t
y = self.y_embedder(y)                       # (Δx,Δy,Δφ) -> ψ_a
time_emb = self.time_embedder(rel_t[..., None])  # k/128 -> ψ_k
c = t + time_emb + y                         # Eq. 3: ξ = ψ_a + ψ_k + ψ_t
```
All three sum into one vector, which then drives AdaLN in every block. **To train on unlabeled data (Ego4D) you simply omit `y`** — the model still has $t$ and $k$ and learns a time-only (and imagination-driven) dynamics prior.

### 3.4 Putting the full forward together

```python
def forward(self, x, t, y, x_cond, rel_t):
    # Patchify target + add position embedding for "target" row
    x = self.x_embedder(x) + self.pos_embed[self.context_size:]       # (B, n, d)
    # Patchify each context frame; pos_embed rows [0 : context_size] are shared for them
    x_cond = self.x_embedder(x_cond.flatten(0,1)).unflatten(0, (x_cond.shape[0], x_cond.shape[1])) \
             + self.pos_embed[:self.context_size]                     # (B, m, n, d)
    x_cond = x_cond.flatten(1, 2)                                     # (B, m*n, d)

    t = self.t_embedder(t[..., None])
    y = self.y_embedder(y)
    time_emb = self.time_embedder(rel_t[..., None])
    c = t + time_emb + y

    for block in self.blocks:
        x = block(x, c, x_cond)
    x = self.final_layer(x, c)
    x = self.unpatchify(x)                 # (B, out_channels, H_lat, W_lat)
    return x
```

`pos_embed` is a **learned** parameter of shape `(context_size + 1, num_patches, d)`. Note that all context frames share the same spatial pos embedding **row**, and the target gets a distinct row — the model knows "which frame am I looking at" through this simple indexing. Temporal ordering within the context has to be learned by the model from the patterns in $c$ (the action embedding encodes direction of motion).

### 3.5 Size ladder

From the code (`CDiT_models` dict):

| Name | depth | hidden | heads | params |
|---|---|---|---|---|
| CDiT-S/2 | 12 | 384 | 6 | ~35 M |
| CDiT-B/2 | 12 | 768 | 12 | ~130 M |
| CDiT-L/2 | 24 | 1024 | 16 | ~450 M |
| **CDiT-XL/2** | **28** | **1152** | **16** | **~1 B** |

The paper's headline results use CDiT-XL/2.

---

## 4. Training pipeline

Files: `train.py` → `diffusion/gaussian_diffusion.py::training_losses`.

### 4.1 The per-batch math

```python
# --- Read batch from TrainingDataset ---
# x : (B, T, 3, H, W)  context + goal images stacked
# y : (B, num_goals, 3)  (Δx, Δy, Δφ) normalized
# rel_t : (B, num_goals) time shift, normalized by /128

with torch.amp.autocast('cuda', enabled=True, dtype=torch.bfloat16):
    # (1) Encode every frame via VAE — move to latent space
    B, T = x.shape[:2]
    x = x.flatten(0, 1)
    x = tokenizer.encode(x).latent_dist.sample().mul_(0.18215)   # SD VAE scale factor
    x = x.unflatten(0, (B, T))                                    # (B, T, 4, H/8, W/8)

    # (2) Split context / goals
    num_cond = config['context_size']     # 4
    num_goals = T - num_cond              # 4 (multi-goal training!)
    x_start = x[:, num_cond:].flatten(0, 1)  # (B*num_goals, 4, Hl, Wl)  -> target
    x_cond  = x[:, :num_cond] \
              .unsqueeze(1).expand(B, num_goals, num_cond, 4, Hl, Wl) \
              .flatten(0, 1)              # (B*num_goals, num_cond, 4, Hl, Wl) -> context
    y      = y.flatten(0, 1)
    rel_t  = rel_t.flatten(0, 1)

    # (3) Sample a random diffusion timestep for each example
    t = torch.randint(0, diffusion.num_timesteps, (x_start.shape[0],), device=device)

    # (4) Compute loss = simple MSE + VLB for sigma
    loss_dict = diffusion.training_losses(
        model, x_start, t, model_kwargs=dict(y=y, x_cond=x_cond, rel_t=rel_t)
    )
    loss = loss_dict['loss'].mean()
```

**Why "num_goals = 4" with the same context:** the dataset returns `goals_per_obs=4` — i.e., for the same 4-frame context, it samples **4 different future frames** at **4 different time offsets $k$**. The training loop unfolds them, effectively multiplying the batch by 4. Paper Table 1 shows going from 1 → 4 goals drops LPIPS ~0.016 (with everything else fixed) — the multi-goal schedule is one of the biggest tricks for avoiding the "action/time entanglement" problem the authors discuss in §3.1 (if every time you've gone to this spot it was at time t=5, the model might collapse to predicting from $k$ alone and ignore $u, \phi$; multi-goal creates counterfactuals).

### 4.2 Inside `training_losses`

From `diffusion/gaussian_diffusion.py`:
```python
def training_losses(self, model, x_start, t, model_kwargs=None, noise=None):
    if noise is None:
        noise = torch.randn_like(x_start)
    x_t = self.q_sample(x_start, t, noise=noise)                # forward process
    model_output = model(x_t, t, **model_kwargs)                # CDiT forward

    # With LEARNED_RANGE variance, output channels = 2*C; split into mean-pred and var-pred
    B, C = x_t.shape[:2]
    model_output, model_var_values = torch.split(model_output, C, dim=1)

    # VLB branch — trains the variance head, detached from mean branch
    frozen_out = torch.cat([model_output.detach(), model_var_values], dim=1)
    terms['vb'] = self._vb_terms_bpd(model=lambda *a, r=frozen_out: r,
                                     x_start=x_start, x_t=x_t, t=t, clip_denoised=False)['output']

    # MSE branch — trains the mean prediction (ε)
    target = noise                              # predict_xstart=False -> ε-prediction
    terms['mse'] = mean_flat((target - model_output) ** 2)
    terms['loss'] = terms['mse'] + terms['vb']
```
Two branches are trained jointly but their gradients are **decoupled** via the `.detach()` — the VLB term trains `model_var_values` but doesn't perturb the mean prediction. This is standard DiT practice (Peebles & Xie 2023); it stabilizes training of the sigma head.

### 4.3 Key hyper-parameters (from `config/nwm_cdit_xl.yaml`)

| Setting | Value | Paper section |
|---|---|---|
| Model | CDiT-XL/2 | §4.1 |
| Params | ~1 B | §4.1 |
| Batch size per GPU | 16 | §4.1 (total batch 1024 × 4 goals = 4096) |
| Learning rate | 8e-5 (AdamW) | §4.1 |
| Context size $m$ | 4 frames | ablation in §4.2 |
| Image size | 224 × 224 (28 × 28 latent) | §4.1 |
| `len_traj_pred` | 64 | — |
| `min/max_dist_cat` | ±64 | ±16 s window at 4 FPS |
| Diffusion steps (train) | 1000 | DiT default |
| Noise schedule | linear | DiT default |
| `learn_sigma` | True | DiT default |
| Precision | bfloat16 (autocast) | §4.1 |
| Training | 300 epochs × 8×8 H100s | §4.1 |
| EMA decay | 0.9999 | `train.py::update_ema` |

The `rel_t` field in the dataset is the time offset *normalized by 128*:
```python
goal_offset = np.random.randint(min_goal_dist, max_goal_dist + 1, size=(goals_per_obs))
goal_time   = curr_time + goal_offset
rel_time    = goal_offset / 128.0    # TODO (from the code): this 128 is currently a fixed const
```
So `rel_t = 1/128` in inference (`model_forward_wrapper`) means "one training step ahead." This normalization constant is hard-coded — if you train on a dataset with very different frame rates you'd want to revisit it.

---

## 5. Navigation planning (**the answer to your question**)

NWM by itself is a video model. Planning comes from wrapping the rollout in an outer optimization loop. Two modes are implemented; §5.2 is the interesting one.

### 5.1 Autoregressive rollout — the "simulate a trajectory" primitive

Both planning modes boil down to the same inner operation:

> Given a starting context of $m$ past frames and a sequence of $T$ actions, predict the $T$ resulting future frames.

Implemented in `planning_eval.py::autoregressive_rollout` and the parallel `isolated_nwm_infer.py::generate_rollout`:

```python
def autoregressive_rollout(self, obs_image, deltas, rollout_stride):
    # deltas: (N, T*rollout_stride, 3)  — one (Δx,Δy,Δφ) per step
    # Collapse groups of rollout_stride steps into one (via Eq. 2: sum translations, sum angles)
    deltas = deltas.unflatten(1, (-1, rollout_stride)).sum(2)   # (N, T, 3)

    preds = []
    curr_obs = obs_image.clone().to(self.device)                # (N, m, 3, H, W)

    for i in range(deltas.shape[1]):
        curr_delta = deltas[:, i:i+1]                          # (N, 1, 3)
        all_models = self.model, self.diffusion, self.vae
        # One forward pass -> one new frame
        x_pred_pixels = model_forward_wrapper(
            all_models, curr_obs, curr_delta,
            num_timesteps=rollout_stride,                      # this is the time shift k !
            latent_size=self.latent_size,
            num_cond=self.num_cond, device=self.device)
        x_pred_pixels = x_pred_pixels.unsqueeze(1)             # (N, 1, 3, H, W)

        # Slide the context window forward
        curr_obs = torch.cat((curr_obs, x_pred_pixels), dim=1)  # append
        curr_obs = curr_obs[:, 1:]                              # drop oldest
        preds.append(x_pred_pixels)

    return torch.cat(preds, 1)   # (N, T, 3, H, W)
```

And `model_forward_wrapper` (`isolated_nwm_infer.py`):

```python
def model_forward_wrapper(all_models, curr_obs, curr_delta, num_timesteps, latent_size,
                          device, num_cond, num_goals=1, rel_t=None, progress=False):
    model, diffusion, vae = all_models
    x, y = curr_obs.to(device), curr_delta.to(device)

    with torch.amp.autocast('cuda', enabled=True, dtype=torch.bfloat16):
        B, T = x.shape[:2]

        # rel_t = num_timesteps / 128   (matches training normalization)
        if rel_t is None:
            rel_t = (torch.ones(B) * (1. / 128.)).to(device)
            rel_t *= num_timesteps

        # (1) Encode current context into latent space
        x = x.flatten(0,1)
        x = vae.encode(x).latent_dist.sample().mul_(0.18215).unflatten(0, (B, T))
        x_cond = x[:, :num_cond].unsqueeze(1)\
                  .expand(B, num_goals, num_cond, *x.shape[2:]).flatten(0, 1)

        # (2) Start from Gaussian noise in LATENT space (NOT pixel space)
        z = torch.randn(B*num_goals, 4, latent_size, latent_size, device=device)
        y = y.flatten(0, 1)

        # (3) Run 250-step DDPM denoising (respaced from 1000)
        model_kwargs = dict(y=y, x_cond=x_cond, rel_t=rel_t)
        samples = diffusion.p_sample_loop(
            model.forward, z.shape, z,
            clip_denoised=False, model_kwargs=model_kwargs,
            progress=progress, device=device
        )

        # (4) Decode back to pixel space (needed for LPIPS scoring)
        samples = vae.decode(samples / 0.18215).sample
        return torch.clip(samples, -1., 1.)
```

Key observations:
1. **Diffusion happens in latent space.** The VAE encode/decode wraps every step. The reason: `z` is `(B, 4, 28, 28)` — 4-channel latent — which is 48× smaller than the `(B, 3, 224, 224)` RGB. One full rollout needs $T=8$ frames × 250 denoising steps = 2000 forward passes of CDiT; doing that in pixel space would be ruinous.
2. **The time shift `rel_t` is set from `rollout_stride`.** If `rollout_stride=4`, each NWM call jumps 4 training-frames ahead and the `deltas` are pre-summed over 4 steps — so 32 ground-truth frames become 8 NWM calls. The paper Table 8 shows this "time skip" cuts runtime from 30 s to 15 s with no navigation-quality drop.
3. **`clip_denoised=False`.** Unlike NoMaD (`clip_sample=True`), here you *must not* clip the intermediate denoised sample, because it's a latent, not RGB. A VAE latent can have values well outside [-1, 1]; clipping would destroy the reconstruction.
4. **The context slides every step.** The model is trained to see $m=4$ frames at a time, so during an 8-step rollout, frames 0–3 go into the first call, then (1,2,3,pred0) into the second, and so on. Errors compound — Fig. 4 shows LPIPS going from 0.30 at $t=1$ s to 0.60 at $t=16$ s because of this.

### 5.2 Standalone planning — CEM (the paper's Eq. 5 in code)

File: `planning_eval.py::generate_actions`. This is the "NWM as a planner, no external policy" setting.

The search space is **not** the full $T$-step action sequence. The authors assume a *straight-line trajectory parameterized by its endpoint*: optimize only $(\Delta x, \Delta y, \Delta \phi)$ — 3 numbers — then map to 8 evenly-spaced delta steps, with the full yaw rotation applied at the final step. This is a huge dimensionality reduction (3 vars instead of 24 for an 8-step path) and is the reason a single CEM iteration with 120 samples is enough.

```python
def init_mu_sigma(self, obs_0, traj_len):
    n_evals = obs_0.shape[0]
    mu    = torch.zeros(n_evals, self.action_dim)  # 3-dim
    mu[:] = torch.tensor(data_hyperparams[self.args.datasets]['mu'])        # dataset-specific prior
    sigma = torch.ones(n_evals, self.action_dim)
    sigma[:] = torch.tensor(data_hyperparams[self.args.datasets]['var_scale'])
    return mu, sigma
```

From `config/data_hyperparams_plan.yaml`:
```yaml
recon:        {mu: [-0.1,  0, 0], var_scale: [0.02, 0.1, 0.1]}
tartan_drive: {mu: [ 0.5,  0, 0], var_scale: [0.07, 0.1, 0.1]}
scand:        {mu: [-0.25, 0, 0], var_scale: [0.04, 0.1, 0.1]}
sacson:       {mu: [-0.33, 0, 0], var_scale: [0.03, 0.1, 0.1]}
```
Each dataset has a different walking/driving speed, so the prior for $\Delta x$ (the forward component) differs. Without this dataset-dependent prior, CEM with a single iteration wouldn't converge — it's the "warm start" that makes it work.

The CEM iteration itself:

```python
for i in range(self.opt_steps):                                       # default 15, paper uses 1
    for traj in range(n_evals):                                       # one trajectory at a time
        # (1) SAMPLE N candidates from current Gaussian
        sample = torch.randn(self.num_samples, 3) * sigma[traj] + mu[traj]     # (N, 3)

        # (2) Build 8-step action sequences from each endpoint
        single_delta = sample[:, :2]                                             # (N, 2)
        deltas = single_delta.unsqueeze(1).repeat(1, len_traj_pred, 1)           # (N, 8, 2) — EVENLY SPACED
        unnorm_deltas = unnormalize_data(deltas, ACTION_STATS_TORCH)
        delta_yaw = calculate_delta_yaw(unnorm_deltas)                           # per-step yaw from Δx,Δy
        deltas = torch.cat((deltas, delta_yaw), dim=-1)                          # (N, 8, 3)
        deltas[:, -1, -1] += sample[:, -1] * np.pi                               # final big yaw at last step

        # (3) ROLL OUT each candidate through NWM
        cur_obs_image  = obs_image[traj:traj+1].repeat(self.num_samples, 1, 1, 1, 1)
        cur_goal_image = goal_image[traj:traj+1].repeat(self.num_samples, 1, 1, 1).squeeze(1)
        # Because NWM is stochastic, repeat each rollout num_repeat_eval=3 times and average
        expanded_deltas     = deltas.repeat(self.num_repeat_eval, 1, 1)
        expanded_obs_image  = cur_obs_image.repeat(self.num_repeat_eval, 1, 1, 1, 1)
        expanded_goal_image = cur_goal_image.repeat(self.num_repeat_eval, 1, 1, 1)
        preds = self.autoregressive_rollout(expanded_obs_image, expanded_deltas, self.args.rollout_stride)
        preds = preds[:, -1]                                         # take LAST predicted frame only

        # (4) SCORE via LPIPS to the goal image
        loss = self.loss_fn(preds, expanded_goal_image).flatten(0)  # LPIPS (AlexNet backbone)
        loss = loss.view(self.num_repeat_eval, -1).mean(dim=0)       # average over stochastic runs

        # (5) SELECT top-K and REFIT Gaussian
        sorted_idx  = torch.argsort(loss)
        topk_idx    = sorted_idx[:self.topk]                          # default 5
        topk_action = deltas[topk_idx][:, -1]                         # last-step action of the winner
        mu[traj]    = topk_action.mean(dim=0)
        sigma[traj] = topk_action.std(dim=0)
```

After the loop, the final trajectory is constructed from the converged $\mu$ and rolled out one more time — that's what goes to ATE/RPE evaluation.

**Constraint handling** is stupid-simple in the code: "parts of the trajectory are zeroed out to respect constraints." For `forward-first` (move straight 5 steps, then turn 3), you literally set `delta_yaw[:, :5] = 0` before the rollout, and CEM only optimizes over the remaining degrees of freedom. Because CEM is derivative-free, it doesn't care that you clipped — it just tries to find the best feasible trajectory. The indicator-function formulation in Eq. 4 is conceptual; in practice, constraints are enforced by *masking the action tensor*.

#### Why this works at all: a visual intuition

At each CEM step, you draw 120 candidate endpoints from a Gaussian centered on "probably the right direction." For each, NWM dreams up 8 future frames ending where that endpoint would take you. If your candidate is wrong, the dreamed last frame will look nothing like the goal (high LPIPS). If your candidate is close, the last frame will perceptually match the goal. The Gaussian refits toward the cluster of candidates that produced goal-matching imaginings. After even a single iteration, you've effectively searched the local neighborhood of plausible trajectories and picked the best.

### 5.3 Mode 2: Ranking an external policy (NoMaD)

This is architecturally simpler and uses NoMaD as the trajectory proposer:

```
1. Sample n ∈ {16, 32} trajectories from NoMaD(context, goal).
2. For each trajectory:
   - Roll it out through NWM -> final predicted frame.
   - Score LPIPS(final_frame, goal_image).
3. Return the trajectory with minimum LPIPS.
```

Not in the released code directly (the repo ships standalone planning only), but easily reconstructed: the same `autoregressive_rollout` + LPIPS scoring, with `deltas` coming from a NoMaD policy instead of a CEM distribution. Paper Table 2 shows this bumps NoMaD's RECON ATE from 1.93 → 1.78 with 32 samples — a modest but consistent gain across all datasets.

### 5.4 Reading the results (paper Tables 2 & 7)

| Method | RECON ATE↓ | RECON RPE↓ |
|---|---|---|
| GNM (no diffusion) | 1.87 | 0.73 |
| NoMaD | 1.93 | 0.52 |
| NWM + NoMaD ×16 | 1.83 | 0.50 |
| NWM + NoMaD ×32 | 1.78 | 0.48 |
| **NWM (standalone planning)** | **1.13** | **0.35** |

The standalone NWM planner *dominates* on RECON because RECON is an in-domain dataset with well-tuned CEM priors. On TartanDrive (forward-heavy) the advantage narrows because "always drive forward" is already near-optimal. Planning with a world model is only as good as (a) the world model's prediction accuracy and (b) the prior you hand to CEM.

---

## 6. End-to-end code flow

```
                             ┌──────────── TRAINING ────────────┐
                             │                                  │
   TrajectoryDataset        │  TrainingDataset                  │
   (4 ctx + 4 goals          │   - 4 goal frames at random       │
    per example)             │     Δt ∈ [-64, +64]               │
                             │   - rel_t = Δt / 128              │
                             │                                  │
                             ▼                                  │
                    ┌────────────────────┐                      │
                    │ SD-VAE encode      │   frozen, 0.18215    │
                    │ (x → z) ×T frames  │   scale factor       │
                    └────────────────────┘                      │
                             │                                  │
                             ▼                                  │
                    ┌────────────────────┐                      │
                    │  q_sample(z_t, t)  │   DDPM forward       │
                    │  add noise at t    │   1000 steps, linear │
                    └────────────────────┘                      │
                             │                                  │
                             ▼                                  │
   ┌───────────────────────────────────────────────────────┐    │
   │ CDiT-XL/2  forward(x=z_t, t, y=(Δx,Δy,Δφ),           │    │
   │                    x_cond=context_latents, rel_t=k/128)│    │
   │                                                       │    │
   │   x_emb + pos_embed[ctx:] ─────────────────┐          │    │
   │   x_cond_emb + pos_embed[:ctx] ────────────┤          │    │
   │                                            ▼          │    │
   │   c = ψ_t + ψ_k + ψ_a (AdaLN conditioning)            │    │
   │                                            │          │    │
   │   28 × CDiTBlock(x, c, x_cond)                        │    │
   │      ├─ Self-Attn (target tokens only)                │    │
   │      ├─ Cross-Attn (target queries, ctx K/V)          │    │
   │      └─ MLP                                           │    │
   │                                                       │    │
   │   FinalLayer + unpatchify                             │    │
   └───────────────────────────────────────────────────────┘    │
                             │                                  │
                             ▼                                  │
              ε̂ and log-σ prediction                            │
                             │                                  │
                             ▼                                  │
              MSE(ε̂, ε) + VLB(σ̂)  →  loss.backward()            │
              EMA update (decay=0.9999)                        ─┘


                         ┌────── PLANNING (CEM) ──────┐
                         │                            │
                         │  init μ,σ from dataset-     │
                         │  specific prior             │
                         │                            │
                         │  for opt_step in 1..K:     │
                         │    sample 120 endpoints    │
                         │      (Δx,Δy,Δφ) ~ N(μ,σ)   │
                         │    build 8-step deltas     │
                         │    ┌─────────────────────┐ │
                         │    │ autoregressive_     │ │
                         │    │ rollout(NWM, 8):    │ │
                         │    │  for t=0..7:        │ │
                         │    │   p_sample_loop     │ │
                         │    │   (250 denoise      │ │
                         │    │    steps per frame) │ │
                         │    │  slide context      │ │
                         │    └─────────────────────┘ │
                         │    score LPIPS(last,goal)  │
                         │    top-K → refit μ,σ       │
                         │                            │
                         │  emit best trajectory      │
                         └────────────────────────────┘
```

---

## 7. How to install and run

### 7.1 Prerequisites

- Linux with NVIDIA GPU (training was done on **8× H100 nodes with 8 GPUs each** — 64 H100s for CDiT-XL)
- Python 3.10
- PyTorch (nightly as of the repo README, but recent stable 2.x works)
- ~250 GB of disk for the training datasets

### 7.2 Setup

```bash
git clone https://github.com/facebookresearch/nwm
cd nwm

mamba create -n nwm python=3.10 && mamba activate nwm
pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu126
mamba install ffmpeg
pip3 install decord einops evo transformers diffusers tqdm timm notebook dreamsim torcheval lpips ipywidgets
```

### 7.3 Data prep

NWM reuses NoMaD's pre-processed datasets, but at **higher resolution** (320×240 vs NoMaD's 160×120). From the README:
1. Download RECON, SCAND, TartanDrive, GoStanford from their sources.
2. In NoMaD's `train/vint_train/data/data_utils.py`, change `IMAGE_SIZE = (160, 120)` to `(320, 240)`.
3. Run NoMaD's `process_bags.py` / `process_recon.py` to get per-trajectory folders of JPGs + `traj_data.pkl`.
4. Place under `nwm/data/<dataset_name>/<traj_name>/{0.jpg, 1.jpg, ..., traj_data.pkl}`.

For **SACSoN/HuRoN** the paper used a private higher-res version — you'd need to contact Noriaki Hirose.

### 7.4 Training

```bash
# 8 nodes × 8 GPUs
torchrun --nnodes=8 --nproc-per-node=8 --node-rank=$NODE_RANK \
         --rdzv-backend=c10d --rdzv-endpoint=$HOST:29500 \
    train.py --config config/nwm_cdit_xl.yaml \
             --ckpt-every 2000 --eval-every 10000 \
             --bfloat16 1 --epochs 300 --torch-compile 0
```

For a smaller model on a single 24 GB GPU you can swap the config to `config/wm_debug_bs_32.yaml` or start from `CDiT-S/2`. Be warned: CDiT-XL at 1 B params, bfloat16, batch 16 uses about 40 GB of VRAM per GPU — an RTX 4090 won't do it.

### 7.5 Inference / evaluation

Download the pretrained CDiT-XL checkpoint from `huggingface.co/facebook/nwm` and place under `logs/nwm_cdit_xl/checkpoints/0100000.pth.tar`.

**Time-prediction eval** (single-step quality at 1, 2, 4, 8, 16 s):
```bash
# 1) Save GT ref frames (one-time)
python isolated_nwm_infer.py --exp config/nwm_cdit_xl.yaml \
    --datasets recon,scand,sacson,tartan_drive --batch_size 96 \
    --eval_type time --output_dir $RESULTS --gt 1
# 2) Run model
python isolated_nwm_infer.py --exp config/nwm_cdit_xl.yaml --ckp 0100000 \
    --datasets recon --batch_size 64 --eval_type time --output_dir $RESULTS
# 3) Score
python isolated_nwm_eval.py --datasets recon \
    --gt_dir $RESULTS/gt --exp_dir $RESULTS/nwm_cdit_xl --eval_types time
```

**Standalone planning eval** (the mode this guide is centered on):
```bash
torchrun --nproc-per-node=8 planning_eval.py \
    --exp config/nwm_cdit_xl.yaml \
    --datasets recon \
    --rollout_stride 1 --batch_size 1 \
    --num_samples 120 --topk 5 \
    --opt_steps 1 --num_repeat_eval 3 \
    --ckp 0100000 --output_dir $RESULTS --save_preds
```

Flags to know:
| Flag | Default | Meaning |
|---|---|---|
| `--num_samples` | 120 | CEM population size |
| `--topk` | 5 | CEM elite count |
| `--opt_steps` | 1 | CEM iterations (paper uses 1, code default is 15) |
| `--num_repeat_eval` | 3 | How many stochastic rollouts per candidate to average |
| `--rollout_stride` | 1 | Time-skip factor — group this many training steps per NWM call |

### 7.6 Interactive demo

The repo ships `interactive_model.ipynb`, which is a standalone notebook: load the XL checkpoint, feed a single image, click yaw/translation sliders, and watch the model imagine what would happen. Great way to build intuition before touching the planning code.

---

## 8. Gaps, rough edges, and things the paper doesn't explain

1. **The repo doesn't ship the NoMaD-ranking experiment.** Paper Table 2's "NWM + NoMaD (×16/×32)" numbers are reproducible in principle, but you have to wire NoMaD (see NoMaD guide) in yourself: sample trajectories from it, call `autoregressive_rollout`, score with LPIPS. There's no glue code.

2. **`rel_t = k / 128` constant is magic.** The `/128` normalization is hard-coded in `datasets.py::__getitem__` and `isolated_nwm_infer.py::model_forward_wrapper`. It works because the training-time `max_dist_cat = 64`, so `|rel_t| ≤ 0.5`. Retrain with a different `max_dist_cat` and you must change this constant or the time embedding will fall outside its learned range.

3. **CEM only optimizes 3 variables, not 24.** The search space is a straight-line endpoint, *not* a general 8-step trajectory. The paper mentions this briefly in Appendix §7 ("we assume the trajectory is a straight line and optimize only its endpoint"). This is why CEM converges in one iteration with 120 samples — it's a 3-D problem, not a 24-D one. For curved trajectories the paper relies on (a) the constraint-aware variant, or (b) the ranking mode, or (c) more CEM iterations.

4. **The final yaw rotation is concentrated at the last step.** Look closely at `generate_actions`:
   ```python
   deltas[:, -1, -1] += sample[:, -1] * np.pi
   ```
   All of the yaw change goes to the *final* waypoint, not distributed evenly. This is another dimensionality-reducing assumption that works well for the 2-second evaluation horizon but would generate unnatural trajectories over longer horizons.

5. **`num_repeat_eval > 1` is necessary for reproducibility.** NWM is genuinely stochastic — `diffusion.p_sample_loop` starts from a fresh Gaussian each time. Running the same candidate twice gives you two different LPIPS scores. With 120 candidates × 1 repeat, the top-K ranking is noisy; the paper uses 3 repeats to denoise. This is also why planning with NWM is slow.

6. **Ego4D "unlabeled" data: where does it help and where does it hurt?** Table 4 shows Ego4D helps on Go-Stanford (OOD) but *hurts* on RECON (in-domain). The paper's hand-wavy explanation is "mode collapse" — when Ego4D's distribution doesn't match RECON's, adding more of it pulls predictions toward Ego4D-looking scenes. If you care about an in-domain environment, don't blindly add unlabeled data.

7. **No online deployment script.** Unlike NoMaD's `navigate.sh` / `pd_controller.py`, there is **no ROS wrapper** for running NWM on a physical robot. The paper's Table 8 discusses real-time feasibility (distillation + time-skip + 4-bit quantization gets you to ~100 ms/rollout) but the code doesn't include any of these optimizations. If you wanted to deploy, you'd need to build the Model-Predictive-Control loop yourself on top of `autoregressive_rollout`.

8. **The "time-only" ablation in Table 1 is revealing.** Running NWM with $k$ but no $(\Delta x, \Delta y, \Delta \phi)$ gives LPIPS 0.76 (near-random). With actions but no $k$: 0.32. With both: 0.30. So actions dominate — $k$ alone is nearly useless — but both together beat either. This suggests $k$ is mostly a "scale" knob that helps disambiguate cases where the same action could apply over different durations.

9. **Context of 4 frames, not 2.** Paper Table 1 shows context=1 → LPIPS 0.304, context=4 → 0.296. Small gap in metrics, but qualitatively the paper notes "with short context the model often 'loses track'." Ablating to 1 frame gives you Fig. 1(c)-style imagined environments (good for novelty) but breaks in-domain accuracy.

10. **`torch.compile` is flagged as unstable.** The README literally says "torch compile can lead to ~40% faster training speed. However, it might lead to instabilities and inconsistent behavior across different pytorch versions. Use carefully." If your trained checkpoint doesn't reproduce paper numbers, try disabling compile first.

11. **Evaluation datasets use fixed windows `(min_dist_cat=8, max_dist_cat=8)` and `len_traj_pred=8`.** The planning eval therefore always predicts exactly 2 seconds at 4 FPS. This is a much shorter horizon than the 16-second generation demos — NWM *can* roll out longer, but ATE/RPE numbers beyond 2 s aren't reported. Longer-horizon standalone planning is an open question.

---

## 9. How NWM relates to NoMaD (since you asked both)

| Aspect | NoMaD | NWM |
|---|---|---|
| Type | Goal-conditioned diffusion **policy** | Conditional diffusion **world model** |
| Output | 8-step (Δx, Δy) action plan | Predicted future RGB frame |
| Conditioning | observation context + optional goal (masked) | observation context + action + time shift |
| Backbone | EfficientNet-B0 + 4-layer Transformer + 1D U-Net | CDiT (28-layer 2D Diffusion Transformer) |
| Params | 19 M | 1 B (50× bigger) |
| Diffusion steps | 10 (inference & train) | 1000 train, 250 inference |
| Latent | None (acts directly on actions) | SD-VAE latent (4×28×28 for 224-res) |
| Planning | Implicit — one diffusion sample = one plan | External — CEM or ranking loop over NWM rollouts |
| Constraints | Bake into training data | Mask action tensor at plan time (dynamic) |
| Runtime | ~9 Hz on Jetson Orin | ~30 s/rollout on RTX 6000, 0.1 s with distillation + 4-bit |
| Exploration mode | Yes (goal-masking, $m=1$) | No (is a simulator, not a policy) |
| Best use | Online low-latency control | Offline / MPC-style planning with custom objectives |

**In Table 2 the two models team up:** NoMaD proposes 16 or 32 trajectory samples, NWM simulates each and ranks them by LPIPS-to-goal. That hybrid gets better ATE/RPE than either alone — and it's a useful template for "reactive policy + contemplative world model" architectures more broadly.

---

## 10. Suggested further reading

- **Peebles & Xie (2023), "Scalable Diffusion Models with Transformers"** — DiT, the direct ancestor of CDiT. CDiT inherits patchify, AdaLN-Zero, and learn_sigma.
- **Ho, Jain & Abbeel (2020), "DDPM"** + **Nichol & Dhariwal (2021), "Improved DDPM"** — forward/reverse process and the learned-variance VLB.
- **Rombach et al. (2022), "Latent Diffusion"** — why you run diffusion in VAE latent space and not pixel space.
- **Ha & Schmidhuber (2018), "World Models"** — the original world-model-as-imagination idea.
- **Hafner et al., Dreamer / DreamerV3** — more recent world models for RL; good for mental model of planning-via-imagination.
- **Alonso et al. (2023), "DIAMOND"** — NWM's diffusion-world-model baseline; worth reading the U-Net design to understand why CDiT wins.
- **Rubinstein (1997), "Cross-Entropy Method"** — the planning optimizer; short and readable.
- **Shah et al. (2024), "NoMaD"** + my accompanying NoMaD guide — for the policy NWM ranks against.

---

*Written from first reading both the paper and the facebookresearch/nwm repository commit on `main`. Equation numbers (Eqs. 1–5) and Table numbers refer to the arXiv v2 PDF (11 Apr 2025).*