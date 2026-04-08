Yes — here’s a practical research map for learning GFlowNets, with emphasis on **architecture**, **training loop**, and **SubTB / subtrajectory balance**.

## What a GFlowNet is

A **GFlowNet** is a stochastic generative policy that constructs an object step by step and is trained so that the probability of sampling a final object (x) is proportional to a nonnegative reward (R(x)). The key point is that it aims to produce **diverse high-reward samples**, not just a single optimum. The standard intuition is: many different action sequences can build the same final object, and the model learns flows/policies so that terminal objects are sampled in proportion to reward. ([Notion][1])

In the basic formulation, you have:

* a **state** (s_t): partial object built so far,
* an **action** (a_t): one constructive step,
* a **forward policy** (P_F(a_t \mid s_t)),
* often a **backward policy** (P_B(s_t \mid s_{t+1})),
* a terminal reward (R(x)),
* and sometimes a learned normalizer / state-flow quantity depending on the loss used. ([Notion][1])

## The simplest architecture

The simplest GFlowNet architecture is just:

1. an **environment** that defines valid states, actions, and rewards,
2. a neural network for the **forward policy**,
3. optionally another neural network for the **backward policy**,
4. a sampler that rolls out trajectories,
5. a loss such as **TB**, **DB**, **FM**, or **SubTB**,
6. an optimizer updating the policy parameters from sampled trajectories. ([Notion][1])

The torchgfn docs make this very explicit: the library separates **states/actions/containers**, **modules/estimators/samplers**, and **losses**. Its simple example uses:

* `HyperGrid` as the environment,
* an MLP for the forward policy,
* another MLP for the backward policy,
* a `Sampler`,
* and a `TBGFlowNet` loss wrapper. ([gfn.readthedocs.io][2])

So, conceptually:

```text
state s_t
   ↓
encoder / preprocessor
   ↓
policy network(s)
   ↓
forward logits P_F(.|s_t), optional backward logits P_B(.|s_{t+1})
   ↓
sample action / trajectory
   ↓
compute reward on terminal state
   ↓
compute GFlowNet loss
   ↓
backprop + optimizer step
```

## Where SubTB fits

Historically:

* **Trajectory Balance (TB)** was introduced to improve credit assignment across long trajectories. ([OpenReview][3])
* **SubTB((\lambda))** was introduced afterward to learn from **partial subtrajectories** rather than only whole trajectories, giving a bias–variance tradeoff similar in spirit to TD((\lambda)) in RL. The ICML 2023 paper reports improved convergence/stability and better behavior on longer sequences and sparser rewards. ([OpenReview][4])

That means SubTB is not a different “network architecture” by itself. It is a **training objective**. In practice, you usually keep almost the same model components as TB, but you change **how the loss is computed from sampled trajectories**. ([OpenReview][4])

## How training usually looks

For learning purposes, this is the clean mental model:

```python
for step in range(num_steps):
    trajectories = sampler.sample_trajectories(policy=PF, env=env, batch_size=B)

    terminal_states = trajectories.last_states()
    rewards = env.reward(terminal_states)

    loss = gflownet_loss(
        trajectories=trajectories,
        rewards=rewards,
        PF=PF,
        PB=PB,      # optional depending on setup
        logZ=logZ   # used in TB-style formulations
    )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

With **SubTB**, the main change is that the loss is computed over **many contiguous subsegments** of each trajectory, instead of only enforcing a whole-trajectory balance equation. That is why it often trains more stably on long-horizon construction tasks. ([OpenReview][4])

## Best resources, in learning order

### 1. Best high-level theory first

**The GFlowNet Tutorial** by Yoshua Bengio, Kolya Malkin, and Moksh Jain is still one of the best conceptual introductions. It explains the forward/backward policy view, trajectories, rewards, and the “sample proportional to reward” objective in a very intuitive way. ([Notion][1])

Then read **GFlowNet Foundations** for the formal framework. It is the standard theory reference and covers core properties and training objectives like detailed balance. ([Journal of Machine Learning Research][5])

### 2. Best practical notebook/tutorial

**Emmanuel Bengio’s practical Colab tutorial** is one of the most useful code-oriented starting points for beginners. It is repeatedly referenced by official and semi-official GFlowNet resources as a practical entry point. ([Google Colab][6])

### 3. Best library for learning architecture and training

**GFNOrg/torchgfn** is the strongest learning-oriented codebase I found for this purpose. Its README and docs are structured around:

* simple examples,
* tutorials,
* environments,
* modules/estimators/samplers,
* and losses including **TB, DB, SubTB, FM, ZVar, and RTB**. ([GitHub][7])

This is probably the best place to study **how the architecture is actually wired together**.

### 4. Best minimal example

The **`train_hypergrid_simple`** tutorial in torchgfn is especially good because it is intentionally simplified:

* only TB loss,
* no replay buffer,
* no wandb,
* simpler architecture with shared trunks,
* and basic CLI options. ([torchgfn.readthedocs.io][8])

Even though it uses **TB rather than SubTB**, it is still the best first code example because it shows the core mechanics without too much engineering clutter. ([torchgfn.readthedocs.io][8])

### 5. Best full training example

The fuller **`train_hypergrid`** example shows a more realistic training setup, including helper functions to build estimators, set up the GFlowNet, and evaluate the exact terminating distribution (P_T). ([torchgfn.readthedocs.io][9])

### 6. Best alternative codebase

The **alexhernandezgarcia/gflownet** project is another useful implementation with docs, Colab notebooks, and grid/custom-environment tutorials. Its README describes the agent as orchestrating the environment, policies, proxy, evaluator, logging, and trajectory sampling. ([gflownet][10])

## Best papers to read for your exact goal

For your goal, I would read these in this order:

1. **The GFlowNet Tutorial** — intuition first. ([Notion][1])
2. **Trajectory Balance: Improved Credit Assignment in GFlowNets** — understand why TB mattered. ([OpenReview][3])
3. **Learning GFlowNets from Partial Episodes for Improved Convergence and Stability** — this is the key SubTB paper. ([OpenReview][4])
4. **GFlowNet Foundations** — formalize everything. ([Journal of Machine Learning Research][5])
5. **GFlowNets for AI-Driven Scientific Discovery** — good broader perspective on where GFlowNets are used. ([arXiv][11])

## My honest recommendation on simple projects

For **learning**, start with these project types:

### Project A — HyperGrid with TB

Why: the environment is small, discrete, and visualizable. You can inspect trajectories and compare the learned terminal distribution to the exact one. The torchgfn simple HyperGrid example is built exactly for this. ([torchgfn.readthedocs.io][8])

### Project B — HyperGrid with SubTB

After you understand TB, replace the TB loss with **SubTB** in the same setup. This is the cleanest way to learn what changes operationally:

* same env,
* same policies,
* same sampler,
* different loss. ([gfn.readthedocs.io][2])

### Project C — custom sequence builder

Build a toy environment where the model constructs:

* a short binary string,
* a small set,
* or a tiny expression tree,
  with reward higher for specific patterns. This makes the “many paths to the same object” idea very concrete. The official tutorials on defining environments are useful for this step. ([GitHub][7])

## What to focus on in the code

When reading a repository, look for these files/classes first:

1. **Environment**

   * how states are represented,
   * valid actions,
   * terminal condition,
   * reward function.

2. **Policy estimators**

   * forward policy head,
   * backward policy head,
   * shared trunk or separate networks.

3. **Sampler**

   * how trajectories are rolled out,
   * exploration settings,
   * on-policy vs off-policy sampling.

4. **Loss**

   * TB or SubTB class/function,
   * what tensors it expects,
   * whether it uses full trajectories or subtrajectories.

5. **Evaluation**

   * terminal-state histogram,
   * KL / L1 / mode coverage,
   * reward distribution. ([torchgfn.readthedocs.io][9])

## Best blog-style resources

For blog-style reading, I would prioritize:

* **Yoshua Bengio’s overview page**, which points to the major tutorial and original materials. ([yoshuabengio.org][12])
* **The GFlowNet Tutorial** on Notion, which reads more like a long-form tutorial than a paper. ([Notion][1])
* **Emmanuel Bengio’s original overview/blog post**, useful for the initial intuition and flow-network picture. ([folinoid.com][13])

I would treat random Medium posts as secondary, not primary, for serious study.

## A concrete study plan

Here is the path I would use:

**Phase 1: intuition**

* Read **The GFlowNet Tutorial**
* Skim the original blog/overview
* Understand forward policy, backward policy, trajectories, reward-proportional sampling. ([Notion][1])

**Phase 2: minimal code**

* Run **torchgfn `train_hypergrid_simple`**
* Trace environment → forward/backward estimators → sampler → TB loss → optimizer. ([torchgfn.readthedocs.io][8])

**Phase 3: SubTB**

* Read the **SubTB paper**
* Modify the HyperGrid training to use SubTB
* Compare stability and terminal-distribution fit. ([OpenReview][4])

**Phase 4: formal theory**

* Read **Trajectory Balance**
* Then **GFlowNet Foundations**. ([OpenReview][3])

## My distilled advice

If your goal is **“understand how GFlowNet architecture is trained in practice”**, the best stack is:

* **Conceptual intro:** The GFlowNet Tutorial
* **Minimal code:** `torchgfn` HyperGrid simple example
* **Main theory step:** TB paper
* **Your target loss:** SubTB paper
* **Broader formal reference:** GFlowNet Foundations ([Notion][1])

One important note: I found strong **TB** tutorials and libraries, and good **SubTB** paper/library support, but I did **not** find a single especially famous beginner notebook dedicated purely to **SubTB** in the same way TB/HyperGrid examples are presented. The practical route is usually: learn on TB first, then switch the loss to SubTB inside the same codebase. ([torchgfn.readthedocs.io][8])

I can turn this into a **curated reading list with direct links and a 2-week study plan**, or into a **minimal PyTorch notebook skeleton for a toy GFlowNet with SubTB-style training**.

[1]: https://milayb.notion.site/The-GFlowNet-Tutorial-95434ef0e2d94c24aab90e69b30be9b3 "The GFlowNet Tutorial | Notion"
[2]: https://gfn.readthedocs.io/ "torchgfn: a Python package for GFlowNets — torchgfn  documentation"
[3]: https://openreview.net/forum?id=5btWTw1vcw1&utm_source=chatgpt.com "Trajectory balance: Improved credit assignment in GFlowNets"
[4]: https://openreview.net/forum?id=UYS38ssi1M&utm_source=chatgpt.com "Learning GFlowNets from partial episodes for improved ..."
[5]: https://jmlr.org/papers/volume24/22-0364/22-0364.pdf?utm_source=chatgpt.com "GFlowNet Foundations"
[6]: https://colab.research.google.com/drive/1fUMwgu2OhYpQagpzU5mhe9_Esib3Q2VR?utm_source=chatgpt.com "GFlowNet tutorial"
[7]: https://github.com/GFNOrg/torchgfn "GitHub - GFNOrg/torchgfn: A modular, easy to extend GFlowNet library · GitHub"
[8]: https://torchgfn.readthedocs.io/en/latest/autoapi/tutorials/examples/train_hypergrid_simple/index.html "torchgfn :: tutorials.examples.train_hypergrid_simple"
[9]: https://torchgfn.readthedocs.io/en/latest/autoapi/tutorials/examples/train_hypergrid/index.html "torchgfn :: tutorials.examples.train_hypergrid"
[10]: https://gflownet.readthedocs.io/ "GFlowNet Documentation — gflownet  documentation"
[11]: https://arxiv.org/abs/2302.00615?utm_source=chatgpt.com "GFlowNets for AI-Driven Scientific Discovery"
[12]: https://yoshuabengio.org/en/blog/generative-flow-networks "Generative Flow Networks | Yoshua Bengio"
[13]: https://folinoid.com/w/gflownet/ "Flow Network based Generative Models for Non-Iterative Diverse Candidate Generation"
