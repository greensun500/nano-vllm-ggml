#!/usr/bin/env python3
"""用 NumPy 演示单 token 的原始投机采样（speculative sampling）。

这个文件刻意不依赖 LLM：把词表缩小到 A、B、C、D、E 五个 token，并人为
给定两个概率分布。这样可以直接观察原始算法的数学性质：

    虽然 token 先由草稿分布 q 提出，最终输出的经验分布仍会收敛到目标分布 p。

符号约定：
    p: 目标模型（target model）的分布；慢、昂贵，但它决定最终输出质量。
    q: 草稿模型（draft model）的分布；快、便宜，只负责提前提出候选 token。

运行：
    python examples/speculative_sampling_toy.py
    python examples/speculative_sampling_toy.py --trials 1000000 --seed 7

和真实 LLM 的关系：
    真实 LLM 会让 q 连续猜 K 个 token；p 用一次前向计算得到这 K 个位置的
    概率，再从左至右套用本文件的“接受/残差采样”规则。此处只演示 K=1，目的是
    把概率正确性的核心从 Transformer、KV cache 等工程细节中剥离出来。
"""

from __future__ import annotations

import argparse

import numpy as np


TOKENS = np.array(["A", "B", "C", "D", "E"])

# p 是慢但准确的目标模型分布。只用目标模型时，最终 token 应服从这个分布。
P = np.array([0.35, 0.25, 0.20, 0.15, 0.05])

# q 是快速草稿模型分布。故意让 q 和 p 差异较大：
# - q 对 B、E 过度自信（q > p），这些候选有可能被拒绝；
# - q 对 A、C、D 不够自信（q < p），这些候选会被直接接受；
# 这样实验中既能看到“接受”，也能看到“残差采样”。
Q = np.array([0.05, 0.50, 0.10, 0.10, 0.25])


def normalize(probabilities: np.ndarray) -> np.ndarray:
    """截断浮点负误差后，把向量归一化为概率分布。"""
    # 理论上 residual = max(p - q, 0) 本来就非负；clip 也能消除浮点误差。
    probabilities = np.clip(probabilities, 0.0, None)
    total = probabilities.sum()
    if total <= 0:
        raise ValueError("probabilities must contain positive mass")
    return probabilities / total


def frequencies(samples: np.ndarray, vocab_size: int) -> np.ndarray:
    """把采样结果转换为每个 token 的经验频率。"""
    return np.bincount(samples, minlength=vocab_size) / len(samples)


def speculative_sample(
    target: np.ndarray, draft: np.ndarray, trials: int, rng: np.random.Generator
) -> tuple[np.ndarray, float]:
    """按“原始投机采样”规则，每次生成一个最终 token。

    对每次 trial，算法严格执行：

    1. 草稿模型先提出候选：y ~ q。
    2. 以 alpha(y) = min(1, p(y) / q(y)) 接受这个候选。
    3. 若拒绝，改从 r = normalize(max(p - q, 0)) 采样。

    为什么第 2 步是这个接受率？
        token y 被“提出且接受”的总概率为：

            q(y) * min(1, p(y) / q(y)) = min(q(y), p(y))

        这正好是 p 与 q 在 y 位置重叠的那部分概率质量。

    为什么第 3 步必须从残差 r，而不能直接从 p？
        接受步骤已经产生了 min(p, q) 这一部分；目标分布 p 尚未产生的部分是
        max(p - q, 0)。因此：

            min(p, q) + max(p - q, 0) = p

        接受部分加上残差部分，才能恰好恢复 p。这就是“无损”投机采样的关键。
    """
    # 第一步：快速草稿模型按 q 提出 token。真实实现还会保存每一步的 q logits，
    # 因为后续要计算 p(y) / q(y)。
    proposed = rng.choice(len(draft), size=trials, p=draft)

    # 第二步：目标模型验证。这里只读取被提出 token 的 p(y)、q(y)，从而得到
    # alpha(y) = min(1, p(y) / q(y))。
    #
    # q(y) > p(y) 表示草稿“过度相信” y，必须按比例拒绝一部分；
    # q(y) <= p(y) 时 alpha=1，目标模型至少同样相信它，可直接接受。
    acceptance_probability = np.minimum(1.0, target[proposed] / draft[proposed])
    accepted = rng.random(trials) < acceptance_probability

    output = proposed.copy()

    # 第三步：只对被拒绝的 trial 采样。残差中只保留“目标模型比草稿模型更多的
    # 概率质量”。对于本例：A/C/D 有正残差，B/E 的残差为 0。
    residual = normalize(target - draft)
    rejected_count = np.count_nonzero(~accepted)
    output[~accepted] = rng.choice(len(target), size=rejected_count, p=residual)
    return output, accepted.mean()


def incorrect_resample_from_target(
    target: np.ndarray, draft: np.ndarray, trials: int, rng: np.random.Generator
) -> np.ndarray:
    """故意错误的对照：拒绝后直接从 p 采样，而不是从 p-q 的残差采样。

    这会把已由接受阶段贡献过的概率质量再加一遍，最终分布通常偏离 p。保留它是
    为了让输出表中的 ``wrong-reject->p`` 直观展示“残差采样不可省略”。
    """
    proposed = rng.choice(len(draft), size=trials, p=draft)
    acceptance_probability = np.minimum(1.0, target[proposed] / draft[proposed])
    accepted = rng.random(trials) < acceptance_probability
    output = proposed.copy()
    output[~accepted] = rng.choice(len(target), size=np.count_nonzero(~accepted), p=target)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    if args.trials <= 0:
        parser.error("--trials must be positive")

    p = normalize(P)
    q = normalize(Q)

    # r 是 rejection 时使用的条件分布。注意 p-q 中可能有负值，必须取正部：
    # max(p-q, 0)，不能把负数也拿来归一化。
    residual = normalize(p - q)

    # sum(min(p, q)) 是理论接受率：每一轮有这么多概率质量可直接通过验证。
    expected_acceptance = np.minimum(p, q).sum()

    # 使用不同但固定的随机种子，使三个实验彼此独立且每次运行都可复现。
    # “直接 target”是我们要验证的基准；它也应当收敛到 p。
    direct = np.random.default_rng(args.seed).choice(len(p), size=args.trials, p=p)
    speculative, observed_acceptance = speculative_sample(
        p, q, args.trials, np.random.default_rng(args.seed + 1)
    )
    wrong = incorrect_resample_from_target(
        p, q, args.trials, np.random.default_rng(args.seed + 2)
    )

    direct_freq = frequencies(direct, len(p))
    speculative_freq = frequencies(speculative, len(p))
    wrong_freq = frequencies(wrong, len(p))

    print(f"trials: {args.trials:,}")
    print(f"理论接受率 sum(min(p, q)): {expected_acceptance:.4f}")
    print(f"实际接受率:                {observed_acceptance:.4f}")
    print(f"残差 r = normalize(max(p - q, 0)): {residual}")
    print()
    print("token       p   直接从 p  正确投机采样  错误：拒绝后从 p 采样")
    print("-----  ------  ---------  ------------  -------------------")
    for token, target_prob, direct_prob, spec_prob, wrong_prob in zip(
        TOKENS, p, direct_freq, speculative_freq, wrong_freq
    ):
        print(
            f"{token:>5}  {target_prob:6.4f}  {direct_prob:10.4f}"
            f"  {spec_prob:11.4f}  {wrong_prob:15.4f}"
        )

    print()
    print("最大误差 max |经验频率 - p|（试验次数有限，正确方法也会有随机误差）")
    print(f"  直接从目标分布 p 采样: {np.max(np.abs(direct_freq - p)):.4f}")
    print(f"  正确投机采样:          {np.max(np.abs(speculative_freq - p)):.4f}")
    print(f"  错误的 reject -> p:    {np.max(np.abs(wrong_freq - p)):.4f}")


if __name__ == "__main__":
    main()
