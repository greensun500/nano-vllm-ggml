import os
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer


def main():
    path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    tokenizer = AutoTokenizer.from_pretrained(path)# 加载tokenizer对象。此处是为了把prompt转换为tokenizer期望的格式。
    llm = LLM(path, enforce_eager=True, tensor_parallel_size=1)
    # 初始化模型，把模型加载到内存中，并进行预热，同时设置好子进程（如果有），并把调度器也初始化好，等待输入任务即开始执行

    sampling_params = SamplingParams(temperature=0.6, max_tokens=256)#设置采样参数
    prompts = [
        "introduce yourself",
        "list all prime numbers within 100",
    ]
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]
    outputs = llm.generate(prompts, sampling_params)

    for prompt, output in zip(prompts, outputs):
        print("\n")
        print(f"Prompt: {prompt!r}")
        print(f"Completion: {output['text']!r}")


if __name__ == "__main__":
    main()
