import os
import json
import argparse
import time
from pathlib import Path
from typing import List, Tuple
import yaml
from tqdm import tqdm

import pandas as pd
from transformers import AutoTokenizer, AutoProcessor
from vllm import LLM, SamplingParams

os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

# adjust it as large as your VRAM allows ^-^
max_length = 128000

# in case you want to use thinking models
def extract_thinking_and_summary(text: str, bot: str = "<think>", eot: str = "</think>") -> Tuple[str, str]:
    if bot in text and eot not in text:
        return "", text
    if eot in text:
        if bot in text:
            return (
                text[text.index(bot) + len(bot):text.index(eot)].strip(),
                text[text.index(eot) + len(eot):].strip(),
            )
        else:
            return (
                text[:text.index(eot)].strip(),
                text[text.index(eot) + len(eot):].strip(),
            )
    return "", text

def load_config(config_path: str):
    if not os.path.exists(config_path):
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def load_documents(folder_path: str) -> List[Tuple[str, str]]:
    docs = []
    for file in sorted(Path(folder_path).glob("*.txt")):
        with open(file, "r", encoding="utf-8") as f:
            docs.append((file.name, f.read()))
    return docs

def concat_documents(docs: List[Tuple[str, str]], tokenizer, max_tokens=max_length) -> Tuple[str, int]:
    combined_text = ""
    token_count = 0
    for name, text in docs:
        header = f"\n\n===== Document: {name} =====\n\n"
        combined_text += header + text
        tokens = tokenizer.encode(combined_text, add_special_tokens=False)
        token_count = len(tokens)
        print(f"Added {name}: cumulative tokens = {token_count}")
        if token_count > max_tokens:
            print(f"⚠️ Warning: Context size exceeded {max_tokens} tokens!")
    return combined_text, token_count

def run_inference(model_path: str, docs_folder: str, system_prompt: str, question_list: str, output_file: str):
    start_time = time.time()

    # Load documents
    docs = load_documents(docs_folder)
    if not docs:
        print(f"No text files found in {docs_folder}.")
        return

    # Initialize tokenizer & processor
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    # Concatenate documents with headers
    combined_text, total_tokens = concat_documents(docs, tokenizer)
    print(f"Total tokens in concatenated context: {total_tokens}")

    # Prepare questions
    df = pd.read_csv(question_list)

    thoughts = []
    answers = []

    # Initialize vLLM model
    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
        max_model_len=max_length
    )

    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=1,
        repetition_penalty=1.05,
        max_tokens=8196,
        stop_token_ids=[]
    )
    
    for i, row in tqdm(df.iterrows(), total=len(df), desc="Inference"):
        try:
            # Safe parsing of messages
            qn = row['question']
            messages = [{"role": "system", "content": system_prompt},
        {"role": "user", "content": combined_text + qn}]
            
            prompt = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
           
            llm_inputs = {"prompt": prompt}

            outputs = llm.generate([llm_inputs], sampling_params=sampling_params)
            generated_text = outputs[0].outputs[0].text

            thought, answer = extract_thinking_and_summary(generated_text)
            print("thought: ", thought)
            print("answer: ", answer)
            thoughts.append(thought)
            answers.append(answer)

        except Exception as e:
            print(f"[!] Error on row {i}: {e}")
            thoughts.append("")
            answers.append("")

    df["thoughts"] = thoughts
    df["answers"] = answers

    # Save result as JSONL
    model_name = os.path.basename(model_path).replace("/", "_")
    output_path = Path(output_file or f"{model_name}_results.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_json(output_path, orient="records", lines=True, force_ascii=False)
    df.to_csv(Path(output_file or f"{model_name}_results.csv"))
    print(f"Results saved to {output_path}")

    elapsed = time.time() - start_time
    print(f"⏱️ Time taken: {elapsed:.2f} seconds")

def main():
    parser = argparse.ArgumentParser(description="vLLM inference on long text documents")
    parser.add_argument("--model_path", type=str, help="Local model directory or HuggingFace ID")
    parser.add_argument("--docs_folder", type=str, help="Folder containing txt documents")
    parser.add_argument("--system_prompt", type=str, default="You are a helpful assistant.", help="System prompt for the model")
    parser.add_argument("--question_list", type=str, help="Questions.csv to ask based on documents")
    parser.add_argument("--config", type=str, default=None, help="Optional YAML config file")
    parser.add_argument("--output_file", type=str, default=None, help="Output JSONL file")

    args = parser.parse_args()

    # Load config if provided
    if args.config:
        config = load_config(args.config)
        # Override args with config values if present
        args.model_path = config.get("model_path", args.model_path)
        args.docs_folder = config.get("docs_folder", args.docs_folder)
        args.system_prompt = config.get("system_prompt", args.system_prompt)
        args.question_list = config.get("question_list", args.question_list)
        args.output_file = config.get("output_file", args.output_file)

    run_inference(
        model_path=args.model_path,
        docs_folder=args.docs_folder,
        system_prompt=args.system_prompt,
        question_list=args.question_list,
        output_file=args.output_file
    )

if __name__ == "__main__":
    main()