import pandas as pd
from pathlib import Path
from typing import List, Tuple

def load_documents(folder_path: str) -> List[Tuple[str, str]]:
    docs = []
    for file in sorted(Path(folder_path).glob("*.txt")):
        with open(file, "r", encoding="utf-8") as f:
            docs.append((file.name, f.read()))
    return docs
    
def concat_documents(docs: List[Tuple[str, str]]) -> Tuple[str, int]:
    combined_text = ""
    for name, text in docs:
        header = f"\n\n===== Document: {name} =====\n\n"
        combined_text += header + text
    return combined_text

### Your solution here! Please feel free to be creative! ###
def hallu_det(long_context, testset):
    ### Assume your solution returns some classification 
    return ["Yes", "Yes", "Yes"]
    
def main():
    long_context = concat_documents(load_documents("dummy_files/hard"))
    testset = pd.read_csv("dummy_testset.csv")
    
    a = hallu_det(long_context, testset)
    b = testset['hallucination']
    print(f"Congratulations, Your accuracy is {sum(1 for x,y in zip(a,b) if x == y) / len(a)*100}% !")

if __name__ == "__main__":
    main()