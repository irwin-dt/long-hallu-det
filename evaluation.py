import pandas as pd
from pathlib import Path
from typing import List, Tuple
import numpy as np

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

def calculate_score(y_true, y_pred):
    # Convert to numpy arrays for easier manipulation
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Accuracy
    accuracy = sum(1 for x,y in zip(y_pred,y_true) if x == y) / len(y_true)
    print(f"Accuracy: {accuracy*100}%")
    
    # True Positives, False Positives, False Negatives
    tp = np.sum((y_true == "Yes") & (y_pred == "Yes"))
    fp = np.sum((y_true == "No") & (y_pred == "Yes"))
    fn = np.sum((y_true == "Yes") & (y_pred == "No"))
    
    # Precision and Recall
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    print(f"Precision: {precision*100}%")
    print(f"Recall: {recall*100}%")
    
    # F1 Score
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    print(f"F1 Score: {f1*100}%")
    return f1
    
def main():
    long_context = concat_documents(load_documents("dummy_files/hard"))
    testset = pd.read_csv("dummy_testset.csv")
    calculate_score(testset['hallucination'], hallu_det(long_context, testset))

if __name__ == "__main__":
    main()
