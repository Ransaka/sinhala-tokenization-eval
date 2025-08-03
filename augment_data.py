import os
import random
import string
from typing import Callable, Dict

import pandas as pd
from sinling import Romanizer

def introduce_minor_typos(text: str, probability: float = 0.05) -> str:
    """
    Introduces minor typos into a string.

    Args:
        text (str): The original Sinhala text.
        probability (float): The approximate ratio of characters to change.

    Returns:
        str: The text with minor typos.
    """
    if not text or len(text) <= 1:
        return text

    chars = list(text)
    text_len = len(chars)

    # Calculate the number of changes to make, ensuring at least one.
    num_changes = max(1, int(text_len * probability))

    for _ in range(num_changes):
        pos = random.randint(0, text_len - 1)
        typo_type = random.choice(["swap", "delete", "insert", "replace"])

        if typo_type == "swap" and pos < text_len - 1:
            chars[pos], chars[pos + 1] = chars[pos + 1], chars[pos]
        
        elif typo_type == "delete" and text_len > 2:
            chars.pop(pos)
            text_len -= 1  # Update length after deletion
        
        elif typo_type == "insert":
            random_char = random.choice(text)
            chars.insert(min(pos + 1, text_len), random_char)
            text_len += 1  # Update length after insertion
        
        elif typo_type == "replace":
            chars[pos] = random.choice(text)

    return "".join(chars)


def introduce_aggressive_typos(text: str, probability: float = 0.1) -> str:
    """
    Introduces more significant typos into a string.

    Args:
        text (str): The original Sinhala text.
        probability (float): The approximate ratio of characters to change.

    Returns:
        str: The text with aggressive typos.
    """
    if not text or len(text) <= 1:
        return text

    chars = list(text)
    text_len = len(chars)

    # Calculate the number of changes, ensuring at least two for aggressiveness.
    num_changes = max(2, int(text_len * probability))

    for _ in range(num_changes):
        pos = random.randint(0, text_len - 1)
        typo_type = random.choice(["swap", "delete", "insert", "replace", "duplicate"])

        if typo_type == "swap" and pos < text_len - 1:
            chars[pos], chars[pos + 1] = chars[pos + 1], chars[pos]
        
        elif typo_type == "delete" and text_len > 3:
            chars.pop(pos)
            text_len -= 1
        
        elif typo_type == "insert":
            random_char = random.choice(text)
            chars.insert(min(pos + 1, text_len), random_char)
            text_len += 1
        
        elif typo_type == "replace":
            chars[pos] = random.choice(text)
            
        elif typo_type == "duplicate" and text_len > 0:
            chars.insert(pos, chars[pos])
            text_len += 1

    return "".join(chars)


def introduce_mixed_coding(
    text: str, romanize_func: Callable[[str], str], probability: float = 0.3
) -> str:
    """
    Introduces mixed-coding by romanizing some words.

    This function first introduces minor typos and then converts a fraction
    of the words into Roman script (Singlish).

    Args:
        text (str): The original Sinhala text.
        romanize_func (Callable[[str], str]): A function to romanize text.
        probability (float): The probability of romanizing a single word.

    Returns:
        str: The text with mixed coding and typos.
    """
    if not text:
        return text

    # First, introduce minor typos to simulate user error
    modified_text = introduce_minor_typos(text, probability=0.05)

    words = modified_text.split()
    romanized_words = []
    
    for word in words:
        if random.random() < probability:
            romanized_words.append(romanize_func(word))
        else:
            romanized_words.append(word)

    return " ".join(romanized_words)


def generate_augmented_datasets(
    df: pd.DataFrame,
    romanize_func: Callable[[str], str],
    output_dir: str = "augmented_datasets",
) -> Dict[str, pd.DataFrame]:
    """
    Generates and saves three augmented datasets from an original DataFrame.

    Args:
        df (pd.DataFrame): The original dataset with 'text' and 'label' columns.
        romanize_func (Callable[[str], str]): The function to romanize Sinhala text.
        output_dir (str): The directory to save the output CSV files.

    Returns:
        Dict[str, pd.DataFrame]: A dictionary containing the three augmented datasets.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Create copies for each augmentation type
    df_minor = df.copy()
    df_aggressive = df.copy()
    df_mixed = df.copy()

    # Apply the augmentation functions to the 'text' column
    df_minor["text"] = df_minor["text"].apply(
        lambda x: introduce_minor_typos(x) if isinstance(x, str) else x
    )
    df_aggressive["text"] = df_aggressive["text"].apply(
        lambda x: introduce_aggressive_typos(x) if isinstance(x, str) else x
    )
    df_mixed["text"] = df_mixed["text"].apply(
        lambda x: introduce_mixed_coding(x, romanize_func) if isinstance(x, str) else x
    )

    # Save the augmented datasets to CSV files
    df_minor.to_csv(os.path.join(output_dir, "dataset_minor_typos.csv"), index=False)
    df_aggressive.to_csv(os.path.join(output_dir, "dataset_aggressive_typos.csv"), index=False)
    df_mixed.to_csv(os.path.join(output_dir, "dataset_mixed_coding.csv"), index=False)
    
    print(f"Datasets successfully saved in '{output_dir}' directory.")

    return {
        "minor_typos": df_minor,
        "aggressive_typos": df_aggressive,
        "mixed_coding": df_mixed,
    }
