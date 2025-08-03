import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from pathlib import Path
import argparse
import os
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import wandb 

from tokenizer import create_tokenizer
from transformer_model import TransformerForClassification
from hate_speech_dataset import SinhalaHateSpeechDataset, load_sinhala_hate_speech_data
import time
import pandas as pd

def calculate_compression_ratio(tokenizer, texts):
    original_lengths = [len(text) for text in texts]
    tokenized_lengths = [len(tokenizer.encode(text)) for text in texts]
    
    if not original_lengths or sum(original_lengths) == 0:
        return 0
        
    return sum(tokenized_lengths) / sum(original_lengths)

def evaluate(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    start_time = time.time()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs['logits']
            
            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
    
    end_time = time.time()
    inference_time = end_time - start_time
    
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro', zero_division=0)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }, inference_time

def train(args):
    wandb.init(
        project="sinhala-tokenization-paper", 
        name=args.run_name,
        config={
            "data_path": args.data_path,
            "max_seq_length": args.max_seq_length,
            "tokenizer_type": args.tokenizer_type,
            "hidden_size": args.hidden_size,
            "num_layers": args.num_layers,
            "num_heads": args.num_heads,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "num_epochs": args.num_epochs,
            "seed": args.seed,
        }
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    print(f"Using device: {device}")
    wandb.config.update({"device": str(device)})
    
    full_df = pd.read_csv(args.data_path)

    train_df = full_df.sample(frac=0.8, random_state=args.seed)
    test_df = full_df.drop(train_df.index)

    train_texts, train_labels = train_df['text'].tolist(), train_df['label'].tolist()
    test_texts, test_labels = test_df['text'].tolist(), test_df['label'].tolist()
    
    # Initialize tokenizer
    print(f"Initializing {args.tokenizer_type} tokenizer...")
    tokenizer = create_tokenizer(
        args.tokenizer_type, 
        max_length=args.max_seq_length, 
        train_texts=train_texts
    )
    
    vocab_size = tokenizer.total_vocab_size
    compression = calculate_compression_ratio(tokenizer, train_texts[:500]) # On a sample
    print(f"Vocabulary size: {vocab_size}")
    print(f"Compression Ratio: {compression:.4f}")
    wandb.config.update({
        "vocabulary_size": vocab_size,
        "compression_ratio": compression
    })
    
    train_dataset = SinhalaHateSpeechDataset(train_texts, train_labels, tokenizer)
    test_dataset = SinhalaHateSpeechDataset(test_texts, test_labels, tokenizer)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=SinhalaHateSpeechDataset.collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=SinhalaHateSpeechDataset.collate_fn)
    
    model = TransformerForClassification(vocab_size=vocab_size, hidden_size=args.hidden_size, num_hidden_layers=args.num_layers, num_attention_heads=args.num_heads, intermediate_size=args.intermediate_size, dropout_prob=args.dropout_prob, max_seq_len=args.max_seq_length, num_labels=args.num_labels)
    model.to(device)
    
    wandb.watch(model, log="all", log_freq=100)
    
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    
    total_training_start_time = time.time()
    best_f1 = 0.0
    
    for epoch in range(args.num_epochs):
        epoch_start_time = time.time()
        model.train()
        total_loss = 0.0
        
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{args.num_epochs}")
        for step, batch in enumerate(progress_bar):

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs['loss']
            
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': total_loss / (step + 1)})

        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        
        # Evaluate on test set
        eval_metrics, inference_time = evaluate(model, test_dataloader, device)
        inference_latency_per_sample = inference_time / len(test_dataset)

        print(f"Epoch {epoch + 1} - F1: {eval_metrics['f1']:.4f}, Time: {epoch_duration:.2f}s, Inference Latency: {inference_latency_per_sample*1000:.4f} ms/sample")
        
        # Log all new metrics to wandb
        wandb.log({
            "eval/accuracy": eval_metrics['accuracy'],
            "eval/precision": eval_metrics['precision'],
            "eval/recall": eval_metrics['recall'],
            "eval/f1": eval_metrics['f1'],
            "speed/epoch_duration_s": epoch_duration,
            "speed/inference_latency_ms_per_sample": inference_latency_per_sample * 1000,
            "epoch": epoch + 1
        })
        
        if eval_metrics['f1'] > best_f1:
            best_f1 = eval_metrics['f1']
            
            model_dir = f"{args.output_dir}/{args.run_name}"
            os.makedirs(model_dir, exist_ok=True)
            model_path = os.path.join(model_dir, 'best_model.pt')
            
            torch.save(model.state_dict(), model_path)
            print(f"Saved new best model with F1: {best_f1:.4f}")
            
            # Log best metrics
            wandb.run.summary["best_f1"] = best_f1
            wandb.run.summary["best_accuracy"] = eval_metrics['accuracy']
            wandb.run.summary["best_precision"] = eval_metrics['precision']
            wandb.run.summary["best_recall"] = eval_metrics['recall']
            wandb.run.summary["best_epoch"] = epoch + 1
            
            # Log best model to wandb
            artifact = wandb.Artifact(
                name=f"model-{wandb.run.id}", 
                type="model", 
                description=f"Best model from run {wandb.run.name} with F1: {best_f1:.4f}")
            artifact.add_file(model_path)
            wandb.log_artifact(artifact)

    total_training_end_time = time.time()
    total_duration = total_training_end_time - total_training_start_time
    wandb.run.summary["total_training_duration_s"] = total_duration
    
    if device.type == 'cuda':
        peak_memory_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
        wandb.run.summary["peak_gpu_memory_mb"] = peak_memory_mb
        
    wandb.finish()

def main():
    parser = argparse.ArgumentParser()
    
    # Data parameters
    parser.add_argument("--data_path", type=str, required=True, help="Path to the data CSV file")
    parser.add_argument("--output_dir", type=str, default="./output", help="Directory to save the model")
    parser.add_argument("--max_seq_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--test_size", type=float, default=0.1, help="Test set proportion")
    
    # Tokenizer parameters
    parser.add_argument("--tokenizer_type", type=str, choices=["byte", "char", "word", "wpe", "sinlib"], 
                       default="byte", help="Type of tokenizer to use")
    
    # Model parameters
    parser.add_argument("--num_labels", type=int, default=2, help="Number of classes in the dataset")
    parser.add_argument("--hidden_size", type=int, default=512, help="Hidden size of the transformer")
    parser.add_argument("--num_layers", type=int, default=4, help="Number of transformer layers")
    parser.add_argument("--num_heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--intermediate_size", type=int, default=512, help="Intermediate size in feed-forward layers")
    parser.add_argument("--dropout_prob", type=float, default=0.2, help="Dropout probability")
    parser.add_argument("--fixed_model_size", action="store_true", 
                       help="Don't automatically adjust model size based on vocabulary")
    
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=32, help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Maximum gradient norm")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--no_cuda", action="store_true", help="Disable CUDA")
    
    # Wandb parameters
    parser.add_argument("--run_name", type=str, required=True, help="tokenizer-papaer-experiment")
    
    args = parser.parse_args()
    
    # Set seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    train(args)

if __name__ == "__main__":
    main()
