import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns

# Function to load a model
def load_model(model_path):
    print(f"Loading model from {model_path}...")
    return AutoModel.from_pretrained(model_path)

# Function to compare parameters between models
def compare_parameters(models, model_names):
    print("\n===== Comparing Model Parameters =====")
    
    # Check if parameters differ
    all_different = True
    
    # Compare each pair of models
    for i in range(len(models)):
        for j in range(i+1, len(models)):
            differences = []
            total_params = 0
            different_params = 0
            
            # Compare corresponding parameters
            for (name1, p1), (name2, p2) in zip(models[i].named_parameters(), models[j].named_parameters()):
                if name1 != name2:
                    print(f"Warning: Parameter names don't match: {name1} vs {name2}")
                    continue
                
                # Compare tensors
                total_params += p1.numel()
                diff = (p1 != p2).float().sum().item()
                different_params += diff
                
                if diff > 0:
                    differences.append((name1, diff / p1.numel() * 100))
            
            # Sort differences by percentage
            differences.sort(key=lambda x: x[1], reverse=True)
            
            different_percentage = (different_params / total_params) * 100
            print(f"\nComparing {model_names[i]} vs {model_names[j]}:")
            print(f"  Total parameters: {total_params:,}")
            print(f"  Different parameters: {different_params:,} ({different_percentage:.2f}%)")
            
            if different_percentage < 0.01:
                all_different = False
                print("  These models appear to be nearly identical!")
            else:
                print("  Top 5 layers with most differences:")
                for name, pct in differences[:5]:
                    print(f"    {name}: {pct:.2f}% different")
    
    return all_different

# Function to test models on sample inputs
def compare_outputs(models, model_names, tokenizer):
    print("\n===== Comparing Model Outputs =====")
    
    # Define test sequences
    test_sequences = [
        "M E T P A W Q P L A A L A I V F G L A P A S A L D H",
        "A K I T K P V H F S P T D L Y I G K G E M Q V D V S K",
        "M K K L L P T A A A G L L L L A A Q P A M A A Q G G R"
    ]
    
    # Process each test sequence
    outputs = {}
    for model, name in zip(models, model_names):
        model_outputs = []
        
        for seq in test_sequences:
            inputs = tokenizer(seq, return_tensors="pt", padding=True)
            with torch.no_grad():
                output = model(**inputs).last_hidden_state[:, 0, :].numpy()  # Use CLS token output
                model_outputs.append(output)
        
        outputs[name] = model_outputs
    
    # Compute similarities between model outputs
    for i in range(len(models)):
        for j in range(i+1, len(models)):
            print(f"\nComparing {model_names[i]} vs {model_names[j]} outputs:")
            similarities = []
            
            for idx in range(len(test_sequences)):
                sim = cosine_similarity(outputs[model_names[i]][idx], outputs[model_names[j]][idx])[0][0]
                similarities.append(sim)
                print(f"  Sequence {idx+1}: Cosine similarity = {sim:.6f}")
            
            avg_sim = np.mean(similarities)
            print(f"  Average similarity: {avg_sim:.6f}")
            
            if avg_sim > 0.9999:
                print("  These models produce nearly identical outputs!")

# Function to visualize parameter distributions
def visualize_parameters(models, model_names):
    print("\n===== Visualizing Parameter Distributions =====")
    
    # Let's examine the attention weights for the first layer
    plt.figure(figsize=(12, 8))
    for i, (model, name) in enumerate(zip(models, model_names)):
        # Get the query weights from the first attention layer
        layer_name = "bert.encoder.layer.0.attention.self.query.weight"
        try:
            # Access the named parameter
            for param_name, param in model.named_parameters():
                if layer_name in param_name:
                    weights = param.detach().cpu().numpy().flatten()
                    plt.subplot(len(models), 1, i+1)
                    plt.hist(weights, bins=50, alpha=0.7)
                    plt.title(f"{name} - {layer_name}")
                    plt.xlabel("Weight Value")
                    plt.ylabel("Count")
                    break
        except Exception as e:
            print(f"Could not extract parameters for {name}: {e}")
    
    plt.tight_layout()
    plt.savefig("parameter_distributions.png")
    print("Saved parameter distributions plot to 'parameter_distributions.png'")

# Main function
def main():
    # Model paths
    model_dirs = [
        "../savings/model/minprotbert_epoch1",
        "../savings/model/minprotbert_epoch2",
        "../savings/model/minprotbert_epoch3"
    ]
    
    model_names = ["epoch1", "epoch2", "epoch3"]
    
    # Load tokenizer (assuming all models use the same vocabulary)
    tokenizer_path = "Rostlab/prot_bert"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    
    # Load models
    models = [load_model(path) for path in model_dirs]
    
    # Set models to eval mode
    for model in models:
        model.eval()
    
    # Compare model parameters
    are_different = compare_parameters(models, model_names)
    
    # Compare outputs on sample inputs
    compare_outputs(models, model_names, tokenizer)
    
    # Visualize parameter distributions
    visualize_parameters(models, model_names)
    
    print("\n===== Summary =====")
    if are_different:
        print("The models have different parameters, suggesting they represent different training checkpoints.")
    else:
        print("The models appear to be very similar or identical in their parameters.")

if __name__ == "__main__":
    main()
