"""
Example usage of PPO inference optimization with Triton.
"""
import numpy as np
from src.triton.ppo_inference import PPOInferenceOptimizer

def main():
    # Load optimizer with config
    optimizer = PPOInferenceOptimizer("config/model/transformer_config.yaml")

    # Create sample input data
    batch_size = 32
    seq_len = 128
    hidden_dim = optimizer.hidden_dim

    # Create random input tensors
    input_ids = np.random.randn(batch_size, seq_len, hidden_dim).astype(np.float32)
    attention_mask = np.ones((batch_size, seq_len), dtype=np.float32)
    old_logits = np.random.randn(batch_size, seq_len, hidden_dim).astype(np.float32)

    # Run PPO-optimized inference
    output = optimizer.optimize_batch(
        input_ids=input_ids,
        attention_mask=attention_mask,
        old_logits=old_logits
    )

    print("Inference completed successfully!")
    print(f"Output shape: {output.shape}")
    print(f"Output dtype: {output.dtype}")

if __name__ == "__main__":
    main()
