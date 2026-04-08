# STRIDE: Stimulus-guided Training for Reward Integration with Directional Exponential Smoothing

A reinforcement learning framework for financial sentiment classification using policy gradient optimization with EMA-smoothed REINFORCE.

## Usage

1. Use the uploaded XML data files.
2. Open `stride_absa.ipynb` and run cells
3. Adjust hyperparameters in the notebook as needed

## How It Works

- Policy network (T5) generates keywords
- Frozen reward model (LLaMA) evaluates sentiment  
- REINFORCE with EMA baseline smoothing optimizes keyword selection
- Learning objective: maximize prediction margin via policy gradients
