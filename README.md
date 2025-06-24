# Stability-Gap-SAM
This code provides evaluation pipeline for sharpness-aware optimizers in continual learning setting. Specifically, effect on stability gap is evaluated. 
## Entropy-SGD
Using default base optimizer (SGD):
```bash
python main.py \
    --optimizer "Entropy-SGD" \
    --lr 0.1 \
    --num_iters 1000 \
    --scale 0.001 \
    --L 15
```

If you want to use Adam as default optimizer:
```bash
python main.py \
    --optimizer "Entropy-SGD" \
    --lr 0.1 \
    --num_iters 1000 \
    --scale 0.001 \
    --L 15 \
    --base_optimizer Adam \
    --base_optimizer_lr 0.01
```

## C-Flat 
```bash
python main.py \
    --optimizer C-Flat \
    --lr 0.1 \
    --num_iters 1000 \
    --rho 0.05 \
    --lamb 2.0
```

To use with Adam run
```bash
python main.py \
    --optimizer C-Flat \
    --lr 0.1 \
    --num_iters 1000 \
    --rho 0.05 \
    --lamb 2.0 \
    --base_optimizer Adam \
    --base_optimizer_lr 0.01
```