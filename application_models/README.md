# PIN Digit Recognition - Model Training Pipeline

Training pipeline for handwritten digit recognition on RPI Pico RP2040.

## Requirements

```bash
pip install tensorflow tensorflow-model-optimization numpy
```

## Quick Start

```bash
# Train MCUNet-tiny with early exit + QAT
python train.py --model mcunet_tiny --samples-dir ../samples

# Export to INT8 TFLite + C header
python export.py checkpoints/mcunet_tiny_ee_final.keras --output-dir exports

# Evaluate and compare models
python evaluate.py checkpoints/*.keras --samples-dir ../samples
```

## Architecture

### MCUNet Tiny (recommended)
- Depthwise separable convolutions
- ~26K params, ~30KB INT8
- Early exit after block 1

### TFLM CNN
- Standard Conv2D (better TFLM compatibility)
- Variants: tiny (~15K params), small (~25K params)

## Data Pipeline

- **MNIST**: 54K train / 6K val / 10K test
- **Custom samples**: 6 train / 2 val / 1 test per digit
- **Augmentation**: Rotation (±15°), shift (±2px), zoom (0.9-1.1x)

## Output Files

```
checkpoints/
├── {model}_best.keras      # Best validation checkpoint
├── {model}_final.keras     # Final trained model
├── {model}_history.json    # Training history
└── {model}_config.json     # Config used

exports/
├── {model}_int8.tflite     # INT8 quantized (deploy this)
├── {model}_float.tflite    # Float32 for comparison
├── {model}_model.h         # C header for Pico
└── {model}_metadata.json   # Size/accuracy stats
```

## Pico Integration

Copy the generated `.h` file to your Pico project:

```c
#include "mcunet_tiny_model.h"

// Model is embedded as: mcunet_tiny_model_model[]
// Length: mcunet_tiny_model_model_len
```

## Target Specs

| Metric | Target | MCUNet Tiny |
|--------|--------|-------------|
| Size (INT8) | <50KB | ~30KB |
| Inference | <10ms | <5ms |
| Accuracy | >97% | ~98% |
| RAM usage | <100KB | ~50KB |
