# ZERO: Zero-shot Event Risk Observation Framework for Taiwan Traffic Risk Accident Predict

This repository contains our submission for the 2025 Taiwan Future Accident Classification Challenge (TAISC). Our ZERO framework achieved a weighted score of 0.5928 in the competition, demonstrating exceptional performance in zero-shot traffic accident risk classification.

## System Architecture

Our system leverages the Qwen2.5-VL-72B-Instruct model to perform zero-shot traffic accident risk classification through:

1. **Video Sampling** - Strategic frame selection from video sequences
2. **Prompt Engineering** - Specialized prompts for hazard detection
3. **Multi-Frame Analysis** - Temporal understanding across key video moments

### Architecture Diagram

```
Input: Dashcam Video Sequences
    ↓
Frame Sampling
    ↓
┌────────────────────────────────────────────────────┐
│           Qwen2.5-VL-72B-Instruct Model            │
│                                                    │
│    ┌─────────────────┐     ┌─────────────────┐     │
│    │  Frame 1        │     │  Frame 2        │     │
│    │  (First 25%)    │     │  (Middle 50%)   │     │
│    └─────────────────┘     └─────────────────┘     │
│                                                    │
│  ┌──────────────────────────────────────────────┐  │
│  │  Last 6 Frames (Sampled)                     │  │
│  └──────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────┘
    ↓
Prompt Engineering
    ↓
Risk Classification (YES/NO)
    ↓
Submission Generation
```

## Dataset Structure

```
AVA Dataset/
├── road/
│   ├── train/
│   │   ├── road_0000/
│   │   │   ├── 00001.jpg
│   │   │   └── ...
│   │   └── ...
│   └── test/
│       ├── road_0179/
│       └── ...
├── freeway/
│   ├── train/
│   └── test/
├── road_train.csv
├── freeway_train.csv
└── sample_submission.csv
```

## Implementation

### Core Strategy: Vision-Language Model Analysis

#### 1. Intelligent Video Sampling Strategy

**Multi-Point Frame Selection:**
```python
# Strategic frame selection for temporal understanding
first_frame = frames[len_frames//4]      # First quarter (25%)
middle_frame = frames[len_frames//2]     # Middle frame (50%)
last_frames = frames[-last_frames_count*2-1::2]  # Last 6 frames (sampled)
```

**Sampling Rationale:**
- **First Frame (25%)**: Captures initial traffic conditions and vehicle positions
- **Middle Frame (50%)**: Shows mid-sequence developments and potential changes
- **Last Frames (Sampled)**: Reveals final moments and critical developments

#### 2. Advanced Prompt Engineering Strategy

**Primary Prompt Design:**
```python
prompt = f"""Based on attached image sequence, representing different moments of the same dashcam video.
Focus on relative movement of vehicles, unnatural/unusual movement of any vehicles (like drastical change in speed, direction, or position), signs of possible collisions or near-misses
whether there be a potential hazard in the near future? Respond 'YES' or 'NO'. If unsure, return 'YES'.

Frame1 (Init): {first_frame}
Frame2 (Middle): {middle_frame}"""
```

**Prompt Engineering Strategy:**
- **Clear Task Definition**: Explicit hazard detection objective
- **Specific Focus Areas**: Vehicle movement, speed changes, collision signs
- **Binary Output**: YES/NO classification for consistency
- **Uncertainty Handling**: Default to "YES" for safety
- **Frame Context**: Temporal labeling for model understanding

#### 3. Model Integration Strategy

**Model Loading and Configuration:**
```python
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor

# Load model and processor
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-VL-72B-Instruct",
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
)
processor = AutoProcessor.from_pretrained(
    "Qwen/Qwen2.5-VL-72B-Instruct",
    trust_remote_code=True
)
```

#### 4. Robust Error Handling Strategy

**Model Inference with Error Recovery:**
```python
try:
    # Prepare inputs
    inputs = processor(
        text=prompt,
        images=images,
        return_tensors="pt"
    ).to(model.device)
    
    # Generate response
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=True,
            temperature=0.7
        )
    
    # Decode response
    response = processor.batch_decode(
        generated_ids[:, inputs.input_ids.shape[1]:],
        skip_special_tokens=True
    )[0]
    
except Exception as e:
    print(f"Error during inference: {e}")
    # Fallback handling
```

### Video Processing Pipeline

#### 1. Frame Extraction and Processing

```python
from PIL import Image
import torch

def load_and_preprocess_image(image_path):
    """Load and preprocess image for model input"""
    image = Image.open(image_path).convert('RGB')
    return image

# Process multiple frames
first_image = load_and_preprocess_image(os.path.join(dir_path, first_frame))
middle_image = load_and_preprocess_image(os.path.join(dir_path, middle_frame))
last_images = [load_and_preprocess_image(os.path.join(dir_path, frame)) for frame in last_frames]
```

#### 2. Multi-Modal Input Construction

```python
# Prepare images for batch processing
images = [first_image, middle_image] + last_images

# Construct prompt with frame context
prompt = f"""Based on attached image sequence, representing different moments of the same dashcam video.
Focus on relative movement of vehicles, unnatural/unusual movement of any vehicles (like drastical change in speed, direction, or position), signs of possible collisions or near-misses
whether there be a potential hazard in the near future? Respond 'YES' or 'NO'. If unsure, return 'YES'.

Frame1 (Init): {first_frame}
Frame2 (Middle): {middle_frame}"""

# Add last frames context
for i, frame in enumerate(last_frames):
    prompt += f"\nFrame{3+i} (Last {len(last_frames)} Frames): {frame}"
```

#### 3. Model Inference and Response Processing

```python
# Process inputs
inputs = processor(
    text=prompt,
    images=images,
    return_tensors="pt"
).to(model.device)

# Generate response
with torch.no_grad():
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=128,
        do_sample=True,
        temperature=0.7,
        pad_token_id=processor.tokenizer.eos_token_id
    )

# Decode response
response = processor.batch_decode(
    generated_ids[:, inputs.input_ids.shape[1]:],
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False
)[0]

# Process response
answer = response.strip().upper()
risk = 1 if "YES" in answer else 0
```

## Competition Results

Our approach achieved the following performance metrics in the TAISC 2025 competition:

| Metric | Value | Description |
|--------|-------|-------------|
| **Score** | **0.5928** | Overall competition score |
| **F1 Score** | **0.5128** | Harmonic mean of precision and recall |
| **Accuracy** | **0.6911** | Overall classification accuracy |
| **AUC** | **0.6455** | Area under the ROC curve |


## System Requirements

### Hardware Requirements

- **VRAM**: Minimum 160GB
- **RAM**: Minimum 64GB system RAM (128GB recommended)
- **Storage**: ~144GB

### Software Requirements

- **Python**: 3.8+
- **CUDA**: 11.8+ (for GPU acceleration)

## Installation and Usage

### Environment Setup

```bash
# Create conda environment
conda create -n taisc2025 python=3.12
conda activate taisc2025

# Install dependencies
pip install torch torchvision transformers
pip install pandas numpy scikit-learn
pip install pillow tqdm
```

### Model Download

The Qwen2.5-VL-72B-Instruct model will be automatically downloaded from Hugging Face on first use:

```python
# Model will be downloaded automatically
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-VL-72B-Instruct",
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
)
```

**Output:**
- `submission.csv` - Generated predictions
- Console output showing real-time processing
- Performance metrics on completion

### Customization

**Adjusting Frame Sampling:**
```python
# Modify sampling strategy
last_frames_count = 8  # Increase last frames
first_frame = frames[len_frames//3]  # Change first frame position
```

**Model Parameters:**
```python
# Adjust generation parameters
generated_ids = model.generate(
    **inputs,
    max_new_tokens=256,  # Increase for longer responses
    temperature=0.3,     # More conservative
    do_sample=True,
    top_p=0.9,          # Nucleus sampling
    repetition_penalty=1.1
)
```

**Image Resolution Control:**
```python
# Optimize for performance vs quality
processor = AutoProcessor.from_pretrained(
    "Qwen/Qwen2.5-VL-72B-Instruct",
    min_pixels=256 * 28 * 28,    # Minimum resolution
    max_pixels=1280 * 28 * 28,   # Maximum resolution
    trust_remote_code=True
)
```

## Technical Strategy

### Advanced Techniques

1. **Intelligent Frame Sampling**
   - Multi-point temporal sampling
   - Adaptive frame selection based on video length
   - Strategic coverage of video timeline

2. **Professional Prompt Engineering**
   - Task-specific instruction design
   - Uncertainty handling strategies
   - Clear output format specification

3. **Robust Model Integration**
   - Direct Hugging Face model loading
   - Automatic device mapping
   - Memory-efficient inference

4. **Temporal Understanding**
   - Multi-frame sequence analysis
   - Relative movement detection
   - Temporal context preservation

### Key Advantages

1. **Zero-Shot Learning**: No training data required
2. **Interpretability**: Clear reasoning through prompts
3. **Scalability**: Easy to adapt to new scenarios
4. **Robustness**: Handles various video qualities and lengths

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{zero-taisc2025,
  title={ZERO: Zero-shot Event Risk Observation Framework for Taiwan Traffic Risk Accident Predict},
  author={Wang, Min-Quan and Wu, Bo-Ching},
  year={2025},
  howpublished={TAISC 2025 Challenge},
  url={https://github.com/MO7YW4NG/ZERO-TAISC-2025},
  institution={Chung Yuan Christian University, Department of Information Management}
}

@misc{qwen2.5-VL,
    title = {Qwen2.5-VL},
    url = {https://qwenlm.github.io/blog/qwen2.5-vl/},
    author = {Qwen Team},
    month = {January},
    year = {2025}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Qwen2.5-VL-72B-Instruct model by Alibaba
- HuggingFace for model hosting and distribution
- Transformers library for model integration
- TAISC 2025 competition organizers for providing the challenge and dataset
