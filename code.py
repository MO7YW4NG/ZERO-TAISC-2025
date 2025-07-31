import os
import csv
import torch
import pandas as pd
from PIL import Image
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
from tqdm import tqdm

def load_and_preprocess_image(image_path):
    """Load and preprocess image for model input"""
    image = Image.open(image_path).convert('RGB')
    return image

def load_model_and_processor():
    """Load Qwen2.5-VL-72B-Instruct model and processor"""
    print("Loading Qwen2.5-VL-72B-Instruct model...")
    
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
    
    print("Model loaded successfully!")
    return model, processor

def intelligent_frame_sampling(frames, last_frames_count=6):
    """Strategic frame selection for temporal understanding"""
    len_frames = len(frames)
    
    # First quarter (25%) - captures initial traffic conditions
    first_frame = frames[len_frames//4]
    
    # Middle frame (50%) - shows mid-sequence developments
    middle_frame = frames[len_frames//2]
    
    # Last frames (sampled) - reveals final moments and critical developments
    if len_frames > last_frames_count * 2 + 1:
        last_frames = frames[-last_frames_count*2-1::2]
    else:
        last_frames = frames[len_frames//2+1:]
    
    return first_frame, middle_frame, last_frames

def create_advanced_prompt(first_frame, middle_frame, last_frames):
    """Advanced prompt engineering for hazard detection"""
    prompt = f"""Based on attached image sequence, representing different moments of the same dashcam video.
Focus on relative movement of vehicles, unnatural/unusual movement of any vehicles (like drastical change in speed, direction, or position), signs of possible collisions or near-misses
whether there be a potential hazard in the near future? Respond 'YES' or 'NO'. If unsure, return 'YES'.

Frame1 (Init): {first_frame}
Frame2 (Middle): {middle_frame}"""

    # Add last frames context
    for i, frame in enumerate(last_frames):
        prompt += f"\nFrame{3+i} (Last {len(last_frames)} Frames): {frame}"
    
    return prompt

def process_video_directory(dir_path, model, processor):
    """Process a single video directory using ZERO framework"""
    frames = os.listdir(dir_path)
    if not frames:
        return None, None
    
    # Intelligent frame sampling
    first_frame, middle_frame, last_frames = intelligent_frame_sampling(frames)
    
    # Load and preprocess images
    first_image = load_and_preprocess_image(os.path.join(dir_path, first_frame))
    middle_image = load_and_preprocess_image(os.path.join(dir_path, middle_frame))
    last_images = [load_and_preprocess_image(os.path.join(dir_path, frame)) for frame in last_frames]
    
    # Prepare images for batch processing
    images = [first_image, middle_image] + last_images
    
    # Create advanced prompt
    prompt = create_advanced_prompt(first_frame, middle_frame, last_frames)
    
    try:
        # Prepare inputs
        inputs = processor(
            text=prompt,
            images=images,
            return_tensors="pt"
        ).to(model.device)
        
        # Generate response with robust error handling
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
        
        return risk, response
        
    except Exception as e:
        print(f"Error during inference: {e}")
        return None, None

def main():
    """Main execution function for ZERO framework"""
    print("ZERO: Zero-shot Event Risk Observation Framework")
    print("=" * 50)
    
    # Load model and processor
    model, processor = load_model_and_processor()
    
    # Process test directories
    test_dirs = ["road/train", "freeway/train"]
    
    with open("submission.csv", "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["file_name", "risk"])
        
        total_processed = 0
        
        for test_dir in test_dirs:
            print(f"\nProcessing {test_dir}...")
            directories = os.listdir(test_dir)
            
            for directory in tqdm(directories, desc=f"Processing {test_dir}"):
                dir_path = os.path.join(test_dir, directory)
                if not os.path.isdir(dir_path):
                    continue
                
                risk, response = process_video_directory(dir_path, model, processor)
                
                if risk is not None:
                    print(f"{directory}[{risk}] : {response}")
                    writer.writerow([directory, risk])
                    total_processed += 1
    
    print(f"\nProcessing completed! Total videos processed: {total_processed}")
    
    # Evaluate performance if ground truth files exist
    evaluate_performance()

def evaluate_performance():
    """Evaluate performance using ground truth files"""
    print("\n" + "=" * 50)
    print("Performance Evaluation")
    print("=" * 50)
    
    # Read predictions
    try:
        pred_df = pd.read_csv("submission.csv", encoding="utf-8")
        pred_dict = dict(zip(pred_df["file_name"], pred_df["risk"]))
    except FileNotFoundError:
        print("No submission.csv file found.")
        return
    
    # Read ground truth
    gt_files = []
    for gt_csv in ["road_train.csv", "freeway_train.csv"]:
        if os.path.exists(gt_csv):
            gt_df = pd.read_csv(gt_csv)
            gt_files.append(gt_df)
    
    if not gt_files:
        print("No ground truth files found.")
        return
    
    gt_df = pd.concat(gt_files, ignore_index=True)
    gt_dict = dict(zip(gt_df["file_name"], gt_df["risk"]))
    
    # Match predictions and ground truth
    y_true = []
    y_pred = []
    for fname in pred_dict:
        if fname in gt_dict:
            y_true.append(gt_dict[fname])
            y_pred.append(pred_dict[fname])
    
    if not y_true:
        print("No matching files between predictions and ground truth.")
        return
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    try:
        auc = roc_auc_score(y_true, y_pred)
        print(f"AUC: {auc:.4f}")
    except Exception as e:
        print(f"AUC could not be computed: {e}")

if __name__ == "__main__":
    main()
