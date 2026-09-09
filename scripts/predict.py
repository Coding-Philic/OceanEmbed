import argparse
from pathlib import Path
from oceanembed.inference.predict import OceanEmbedPredictor

def main():
    parser = argparse.ArgumentParser(description="Run OceanEmbed 3D Predictions")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .ckpt file")
    parser.add_argument("--config", type=str, default="configs/poc_bob.yaml", help="Path to config")
    parser.add_argument("--year", type=int, default=2023, help="Year of data to predict")
    args = parser.parse_args()

    print(f"Loading AI Model from {args.checkpoint}...")
    
    # Initialize the predictor
    predictor = OceanEmbedPredictor(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        stats_path="data/processed/normalization_stats.json",
        device="mps"  # Using Apple Silicon GPU
    )

    output_file = Path(f"outputs/predictions_{args.year}.nc")
    
    print(f"\nRunning 3D prediction on {args.year} surface data...")
    print("This will take the 2D surface maps and generate the 3D deep ocean volume.")
    
    # Run the prediction
    predictor.predict_dataset(
        data_dir="data/aligned",
        years=[args.year],
        batch_size=8,  # Batch size for Mac
        output_path=output_file
    )

    print(f"\n✅ SUCCESS! 3D Predictions saved to {output_file}")
    print("You can now visualize this file in your frontend using Plotly.js or Three.js!")

if __name__ == "__main__":
    main()
