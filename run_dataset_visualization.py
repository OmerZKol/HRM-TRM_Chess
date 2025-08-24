"""
Run chess dataset visualization and inspection.
"""

import os
import sys

def main():
    """Run dataset visualization."""
    
    dataset_path = "data/chess-move-prediction"
    
    print("Chess Dataset Visualization Tool")
    print("=" * 40)
    
    # Check if dataset exists
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset not found at {dataset_path}")
        print("Please run build_chess_dataset.py first to create the dataset.")
        return
    
    print(f"📂 Dataset found at: {dataset_path}")
    
    # Ask user which tool to run
    print("\nAvailable tools:")
    print("1. Quick inspection (text-only)")
    print("2. Full visualization (with plots)")
    print("3. Both")
    
    try:
        choice = input("Enter choice (1-3): ").strip()
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        return
    
    if choice in ["1", "3"]:
        print("\n🔍 Running quick inspection...")
        try:
            from inspect_chess_dataset import inspect_chess_dataset
            inspect_chess_dataset(dataset_path)
        except Exception as e:
            print(f"❌ Error running inspection: {e}")
    
    if choice in ["2", "3"]:
        print("\n🎨 Running full visualization...")
        try:
            from visualize_chess_dataset import ChessDatasetVisualizer
            
            visualizer = ChessDatasetVisualizer(dataset_path)
            visualizer.generate_report()
            
            # Ask if user wants plots
            try:
                show_plots = input("\nGenerate plots? (y/n): ").strip().lower()
                if show_plots in ['y', 'yes']:
                    print("Generating plots...")
                    if visualizer.train_data:
                        visualizer.plot_move_distribution("train")
                        visualizer.plot_position_statistics("train")
                    if visualizer.test_data:
                        visualizer.plot_move_distribution("test")
                        visualizer.plot_position_statistics("test")
            except KeyboardInterrupt:
                print("\nSkipping plots.")
                
        except ImportError as e:
            print(f"❌ Could not import visualization modules: {e}")
            print("Make sure matplotlib and seaborn are installed:")
            print("pip install matplotlib seaborn")
        except Exception as e:
            print(f"❌ Error running visualization: {e}")
    
    print("\n✨ Done!")


if __name__ == "__main__":
    main()
