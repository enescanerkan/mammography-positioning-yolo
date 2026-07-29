import os
import sys
from pathlib import Path

# Provide access to the src package from within the scripts directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))
from src.preprocessing.pipeline import PreprocessingPipeline
from src.preprocessing.strategies import AdvancedWaveletStrategy
from src.augmentation.flip_augmenter import FlipAugmenter
from src.utils.logger import get_logger

logger = get_logger("DatasetGenerator")

def generate_full_dataset():
    """
    Automates the generation of the entire dataset (MLO and CC),
    using the advanced Strategy, followed by automatic data augmentation.
    """
    # 1. Configuration
    # Set this to your local VinDr-Mammo `images/` directory.
    data_path = r"/path/to/vindr-mammo/images"
    output_dir = str(PROJECT_ROOT / "data" / "processed" / "advanced")
    
    label_files = [
        str(PROJECT_ROOT / "data" / "labels" / "mlo_labels.csv"), 
        str(PROJECT_ROOT / "data" / "labels" / "cc_labels.csv")
    ]
    
    # We choose the V3 winning strategy
    strategy = AdvancedWaveletStrategy()
    
    # 2. Execution Loop
    for view in ['MLO', 'CC']:
        logger.info(f"==================================================")
        logger.info(f" PHASE 1: PREPROCESSING {view} IMAGES")
        logger.info(f"==================================================")
        
        view_output_dir = Path(output_dir) / view
        
        pipeline = PreprocessingPipeline(
            data_dir=data_path, 
            output_dir=str(view_output_dir), 
            labels_csv=label_files, 
            strategy=strategy
        )
        pipeline.run(view_mode=view)
        
        logger.info(f"==================================================")
        logger.info(f" PHASE 2: AUGMENTATION (FLIP) {view} IMAGES")
        logger.info(f"==================================================")
        
        augmenter = FlipAugmenter(str(view_output_dir), view_type=view)
        
        # Log before split sizes
        before_stats = augmenter.get_statistics()
        logger.info(f"Before augmentation: {before_stats}")
        
        # Run Augmentation on all splits
        results = augmenter.run(splits=['train', 'val', 'test'])
        
        # Log after split sizes
        after_stats = augmenter.get_statistics()
        logger.info(f"After augmentation: {after_stats}")
        
        logger.info(f"Successfully finished processing {view}!")

if __name__ == "__main__":
    generate_full_dataset()
