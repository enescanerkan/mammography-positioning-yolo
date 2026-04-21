"""
Result saver for saving analysis results to files.
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional


class ResultSaver:
    """Class for saving analysis results to various file formats."""
    
    def __init__(self, output_dir: str = "results"):
        """Initialize the result saver.
        
        Args:
            output_dir: Directory to save results (default: "results")
        """
        self.output_dir = output_dir
        self._ensure_output_dir()
    
    def _ensure_output_dir(self):
        """Ensure output directory exists."""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
    
    def save(self, save_data: Dict[str, Any]) -> List[str]:
        """Save analysis results to files.
        
        Args:
            save_data: Dictionary containing results and metadata
            
        Returns:
            List of saved filenames
        """
        saved_files = []
        timestamp = save_data.get('timestamp', pd.Timestamp.now().strftime("%Y%m%d_%H%M%S"))
        
        text_filename = self._save_text_results(save_data, timestamp)
        if text_filename:
            saved_files.append(text_filename)
        
        csv_filename = self._save_csv_results(save_data, timestamp)
        if csv_filename:
            saved_files.append(csv_filename)
        
        json_filename = self._save_json_results(save_data, timestamp)
        if json_filename:
            saved_files.append(json_filename)
        
        return saved_files
    
    def _save_text_results(self, save_data: Dict[str, Any], timestamp: str) -> Optional[str]:
        """Save results as a detailed text file."""
        try:
            filename = os.path.join(self.output_dir, f"analysis_results_{timestamp}.txt")
            mlo_results = save_data.get('mlo_results')
            cc_results = save_data.get('cc_results')
            filenames = save_data.get('filenames', (None, None))
            
            with open(filename, 'w', encoding='utf-8') as f:
                f.write("MAMMOGRAM POSITIONING ANALYSIS RESULTS\n")
                f.write("="*50 + "\n")
                f.write(f"MLO File: {filenames[0]}\n")
                f.write(f"CC File: {filenames[1]}\n")
                f.write(f"Analysis Date: {pd.Timestamp.now()}\n\n")
                
                if mlo_results:
                    f.write("MLO ANALYSIS:\n")
                    f.write("-"*20 + "\n")
                    f.write(f"Pixel Spacing: {mlo_results['pixel_spacing']:.3f} mm/pixel\n")
                    landmarks = mlo_results['landmarks']
                    f.write(f"Pectoral 1: [{landmarks[0][0]:.1f}, {landmarks[0][1]:.1f}]\n")
                    f.write(f"Pectoral 2: [{landmarks[1][0]:.1f}, {landmarks[1][1]:.1f}]\n")
                    f.write(f"Nipple: [{landmarks[2][0]:.1f}, {landmarks[2][1]:.1f}]\n")
                    f.write(f"Distance (pixels): {mlo_results['distance_pixels']:.1f}\n")
                    f.write(f"Distance (mm): {mlo_results['distance_mm']:.2f}\n\n")
                
                if cc_results:
                    f.write("CC ANALYSIS:\n")
                    f.write("-"*20 + "\n")
                    f.write(f"Pixel Spacing: {cc_results['pixel_spacing']:.3f} mm/pixel\n")
                    landmarks = cc_results['landmarks']
                    f.write(f"Nipple: [{landmarks[0][0]:.1f}, {landmarks[0][1]:.1f}]\n")
                    f.write(f"Direction: {cc_results['direction']}\n")
                    f.write(f"Distance (pixels): {cc_results['distance_pixels']:.1f}\n")
                    f.write(f"Distance (mm): {cc_results['distance_mm']:.2f}\n\n")
                
                if mlo_results and cc_results:
                    difference = abs(mlo_results['distance_mm'] - cc_results['distance_mm'])
                    f.write("COMPARISON:\n")
                    f.write("-"*20 + "\n")
                    f.write(f"Difference: {difference:.2f} mm\n")
                    f.write(f"Quality: {'GOOD' if difference <= 10.0 else 'POOR'} (threshold: 10mm)\n")
            
            return filename
            
        except Exception as e:
            print(f"Error saving text results: {e}")
            return None
    
    def _save_csv_results(self, save_data: Dict[str, Any], timestamp: str) -> Optional[str]:
        """Save results as CSV file."""
        try:
            filename = os.path.join(self.output_dir, f"analysis_results_{timestamp}.csv")
            mlo_results = save_data.get('mlo_results')
            cc_results = save_data.get('cc_results')
            filenames = save_data.get('filenames', (None, None))
            
            data = {
                'timestamp': [pd.Timestamp.now()],
                'mlo_filename': [filenames[0]],
                'cc_filename': [filenames[1]]
            }
            
            if mlo_results:
                data.update({
                    'mlo_pixel_spacing': [mlo_results['pixel_spacing']],
                    'mlo_distance_pixels': [mlo_results['distance_pixels']],
                    'mlo_distance_mm': [mlo_results['distance_mm']],
                    'mlo_pectoral1_x': [mlo_results['landmarks'][0][0]],
                    'mlo_pectoral1_y': [mlo_results['landmarks'][0][1]],
                    'mlo_pectoral2_x': [mlo_results['landmarks'][1][0]],
                    'mlo_pectoral2_y': [mlo_results['landmarks'][1][1]],
                    'mlo_nipple_x': [mlo_results['landmarks'][2][0]],
                    'mlo_nipple_y': [mlo_results['landmarks'][2][1]]
                })
            
            if cc_results:
                data.update({
                    'cc_pixel_spacing': [cc_results['pixel_spacing']],
                    'cc_distance_pixels': [cc_results['distance_pixels']],
                    'cc_distance_mm': [cc_results['distance_mm']],
                    'cc_direction': [cc_results['direction']],
                    'cc_nipple_x': [cc_results['landmarks'][0][0]],
                    'cc_nipple_y': [cc_results['landmarks'][0][1]]
                })
            
            if mlo_results and cc_results:
                difference = abs(mlo_results['distance_mm'] - cc_results['distance_mm'])
                data.update({
                    'difference_mm': [difference],
                    'quality': ['GOOD' if difference <= 10.0 else 'POOR']
                })
            
            df = pd.DataFrame(data)
            df.to_csv(filename, index=False)
            
            return filename
            
        except Exception as e:
            print(f"Error saving CSV results: {e}")
            return None
    
    def _save_json_results(self, save_data: Dict[str, Any], timestamp: str) -> Optional[str]:
        """Save results as JSON file."""
        try:
            filename = os.path.join(self.output_dir, f"analysis_results_{timestamp}.json")
            
            json_data = {
                'timestamp': pd.Timestamp.now().isoformat(),
                'filenames': save_data.get('filenames', (None, None))
            }
            
            mlo_results = save_data.get('mlo_results')
            if mlo_results:
                json_data['mlo_results'] = self._convert_to_json_serializable(mlo_results.copy())
            
            cc_results = save_data.get('cc_results')
            if cc_results:
                json_data['cc_results'] = self._convert_to_json_serializable(cc_results.copy())
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)
            
            return filename
            
        except Exception as e:
            print(f"Error saving JSON results: {e}")
            return None
    
    def _convert_to_json_serializable(self, data):
        """Convert numpy types to JSON serializable types."""
        if isinstance(data, dict):
            return {key: self._convert_to_json_serializable(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [self._convert_to_json_serializable(item) for item in data]
        elif isinstance(data, np.ndarray):
            return data.tolist()
        elif isinstance(data, (np.float32, np.float64)):
            return float(data)
        elif isinstance(data, (np.int32, np.int64)):
            return int(data)
        elif isinstance(data, np.bool_):
            return bool(data)
        else:
            return data
