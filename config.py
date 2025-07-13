#!/usr/bin/env python3
"""
Configuration management for Face Recognition System

This module handles all configuration settings and provides easy customization.
"""

import os
import json
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional

@dataclass
class ModelConfig:
    """Model configuration settings"""
    image_size: int = 160
    margin: int = 20
    pretrained: str = 'vggface2'
    min_detection_confidence: float = 0.9
    unknown_threshold: float = 40.0
    svm_kernel: str = 'linear'
    svm_c: float = 1.0
    embedding_size: int = 512

@dataclass
class UIConfig:
    """User interface configuration"""
    theme: str = 'cyberpunk'
    primary_color: str = '#00ffff'
    secondary_color: str = '#ff3333'
    background_color: str = '#0a0a1a'
    text_color: str = '#ffffff'
    animation_enabled: bool = True
    sidebar_expanded: bool = True

@dataclass
class ProcessingConfig:
    """Processing configuration"""
    batch_size: int = 32
    max_image_size: int = 1920
    supported_formats: list = None
    enable_gpu: bool = True
    num_workers: int = 4
    cache_enabled: bool = True

    def __post_init__(self):
        if self.supported_formats is None:
            self.supported_formats = ['.jpg', '.jpeg', '.png', '.bmp']

@dataclass
class DeploymentConfig:
    """Deployment configuration"""
    port: int = 8501
    host: str = 'localhost'
    debug_mode: bool = False
    log_level: str = 'INFO'
    max_upload_size: int = 200  # MB
    enable_cors: bool = False

class ConfigManager:
    """Configuration manager for the face recognition system"""
    
    def __init__(self, config_file: str = 'config.json'):
        self.config_file = config_file
        self.model = ModelConfig()
        self.ui = UIConfig()
        self.processing = ProcessingConfig()
        self.deployment = DeploymentConfig()
        
        self.load_config()
    
    def load_config(self):
        """Load configuration from file"""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    data = json.load(f)
                
                # Update configurations
                if 'model' in data:
                    self._update_dataclass(self.model, data['model'])
                if 'ui' in data:
                    self._update_dataclass(self.ui, data['ui'])
                if 'processing' in data:
                    self._update_dataclass(self.processing, data['processing'])
                if 'deployment' in data:
                    self._update_dataclass(self.deployment, data['deployment'])
                
                print(f"✅ Configuration loaded from {self.config_file}")
                
            except (json.JSONDecodeError, KeyError) as e:
                print(f"⚠️ Error loading config: {e}, using defaults")
        else:
            print(f"📝 Config file {self.config_file} not found, using defaults")
            self.save_config()  # Create default config file
    
    def save_config(self):
        """Save current configuration to file"""
        config_data = {
            'model': asdict(self.model),
            'ui': asdict(self.ui),
            'processing': asdict(self.processing),
            'deployment': asdict(self.deployment)
        }
        
        try:
            with open(self.config_file, 'w') as f:
                json.dump(config_data, f, indent=2)
            print(f"💾 Configuration saved to {self.config_file}")
        except Exception as e:
            print(f"❌ Error saving config: {e}")
    
    def _update_dataclass(self, instance, data: Dict[str, Any]):
        """Update dataclass instance with dictionary data"""
        for key, value in data.items():
            if hasattr(instance, key):
                setattr(instance, key, value)
    
    def get_streamlit_config(self) -> Dict[str, Any]:
        """Get configuration for Streamlit"""
        return {
            'page_title': '🎯 Face Recognition System',
            'page_icon': '🎯',
            'layout': 'wide',
            'initial_sidebar_state': 'expanded' if self.ui.sidebar_expanded else 'collapsed'
        }
    
    def get_model_paths(self) -> Dict[str, str]:
        """Get model file paths"""
        return {
            'svm_model': 'svm_model.pkl',
            'label_encoder': 'label_encoder.pkl',
            'embeddings': 'embeddings.npy',
            'labels': 'labels.npy'
        }
    
    def get_css_theme(self) -> str:
        """Get CSS theme based on configuration"""
        if self.ui.theme == 'cyberpunk':
            return f"""
            <style>
                .main-header {{
                    background: linear-gradient(45deg, #1a1a3a, #2e2e5c);
                    padding: 2rem;
                    border-radius: 10px;
                    margin-bottom: 2rem;
                    text-align: center;
                    border: 2px solid {self.ui.primary_color};
                }}
                
                .stApp {{
                    background: linear-gradient(135deg, {self.ui.background_color}, #1a1a3a);
                    color: {self.ui.text_color};
                }}
                
                .detection-box {{
                    border: 2px solid {self.ui.primary_color};
                    border-radius: 8px;
                    padding: 1rem;
                    margin: 1rem 0;
                    background: rgba(0, 255, 255, 0.1);
                }}
                
                .unknown-box {{
                    border: 2px solid {self.ui.secondary_color};
                    border-radius: 8px;
                    padding: 1rem;
                    margin: 1rem 0;
                    background: rgba(255, 51, 51, 0.1);
                }}
            </style>
            """
        else:
            return ""  # Default theme
    
    def update_setting(self, section: str, key: str, value: Any):
        """Update a specific setting"""
        if hasattr(self, section):
            section_obj = getattr(self, section)
            if hasattr(section_obj, key):
                setattr(section_obj, key, value)
                self.save_config()
                return True
        return False
    
    def reset_to_defaults(self):
        """Reset all settings to defaults"""
        self.model = ModelConfig()
        self.ui = UIConfig()
        self.processing = ProcessingConfig()
        self.deployment = DeploymentConfig()
        self.save_config()
        print("🔄 Configuration reset to defaults")

# Global configuration instance
config = ConfigManager()

# Convenience functions
def get_config() -> ConfigManager:
    """Get the global configuration instance"""
    return config

def update_config(section: str, **kwargs):
    """Update configuration settings"""
    for key, value in kwargs.items():
        config.update_setting(section, key, value)

# Environment-specific configurations
def setup_cloud_config():
    """Setup configuration for cloud deployment"""
    config.update_setting('deployment', 'host', '0.0.0.0')
    config.update_setting('deployment', 'debug_mode', False)
    config.update_setting('processing', 'enable_gpu', False)  # Most cloud services don't have GPU
    config.update_setting('processing', 'cache_enabled', True)

def setup_local_config():
    """Setup configuration for local development"""
    config.update_setting('deployment', 'host', 'localhost')
    config.update_setting('deployment', 'debug_mode', True)
    config.update_setting('processing', 'enable_gpu', True)

def setup_production_config():
    """Setup configuration for production"""
    config.update_setting('deployment', 'debug_mode', False)
    config.update_setting('deployment', 'log_level', 'WARNING')
    config.update_setting('processing', 'cache_enabled', True)
    config.update_setting('ui', 'animation_enabled', False)  # Better performance

if __name__ == "__main__":
    # Example usage
    print("🔧 Face Recognition System Configuration")
    print("=" * 50)
    
    print(f"Model Config: {config.model}")
    print(f"UI Config: {config.ui}")
    print(f"Processing Config: {config.processing}")
    print(f"Deployment Config: {config.deployment}")
    
    # Example of updating a setting
    config.update_setting('model', 'unknown_threshold', 35.0)
    print(f"Updated unknown threshold: {config.model.unknown_threshold}")
