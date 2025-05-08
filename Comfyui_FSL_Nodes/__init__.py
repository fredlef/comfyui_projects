import importlib
import os

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# Load all node modules
current_dir = os.path.dirname(__file__)
for filename in os.listdir(current_dir):
    if filename.endswith(".py") and filename != "__init__.py":
        module_name = filename[:-3]
        module = importlib.import_module(f".{module_name}", __package__)
        
        if hasattr(module, "NODE_CLASS_MAPPINGS"):
            # Add category override here
            for cls_name, cls in module.NODE_CLASS_MAPPINGS.items():
                cls.CATEGORY = "FSL Nodes"  # Your unified category name
                
            NODE_CLASS_MAPPINGS.update(module.NODE_CLASS_MAPPINGS)
            
        if hasattr(module, "NODE_DISPLAY_NAME_MAPPINGS"):
            NODE_DISPLAY_NAME_MAPPINGS.update(module.NODE_DISPLAY_NAME_MAPPINGS)

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
