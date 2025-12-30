"""Script to run the FastAPI server"""

import uvicorn
import yaml
from pathlib import Path

if __name__ == "__main__":
    # Load config
    config_path = Path(__file__).parent / "config" / "config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    uvicorn.run(
        "src.api.main:app",
        host=config['api']['host'],
        port=config['api']['port'],
        reload=config['api']['reload']
    )


