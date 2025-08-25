import logging
import sys
from pathlib import Path

def setup_logging(level="INFO", format_string=None):
    if format_string is None:
        format_string = "%(asctimes)s - %(name)s - %(levelname)s - %(message)s"
        
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=format_string,
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    
    return logging.getLogger()
        