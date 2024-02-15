import logging
from logging.handlers import RotatingFileHandler

# Define the logging configuration
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),  # Console logging
            RotatingFileHandler(      # File logging with rotation
                '/tmp/minihf.log',
                maxBytes=10*1024*1024,  # 10 MB
                backupCount=5
            )
        ]
    )

# Call the setup_logging function to set up the logger
setup_logging()