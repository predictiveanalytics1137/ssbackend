import logging
import os

# Create a logs directory if it doesn't exist
os.makedirs("logs", exist_ok=True)

# Configure logging
logging.basicConfig(
    filename="logs/app.log",  # Log file location
    filemode="a",             # Append to the log file
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO        # Set the logging level
)

# Function to get logger
def get_logger(module_name):
    logger = logging.getLogger(module_name)
    return logger
