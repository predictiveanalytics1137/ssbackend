import logging
import os

# # Create a logs directory if it doesn't exist
# os.makedirs("logs", exist_ok=True)

# # Configure logging
# logging.basicConfig(
#     filename="logs/app.log",  # Log file location
#     filemode="a",             # Append to the log file
#     format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
#     level=logging.INFO        # Set the logging level
# )

# # Function to get logger
# def get_logger(module_name):
#     logger = logging.getLogger(module_name)
#     return logger



import logging
import os

# Create logs directory if it doesn't exist
os.makedirs("logs", exist_ok=True)

# Configure root logger only once
if not logging.getLogger('').handlers:
    logging.basicConfig(
        filename="logs/app.log",
        filemode="a",
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO
    )

def get_logger(module_name):
    logger = logging.getLogger(module_name)
    # Ensure propagation to root logger
    logger.propagate = True
    # Avoid adding handlers if already configured by root
    if not logger.handlers:
        handler = logging.FileHandler("logs/app.log")
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger