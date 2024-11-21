import logging

def setup_logger(name: str, log_file: str, level=logging.INFO):
    """Setup logger with file handler and stream handler."""
    logger = logging.getLogger(name)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.setLevel(level)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger

# Example: logger = setup_logger('chatbot', 'chatbot.log')

