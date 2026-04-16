import logging

logging.basicConfig(level=logging.INFO)

def log_step(step, data=None):
    logging.info({"step": step, "data": data})
