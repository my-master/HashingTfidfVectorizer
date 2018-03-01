import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)
fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')
file = logging.FileHandler('log.txt')
file.setFormatter(fmt)
logger.addHandler(file)

# console = logging.StreamHandler()
# console.setFormatter(fmt)
# logger.addHandler(console)
