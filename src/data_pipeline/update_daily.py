import os
import logging
from src.data_pipeline.database_manager import DatabaseManager

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logger.info("Initializing daily database sync...")
    db = DatabaseManager()
    db.update_latest_data()
    logger.info("Daily database sync completed successfully.")