import json
from typing import Dict

from app.utils.logging_utils import get_logger

logger = get_logger(__name__)


# Load feature flags from a configuration file
def load_feature_flags(flag_type="chat"):
    try:
        if flag_type == "chat":
            file_path = 'chatfeatureflags.json'
        elif flag_type == "indexer":
            file_path = 'indexerfeatureflags.json'
        else:
            return {}
        
        logger.debug(f"Loading feature flags from {file_path}")
        with open(file_path, 'r') as file:
            content = file.read()
            logger.debug(f"Feature flags file content: {content}")
            flags = json.loads(content)
            logger.debug(f"Parsed feature flags: {flags}")
            return flags
    except FileNotFoundError as e:
        logger.error(f"Feature flags file not found: {e}")
        return {}
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error in feature flags: {e}")
        return {}

def load_json(json_file) -> Dict[str, str]:
    """
    Load QA pairs from a JSON file and cache them in memory.

    Args:
        json_file (str): Path to the JSON file
    """
    qa_data = {}
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Build a dictionary with questions as keys and answers as values
        for item in data:
            qa_data[item['Prompt']] = item['Expected Result']

        logger.debug(f"Loaded {len(qa_data)} QA pairs into cache")
    except Exception as e:
        logger.error(f"Error loading JSON file: {e}")

    return qa_data

def write_data_logfile (context, data, log_file):
    json_data = json.dumps(data)
    logger.debug(f"DEBUG: Writing LLM {context} log to {log_file}, data length: {len(json_data)}")

    # Direct file write as a fallback mechanism
    with open(log_file, "a") as f:
        f.write(json_data + "\n")

    return json_data