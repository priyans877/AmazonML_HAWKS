import os
import pandas as pd
import easyocr
import re
from tqdm import tqdm
import logging
from src.constants import entity_unit_map
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def load_ocr_reader():
    logging.info("loading OCR reader...")
    try:
        reader = easyocr.Reader(['en'])
        logging.info("OCR reader loaded")
        return reader
    except Exception as e:
        logging.error(f"failed to load OCR reader: {str(e)}")
        raise

def ocr_on_image(image_path, reader):
    logging.info(f"processing image: {image_path}")
    try:
        if not os.path.exists(image_path):
            logging.error(f"image file not found: {image_path}")
            return ""
        result = reader.readtext(image_path)
        text = ' '.join([text for _, text, _ in result])
        logging.debug(f"OCR result: {text}")
        return text
    except Exception as e:
        logging.error(f"error in OCR processing for {image_path}: {str(e)}")
        return ""

def convert_to_standard_unit(value, unit, entity_name):
    return float(value), unit

def extract_entity_value(text, entity_name):
    logging.debug(f"Extracting entity value for {entity_name} from: {text}")
    allowed_units = entity_unit_map.get(entity_name, set())
    #create a pattern that matches numbers followed by any unit, not just allowed ones
    pattern = r'(\d+(\.\d+)?)\s*(cm|inch|"|in|\w+)'
    matches = re.findall(pattern, text, re.IGNORECASE)
    
    valid_measurements = []
    print(matches)
    for match in matches:
        value, _, unit = match
        # Normalize units
        if unit == '"' or unit.lower() == 'in':
            unit = 'inch'
        
        if unit == '"' or unit.lower() == 'w':
            unit = 'watt'
        if unit == '"' or unit.lower() == 'v':
            unit = 'volt'
        elif unit.lower() == 'cm':
            unit = 'centimetre'
        
        if unit.lower() in allowed_units:
            converted_value, converted_unit = convert_to_standard_unit(value, unit.lower(), entity_name)
            valid_measurements.append((converted_value, converted_unit))
    
    if valid_measurements:
        #choose the first valid measurement (you might want to implement a different selection logic)
        value, unit = valid_measurements[0]
        result = f"{value:.2f} {unit}"
        logging.debug(f"Extracted value: {result}")
        return result
    
    logging.debug(f"No valid value extracted for {entity_name}")
    return ""

def predictor(image_path, category_id, entity_name, reader):
    logging.info(f"Predicting for image: {image_path}, category: {category_id}, entity: {entity_name}")
    try:
        ocr_text = ocr_on_image(image_path, reader)
        result = extract_entity_value(ocr_text, entity_name)
        logging.info(f"Prediction result: {result}")
        return result
    except Exception as e:
        logging.error(f"Error in predictor for {image_path}: {str(e)}")
        return ""

if __name__ == "__main__":
    DATASET_FOLDER = 'dataset'
    IMAGE_FOLDER = 'images'
    
    logging.info("Script started")
    
    try:
        logging.info("Loading test data...")
        test = pd.read_csv(os.path.join(DATASET_FOLDER, 'test.csv'))
        logging.info(f"Loaded {len(test)} test samples")
        
        #load OCR
        reader = load_ocr_reader()
        
        #make predictions
        logging.info("Making predictions...")
        tqdm.pandas()
        test['prediction'] = test.progress_apply(
            lambda row: predictor(
                os.path.join(IMAGE_FOLDER, os.path.basename(row['image_link'])),
                row['group_id'],
                row['entity_name'],
                reader
            ),
            axis=1
        )
        
        #save results
        output_filename = os.path.join(DATASET_FOLDER, 'test_out.csv')
        test[['index', 'prediction']].to_csv(output_filename, index=False)
        logging.info(f"Predictions saved to {output_filename}")
        
        #print first few predictions for quick check
        logging.info("Sample predictions:")
        print(test[['index', 'prediction']].head())
        
    except Exception as e:
        logging.error(f"An error occurred in the main script: {str(e)}")

    logging.info("Script completed")