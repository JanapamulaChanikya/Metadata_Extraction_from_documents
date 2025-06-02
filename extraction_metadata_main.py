#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import csv
import dotenv
import easyocr
import docx
import pypdf
from pydantic import BaseModel, Field, ValidationError
from typing import Optional, Any

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate


# In[3]:


dotenv.load_dotenv()
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

llm = None
if not GROQ_API_KEY:
    print("Error: GROQ_API_KEY environment variable not set.")
else:
    llm = ChatGroq(api_key=GROQ_API_KEY, model="llama3-8b-8192", temperature=0)


# In[5]:


class RentalAgreementInfo(BaseModel):
    agreement_value: Optional[Any] = Field(None, description="The numerical value of the rent amount in integer format only, no currency symbols or commas.")
    agreement_start_date: Optional[str] = Field(None, description="The start date of the lease agreement in DD.MM.YYYY format.")
    agreement_end_date: Optional[str] = Field(None, description="The end date of the lease agreement in DD.MM.YYYY format.")
    renewal_notice_days: Optional[Any] = Field(None, description="The number of days for the renewal notice period, as a number only.")
    party_one: Optional[str] = Field(None, description="The full name of Party One (usually the Landlord). Keep the CASE as it is.")
    party_two: Optional[str] = Field(None, description="The full name of Party Two (usually the Tenant). Keep the CASE as it is.")


# In[7]:


def extract_text_from_docx(file_path):
    text = ""
    try:
        document = docx.Document(file_path)
        for paragraph in document.paragraphs:
            text += paragraph.text + "\n"
    except Exception as e:
        print(f"Error reading docx file {file_path}: {e}")
        return None
    return text


# In[9]:


def extract_text_from_png(file_path):
    try:
        reader = easyocr.Reader(['en'], model_storage_directory='.')
    except Exception as e:
        print(f"Error initializing EasyOCR reader: {e}")
        print("Please ensure EasyOCR models are downloaded or internet is available for the first run.")
        return None
    text = ""
    try:
        results = reader.readtext(file_path)
        for (bbox, text_content, prob) in results:
            text += text_content + "\n"
    except Exception as e:
        print(f"Error reading png file {file_path} with EasyOCR: {e}")
        return None
    return text


# In[11]:


def extract_text_from_pdf(file_path):
    text = ""
    try:
        with open(file_path, 'rb') as file:
            reader = pypdf.PdfReader(file)
            for page_num in range(len(reader.pages)):
                text += reader.pages[page_num].extract_text()
    except Exception as e:
        print(f"Error reading pdf file {file_path}: {e}")
        return None
    return text


# In[13]:


def extract_info_with_groq(text_content: str) -> Optional[RentalAgreementInfo]:
    global llm
    if not llm:
         print("Groq client not initialized (API key missing?). Skipping information extraction.")
         return None

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant that extracts key information from rental agreements and formats the output as a JSON object with the following keys: agreement_value, agreement_start_date, agreement_end_date, renewal_notice_days, party_one, party_two. Ensure the output is ONLY the JSON object."),
        ("human", "Extract the following information from the rental agreement text:\n"
                   "- The numerical value of the rent amount (integer only, no currency or commas).\n"
                   "- The start date of the lease agreement in DD.MM.YYYY format.\n"
                   "- The end date of the lease agreement in DD.MM.YYYY format.\n"
                   "- The number of days for the renewal notice period (number only).\n"
                   "- The full name of Party One (usually the Landlord). Maintain original casing.\n"
                   "- The full name of Party Two (usually the Tenant). Maintain original casing.\n\n"
                   "Text:\n{text_content}\n\n"
                   "Please provide the extracted information as a JSON object with the keys: agreement_value, agreement_start_date, agreement_end_date, renewal_notice_days, party_one, party_two."),
    ])

    try:
        response = llm.invoke(prompt.format_messages(text_content=text_content))
        response_text = response.content

        json_string = None

        if '```json' in response_text:
            json_start = response_text.find('```json') + 7
            json_end = response_text.find('```', json_start)
            if json_start != -1 and json_end != -1:
                json_string = response_text[json_start:json_end].strip()

        if not json_string and '<tool-use>' in response_text and '</tool-use>' in response_text:
             tool_use_start = response_text.find('<tool-use>')
             tool_use_end = response_text.find('</tool-use>', tool_use_start)
             if tool_use_start != -1 and tool_use_end != -1:
                 tool_use_content = response_text[tool_use_start + len('<tool-use>'):tool_use_end].strip()
                 parameters_start = tool_use_content.find('"parameters":')
                 if parameters_start != -1:
                     json_content_start = tool_use_content.find('{', parameters_start)
                     json_content_end = tool_use_content.rfind('}') + 1
                     if json_content_start != -1 and json_content_end != -1 and json_content_end > json_content_start:
                          json_string = tool_use_content[json_content_start:json_content_end].strip()
                     else:
                          json_start = tool_use_content.find('{')
                          json_end = tool_use_content.rfind('}') + 1
                          if json_start != -1 and json_end != -1 and json_end > json_start:
                              json_string = tool_use_content[json_start:json_end].strip()

        if not json_string:
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            if json_start != -1 and json_end != -1 and json_end > json_start:
                json_string = response_text[json_start:json_end].strip()

        if not json_string:
             print(f"Warning: Could not extract potential JSON string from response: {response_text}")
             return None

        try:
            extracted_data_model = RentalAgreementInfo.model_validate_json(json_string)
            return extracted_data_model
        except (json.JSONDecodeError, ValidationError) as e:
            print(f"Error parsing extracted string as JSON or validating with Pydantic: {e}")
            print(f"Attempted to parse: {json_string}")
            print(f"Original response: {response_text}")
            return None

    except Exception as e:
        print(f"Error during Langchain invoke or processing: {e}")
        return None


# In[15]:


def process_single_file(file_path, output_csv_path):
    if not os.path.isfile(file_path):
        print(f"Error: Input file '{file_path}' not found.")
        return False

    csv_headers = ["File Name", "Aggrement Value", "Aggrement Start Date", "Aggrement End Date", "Renewal Notice (Days)", "Party One", "Party Two"]
    write_headers = not os.path.exists(output_csv_path)

    try:
        with open(output_csv_path, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_headers)

            if write_headers:
                writer.writeheader()

            full_filename = os.path.basename(file_path)
            filename_without_extension, file_extension = os.path.splitext(full_filename)

            print(f"Processing file: {full_filename}")
            text_content = None
            lower_filename = full_filename.lower()

            if lower_filename.endswith('.docx') or lower_filename.endswith('.pdf.docx'):
                text_content = extract_text_from_docx(file_path)
            elif lower_filename.endswith('.png'):
                text_content = extract_text_from_png(file_path)
            elif lower_filename.endswith('.pdf'):
                 text_content = extract_text_from_pdf(file_path)
            else:
                print(f"Skipping unsupported file type: {full_filename}")
                return False

            if text_content:
                extracted_data_model = extract_info_with_groq(text_content)
                if extracted_data_model:
                    extracted_data = extracted_data_model.model_dump()
                    row_data = {
                        "File Name": filename_without_extension,
                        "Aggrement Value": extracted_data.get("agreement_value", "N/A"),
                        "Aggrement Start Date": extracted_data.get("agreement_start_date", "N/A"),
                        "Aggrement End Date": extracted_data.get("agreement_end_date", "N/A"),
                        "Renewal Notice (Days)": extracted_data.get("renewal_notice_days", "N/A"),
                        "Party One": extracted_data.get("party_one", "N/A"),
                        "Party Two": extracted_data.get("party_two", "N/A"),
                    }

                    writer.writerow(row_data)
                    print(f"Successfully extracted and wrote data for {full_filename}")
                    return True
                else:
                    print(f"Could not extract information from {full_filename}")
                    return False
            else:
                print(f"Could not extract text from {full_filename}")
                return False
    except Exception as e:
        print(f"Error writing to CSV file {output_csv_path}: {e}")
        return False

file_path = r"C:\Users\janap\Downloads\data\train\50070534-RENTAL-AGREEMENT.docx"
output_csv_path = "extracted_data.csv"

process_single_file(file_path, output_csv_path)


# In[17]:


import os
import csv
from collections import defaultdict
from typing import Dict, Any, List, Optional

import pandas as pd


# In[19]:


TRAIN_DIR = r"C:\Users\janap\Downloads\data\train"
TEST_DIR = r"C:\Users\janap\Downloads\data\test"
TRAIN_CSV = r"C:\Users\janap\Downloads\data\train.csv"
TEST_CSV = r"C:\Users\janap\Downloads\data\test.csv"
EXTRACTED_TRAIN_CSV = 'extracted_train_data.csv' 
EXTRACTED_TEST_CSV = 'extracted_test_data.csv' 


# In[21]:


FIELD_TO_CSV_HEADER_MAP = {
    "agreement_value": "Aggrement Value",
    "agreement_start_date": "Aggrement Start Date",
    "agreement_end_date": "Aggrement End Date",
    "renewal_notice_days": "Renewal Notice (Days)",
    "party_one": "Party One",
    "party_two": "Party Two",
}


# In[23]:


CSV_HEADER_TO_FIELD_MAP = {v: k for k, v in FIELD_TO_CSV_HEADER_MAP.items()}


# In[27]:


def read_ground_truth_csv(csv_path: str) -> Dict[str, Dict[str, str]]:
    ground_truth_data: Dict[str, Dict[str, str]] = {}
    if not os.path.exists(csv_path):
        print(f"Warning: Ground truth CSV file not found at {csv_path}")
        return ground_truth_data

    with open(csv_path, mode='r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            file_name_with_ext = row.get("File Name")
            if file_name_with_ext:
                file_name_without_ext, _ = os.path.splitext(file_name_with_ext)
                ground_truth_data[file_name_without_ext] = {
                    header: row.get(header, "N/A") for header in row.keys() if header != "File Name"
                }
            else:
                print(f"Warning: Row missing 'File Name' in {csv_path}: {row}")

    return ground_truth_data



# In[35]:


def compare_extracted_with_truth(
    extracted_model: Optional[RentalAgreementInfo],
    ground_truth: Optional[Dict[str, str]]
) -> Dict[str, bool]:
    
    comparison_results: Dict[str, bool] = {}
   
    if not extracted_model or not ground_truth:

        for field in RentalAgreementInfo.model_fields.keys():
             comparison_results[field] = False
        return comparison_results

    extracted_data = extracted_model.model_dump()

    for field, csv_header in FIELD_TO_CSV_HEADER_MAP.items():
        extracted_value = extracted_data.get(field)
        ground_truth_value = ground_truth.get(csv_header)

        if isinstance(extracted_value, int):
            extracted_value_str = str(extracted_value).strip()
        else:
            extracted_value_str = str(extracted_value).strip() if extracted_value is not None else ""


        ground_truth_value_str = str(ground_truth_value).strip() if ground_truth_value is not None else ""

        is_match = extracted_value_str == ground_truth_value_str
        comparison_results[field] = is_match

    return comparison_results



# In[54]:


def run_validation(train_dir: str, test_dir: str, train_csv: str, test_csv: str):
   
    print("Loading ground truth data...")
    train_ground_truth = read_ground_truth_csv(train_csv)
    test_ground_truth = read_ground_truth_csv(test_csv)
    print(f"Loaded {len(train_ground_truth)} entries from {train_csv}")
    print(f"Loaded {len(test_ground_truth)} entries from {test_csv}")

   
    metrics_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: {"TP": 0, "FP": 0, "FN": 0, "TN": 0})

    directories_to_process = {
        train_dir: (train_ground_truth, EXTRACTED_TRAIN_CSV),
        test_dir: (test_ground_truth, EXTRACTED_TEST_CSV),
    }

    csv_headers = ["File Name", "Aggrement Value", "Aggrement Start Date", "Aggrement End Date", "Renewal Notice (Days)", "Party One", "Party Two"]

    print("\nStarting extraction and comparison...")
    for directory, (ground_truth, output_csv_path) in directories_to_process.items():
        extracted_data_list: List[Dict[str, str]] = [] 

        if not os.path.exists(directory):
            print(f"Warning: Directory '{directory}' not found. Skipping.")
            continue

        print(f"\nProcessing files in '{directory}'...")
        for filename_with_ext in os.listdir(directory):
             file_path = os.path.join(directory, filename_with_ext)
             file_name_without_ext, file_extension = os.path.splitext(filename_with_ext)

             gt_data = ground_truth.get(file_name_without_ext)
             if gt_data:
                print(f"Processing file: {filename_with_ext}")
                text_content = None
                lower_filename = filename_with_ext.lower()

                if lower_filename.endswith('.docx') or lower_filename.endswith('.pdf.docx'):
                    text_content = extract_text_from_docx(file_path)
                elif lower_filename.endswith('.png'):
                    text_content = extract_text_from_png(file_path)
                elif lower_filename.endswith('.pdf'):
                     text_content = extract_text_from_pdf(file_path)
                else:
                    print(f"Skipping unsupported file type: {filename_with_ext}")
                    continue 

                extracted_data_model = None
                if text_content:
                   
                    extracted_data_model = extract_info_with_groq(text_content)

                extracted_data_dict = extracted_data_model.model_dump() if extracted_data_model else {}
                row_data = {
                    "File Name": file_name_without_ext,
                    "Aggrement Value": extracted_data_dict.get("agreement_value", "N/A"),
                    "Aggrement Start Date": extracted_data_dict.get("agreement_start_date", "N/A"),
                    "Aggrement End Date": extracted_data_dict.get("agreement_end_date", "N/A"),
                    "Renewal Notice (Days)": extracted_data_dict.get("renewal_notice_days", "N/A"),
                    "Party One": extracted_data_dict.get("party_one", "N/A"),
                    "Party Two": extracted_data_dict.get("party_two", "N/A"),
                }
                extracted_data_list.append(row_data)


                if extracted_data_model:
                    comparison_results = compare_extracted_with_truth(
                        extracted_data_model,
                        gt_data 
                    )

                    for field in RentalAgreementInfo.model_fields.keys():
                        is_match = comparison_results.get(field, False)
                        extracted_value = extracted_data_dict.get(field)
                        ground_truth_value = gt_data.get(FIELD_TO_CSV_HEADER_MAP.get(field))

                      
                        ground_truth_exists = ground_truth_value is not None and str(ground_truth_value).strip() != "" and str(ground_truth_value).strip().upper() != "N/A" 

                        extracted_exists = extracted_value is not None and str(extracted_value).strip() != "" and str(extracted_value).strip().upper() != "N/A"


                        if is_match and ground_truth_exists:
                            metrics_counts[field]["TP"] += 1
                        elif not is_match and extracted_exists and ground_truth_exists:
                            metrics_counts[field]["FP"] += 1 
                        elif not extracted_exists and ground_truth_exists:
                             metrics_counts[field]["FN"] += 1 
                        elif not ground_truth_exists and not extracted_exists:
                             metrics_counts[field]["TN"] += 1
                        elif extracted_exists and not ground_truth_exists:
                             metrics_counts[field]["FP"] += 1

                else:
                     print(f"Could not extract structured information from {filename_with_ext} for comparison.")
             else:
                 print(f"Skipping {filename_with_ext}: No corresponding entry found in ground truth CSV.")

        
        if extracted_data_list:
             extracted_df = pd.DataFrame(extracted_data_list, columns=csv_headers)
             extracted_df.to_csv(output_csv_path, index=False, encoding='utf-8')
             print(f"\nExtracted data saved to {output_csv_path}")


  

    print("\n--- Validation Results ---")

    overall_tp = 0
    overall_fp = 0
    overall_fn = 0
    overall_tn = 0
    overall_total = 0


    for field, counts in metrics_counts.items():
        tp = counts["TP"]
        fp = counts["FP"]
        fn = counts["FN"]
        tn = counts["TN"]

        overall_tp += tp
        overall_fp += fp
        overall_fn += fn
        overall_tn += tn

       
        total_for_field = tp + fp + fn + tn
        overall_total += total_for_field 

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        accuracy = (tp + tn) / total_for_field if total_for_field > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        print(f"\nParameter: {field} (CSV Header: {FIELD_TO_CSV_HEADER_MAP.get(field, 'N/A')})")
        print(f"  Counts: TP={tp}, FP={fp}, FN={fn}, TN={tn}, Total={total_for_field}")
        print(f"  Precision: {precision:.2f}")
        print(f"  Recall: {recall:.2f}")
        print(f"  Accuracy: {accuracy:.2f}")
        print(f"  F1 Score: {f1:.2f}")

    overall_precision = overall_tp / (overall_tp + overall_fp) if (overall_tp + overall_fp) > 0 else 0
    overall_recall = overall_tp / (overall_tp + overall_fn) if (overall_tp + overall_fn) > 0 else 0
    overall_accuracy = (overall_tp + overall_tn) / overall_total if overall_total > 0 else 0
    overall_f1 = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0

    print("\n--- Overall Metrics ---")
    print(f"Total parameters evaluated across all files and fields: {overall_total}")
    print(f"Overall Counts: TP={overall_tp}, FP={overall_fp}, FN={overall_fn}, TN={overall_tn}")
    print(f"Overall Precision: {overall_precision:.2f}")
    print(f"Overall Recall: {overall_recall:.2f}")
    print(f"Overall Accuracy: {overall_accuracy:.2f}")
    print(f"Overall F1 Score: {overall_f1:.2f}")



if __name__ == "__main__":
    run_validation(TRAIN_DIR, TEST_DIR, TRAIN_CSV, TEST_CSV)

