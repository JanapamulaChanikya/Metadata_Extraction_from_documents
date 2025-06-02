# Rental Agreement Information Extraction and Validation

## Overview

This project automates the extraction of key information from rental agreement documents using text extraction combined with a Groq-based information extraction API. The goal is to accurately extract fields such as agreement value, start/end dates, renewal notice days, and involved parties, then validate these extractions against ground truth data.

## Workflow

1. Data Collection 
   - Rental agreements are stored in `train` and `test` folders.  
   - Ground truth data for these agreements are provided in CSV files (`train.csv`, `test.csv`).

2. Text Extraction
   - Text is extracted from documents based on file type (`.docx`, `.pdf`, `.png`) using corresponding extraction functions.

3. Information Extraction (Groq API)  
   - Extracted raw text is sent to the Groq API, which parses and extracts structured fields:  
     - Agreement Value  
     - Agreement Start Date  
     - Agreement End Date  
     - Renewal Notice (Days)  
     - Party One  
     - Party Two  

4. Validation and Comparison 
   - Extracted data is compared with ground truth CSV data.  
   - Counts of True Positives (TP), False Positives (FP), False Negatives (FN), and True Negatives (TN) are calculated per field.

5. Metrics Calculation  
   - Precision, Recall, Accuracy, and F1 Score are computed per field and overall.  

6. Output
   - Extracted data saved to CSV files (`extracted_train_data.csv`, `extracted_test_data.csv`).  
   - Console output displays detailed metrics.


## Predictions for the files in the test set (test/ folder) 

| File Name                         | Aggrement Value | Aggrement Start Date | Aggrement End Date | Renewal Notice (Days) | Party One             | Party Two                          |
|----------------------------------|-----------------|---------------------|--------------------|----------------------|-----------------------|----------------------------------|
| 156155545-Rental-Agreement-Kns-Home | 12000           | 15.12.2012          | 15.11.2013         | 30                   | V.K.NATARAJ           | SRI VYSHNAVI DAIRY SPECIALITIES Private Ltd. |
| 228094620-Rental-Agreement       | 15000           | 07.07.2013          | 06.06.2014         |                      | Mr. KAPIL MEHROTRA    | Mr.B.Kishore                     |
| 24158401-Rental-Agreement        | 12000           | 01.04.2008          | 31.03.2009         | 60                   | Sri Hanumaiah         | Sri Vishal Bhardwaj S/O Charnel Singh |
| 95980236-Rental-Agreement        | N/A             | N/A                 | N/A                | N/A                  | N/A                   | N/A                              |




## Per field Recall metric score with including other metrics too 

| Parameter            | Precision | Recall | Accuracy | F1 Score |
|----------------------|-----------|--------|----------|----------|
| Agreement Value      | 1.00      | 1.00   | 1.00     | 1.00     |
| Agreement Start Date | 0.88      | 1.00   | 0.88     | 0.93     |
| Agreement End Date   | 0.62      | 1.00   | 0.62     | 0.77     |
| Renewal Notice (Days)| 0.71      | 0.83   | 0.62     | 0.77     |
| Party One            | 0.25      | 1.00   | 0.25     | 0.40     |
| Party Two            | 0.00      | 0.00   | 0.00     | 0.00     |

- Overall Precision:0.57  
- Overall Recall:0.96  
- Overall Accuracy:0.56  
- Overall F1 Score: 0.72  

## About the Groq API

- The Groq API is the core engine for extracting structured data from raw document text.  
- It uses domain-specific query logic to identify contractual parameters from unstructured text.  
- Designed to maximize recall and maintain reasonable precision across various document formats.
- Open groq_api portal and create a api.
- copy the created api.
- place it in the environment in which your code is running

## How to Run

1. Place your documents in the `train` and `test` folders.  
2. Ensure ground truth CSV files (`train.csv`, `test.csv`) are available in the data directory.  
3. Update file paths in the script as needed.  
4. Run the main extraction script:  
  
   python extraction_metadata.py


   
## Rental Agreement Metadata Extraction API
========================================

1. Setup Instructions
---------------------
- Make sure you have Python 3.7+ installed.
- Install required packages by running:

    pip install fastapi uvicorn python-multipart pandas

2. Running the API Server
-------------------------
- Save the FastAPI code in a file named `app.py`.
- Run the server using the command:

    uvicorn app:app --reload

- You will see output like:

    INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)

3. Accessing the API
--------------------
- Open your browser and navigate to:

    http://127.0.0.1:8000/docs

- This opens an interactive Swagger UI where you can test the `/extract/` endpoint.

4. Using the `/extract/` API Endpoint
-------------------------------------
- Method: POST
- URL: http://127.0.0.1:8000/extract/
- Form-data field: `file` (upload a `.docx` or `.png` document file)

5. Example Using curl
---------------------
To extract metadata from a document file (`sample.docx`), run:

    curl -X POST "http://127.0.0.1:8000/extract/" -F "file=@sample.docx"

- The response will be a JSON containing extracted metadata fields like:

    {
        "Agreement Start Date": "2023-01-01",
        "Agreement End Date": "2024-01-01",
        "Renewal Notice (Days)": 30,
        "Party One": "Company A",
        "Party Two": "Company B"
    }

6. Notes
--------
- Temporary files are automatically deleted after processing.
- Make sure to handle errors appropriately; the API returns status codes 400 or 500 on failure.

---

Thank you for using this API!
