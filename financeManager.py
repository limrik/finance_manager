import csv
import gspread
from decimal import Decimal, InvalidOperation
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import time
import os

def safe_decimal(value):
    """Convert value to Decimal safely, returning 0 if conversion fails."""
    try:
        return Decimal(value)
    except (InvalidOperation, ValueError):
        return Decimal(0)
    
# Load the model and tokenizer
model_name = "mgrella/autonlp-bank-transaction-classification-5521155"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

def classify_transaction(description):
    """Classify the transaction description into a category."""
    inputs = tokenizer(description, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_id = torch.argmax(logits, dim=1).item()
    predicted_classes = model.config.id2label[predicted_class_id][len("Category."):] # remove Category. prefix
    predicted_classes_array = predicted_classes.split('_')
    return predicted_classes_array[0]

def dbsFin(file):

    sum_in = 0
    sum_out = 0
    transactions = []

    with open(file, mode='r') as csv_file:
        csv_reader = csv.reader(csv_file)

        # get rid of eveyrthing before header rows
        for row in csv_reader:
            if row == header_row:
                for _ in range(2):
                    next(csv_reader)
                break

        for row in csv_reader:
            if row:
                date = row[0]
                name = row[4]
                amt_out = safe_decimal(row[2])
                amt_in = safe_decimal(row[3])
                category = classify_transaction(name)
                transaction = ((date, name, amt_out, amt_in, category))
                transactions.append(transaction)

                sum_in += amt_in
                sum_out += amt_out
        return transactions, sum_in, sum_out

header_row = ['Transaction Date', 'Reference', 'Debit Amount', 'Credit Amount', 'Transaction Ref1', 'Transaction Ref2', 'Transaction Ref3']
start_index = -1

path = "./transaction_statements"

for filename in os.listdir(path):
    file=filename
    clean_name = os.path.splitext(filename)[0]

    sa = gspread.service_account()
    sh = sa.open("Personal Finances")

    try:
        wks = sh.worksheet(clean_name)
    except gspread.exceptions.WorksheetNotFound:
        wks = sh.worksheet('template').duplicate(new_sheet_name=clean_name)

    rows = dbsFin(f'./transaction_statements/{file}')[0]

    for row in rows:
        wks.insert_row([row[0],row[1],row[4], float(row[3]), float(row[2])], 8)
        time.sleep(2)

