import openpyxl
import os
import torch

def create_or_modify_excel(file_path, data_path, threshold, model, data):
    prased_data = parse_path_and_data(data_path, threshold, model, data)
    if os.path.exists(file_path):
        # If the file already exists, load it and append data
        wb = openpyxl.load_workbook(file_path)
        ws = wb.active
        # Check if the worksheet is empty
        if ws.max_row == 0:
            # If the worksheet is empty, add column names
            ws.append(['Sensor', 'Condition', 'Detector', 'IoU Threshold', 'Number of cases', 'Successful detections'])
        ws.append(prased_data)
    else:
        # If the file doesn't exist, create a new one and add data
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.append(['Sensor', 'Condition', 'Detector', 'IoU Threshold', 'Number of cases', 'Successful detections'])
        ws.append(prased_data)
            
    wb.save(file_path)

def parse_path_and_data(data_path, threshold, model, data):
    folders = data_path.split(os.path.sep)
    output = []
    if 'HDL-64E' in folders:
        output.append('HDL-64E')
    else:
        output.append('VLP-16')

    if 'clear' in folders:
        output.append('clear')
    elif 'light' in folders:
        output.append('light')
    elif 'moderate' in folders:
        output.append('moderate')
    elif 'heavy' in folders:
        output.append('heavy')
    
    output.append(model)
    output.append(threshold)
    output.append(len(data))
    output.append(sum(1 for x in data if x > threshold))
    return output