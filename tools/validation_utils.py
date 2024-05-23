import openpyxl
import os
import torch

def create_or_modify_excel(file_path, data_path, threshold, model, data):
    """
    Create or modify an Excel file to append detection results.
    
    Args:
        file_path (str): Path to the Excel file.
        data_path (str): Path to the data directory.
        threshold (float): IoU threshold for detection.
        model (str): Model name.
        data (list): List of detection results.
    """
    parsed_data = parse_path_and_data(data_path, threshold, model, data)
    
    if os.path.exists(file_path):
        # If the file already exists, load it and append data
        wb = openpyxl.load_workbook(file_path)
        ws = wb.active
        # Check if the worksheet is empty and add column names if necessary
        if ws.max_row == 0:
            ws.append(['Sensor', 'Condition', 'Detector', 'IoU Threshold', 'Number of cases', 'Successful detections'])
        ws.append(parsed_data)
    else:
        # If the file doesn't exist, create a new one and add data
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.append(['Sensor', 'Condition', 'Detector', 'IoU Threshold', 'Number of cases', 'Successful detections'])
        ws.append(parsed_data)
    
    wb.save(file_path)

def create_or_modify_excel_recall(file_path, data_path, threshold, model, data, budget):
    """
    Create or modify an Excel file to append recall results.
    
    Args:
        file_path (str): Path to the Excel file.
        data_path (str): Path to the data directory.
        threshold (float): IoU threshold for detection.
        model (str): Model name.
        data (list): List of detection results.
        budget (int): Budget for the recall calculation.
    """
    parsed_data = parse_path_and_data_recall(data_path, threshold, model, data, budget)
    
    if os.path.exists(file_path):
        # If the file already exists, load it and append data
        wb = openpyxl.load_workbook(file_path)
        ws = wb.active
        # Check if the worksheet is empty and add column names if necessary
        if ws.max_row == 0:
            ws.append(['Sensor', 'Condition', 'Detector', 'IoU Threshold', 'Budget', 'Recall'])
        ws.append(parsed_data)
    else:
        # If the file doesn't exist, create a new one and add data
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.append(['Sensor', 'Condition', 'Detector', 'IoU Threshold', 'Budget', 'Recall'])
        ws.append(parsed_data)
    
    wb.save(file_path)

def parse_path_and_data(data_path, threshold, model, data):
    """
    Parse the data path and detection results for appending to Excel.
    
    Args:
        data_path (str): Path to the data directory.
        threshold (float): IoU threshold for detection.
        model (str): Model name.
        data (list): List of detection results.
    
    Returns:
        list: Parsed data.
    """
    folders = data_path.split(os.path.sep)
    output = []
    
    # Determine the sensor type
    if 'HDL-64E' in folders:
        output.append('HDL-64E')
    else:
        output.append('VLP-16')
    
    # Determine the condition type
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

def parse_path_and_data_recall(data_path, threshold, model, data, budget):
    """
    Parse the data path and recall results for appending to Excel.
    
    Args:
        data_path (str): Path to the data directory.
        threshold (float): IoU threshold for detection.
        model (str): Model name.
        data (list): List of detection results.
        budget (int): Budget for the recall calculation.
    
    Returns:
        list: Parsed data.
    """
    folders = data_path.split(os.path.sep)
    output = []
    
    # Determine the sensor type
    if 'HDL-64E' in folders:
        output.append('HDL-64E')
    else:
        output.append('VLP-16')
    
    # Determine the condition type
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
    output.append(budget)
    output.append(sum(1 for x in data if x > threshold) / len(data))
    
    return output
