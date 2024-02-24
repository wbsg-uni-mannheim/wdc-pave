# get raw data
import json
from tqdm import tqdm
import pandas as pd
import re
import string
from collections import defaultdict

jsonl_path = f'../data/raw/wdc/final_target_scores.jsonl'
data_list = []
with open(jsonl_path, "r", encoding="utf-8") as file:
    for line in file:
        #print(line)
        json_obj = json.loads(line)
        data_list.append(json_obj)
wdc_data = pd.DataFrame(data_list)

def categorize_measurement(value):
    if value.replace(' ', '').isalpha():
        return 'Free Text'
    
    has_unit = any(unit in value for unit in ['"', 'cm', 'in', "'", 'ft', "inch", "inches", "In", "Inches", "mm", "m", "yds", "yards", "Yards"])

    if '/' in value:
        category = 'Numeric with Fraction'
    elif '.' in value:
        category = 'Numeric with Decimal'
    else:
        category = 'Numeric'

    return category + (' with Unit' if has_unit else '')

def categorize_capacity(value):
    numeric_pattern = r"\d+\.?\d*"  # Pattern to match numeric values
    unit_pattern = r"qt\.?|Gallon|gallon|gal\.?|lbs\.?|lb\.?|oz\.?|cu\. ft\.?|litres?|liter|L|ml|GB|barrel|cubic feet|cu yd|cu\. ft\.|ltr"
    
    has_numeric = bool(re.search(numeric_pattern, value))
    has_unit = bool(re.search(unit_pattern, value))

    # Check for non-unit and non-numeric alphabetic strings
    non_unit_alpha = re.sub(numeric_pattern, '', value)  # Remove numeric parts
    non_unit_alpha = re.sub(unit_pattern, '', non_unit_alpha)  # Remove unit parts
    has_other_alpha = any(char.isalpha() for char in non_unit_alpha)

    # Determine the category
    if has_numeric and has_unit:
        return 'Numeric with Unit' if not has_other_alpha else 'Free Text and Numeric with Unit'
    elif has_numeric and has_other_alpha:
        return 'Free Text with Numeric'
    elif has_numeric:
        return 'Numeric'
    elif has_other_alpha:
        return 'Free Text'
    else:
        return 'Unknown'

def categorize_weight(value):
    if value.replace(' ', '').isalpha():
        return 'Free Text'
    
    has_unit = any(unit in value for unit in ['"', "oz", "Pound", "lbs", "lb", "in", "Ounce"])

    if '/' in value:
        category = 'Numeric with Fraction'
    elif '.' in value:
        # find all decimal  values
        all_decimal_values = re.findall("\d+\.\d+", value)
        if len(all_decimal_values) > 0:
            category = 'Numeric with Decimal'
        else:
            category = "Numeric"
    else:
        category = 'Numeric'

    return category + (' with Unit' if has_unit else '')

def categorize_manufacturer(value):
    if value in ["Hewlett-Packard",
                "Dell",
                "Compaq Proliant",
                "Seagate Enterprise",
                "Cooler Master",
                "Compaq",
                "Hewlett-Packard Proliant",
                "Fujitsu",
                "Hitachi",
                "COMPAQ",
                "Kingston",
                "Seagate",
                "Noctua",
                "Artesyn",
                "Quantum",
                "Sun",
                "ProLiant",
                "PROLIANT",
                "Proliant",
                "Crucial"]:
        return "Name"
    elif value in ["HPE",
                    "IBM",
                    "HP Pro",
                    "AMD",
                    "EMC",
                    "HP"]:
        return "Abbreviated Name"
    elif value in ["HP Proliant",
                   "HP ProLiant",
                    "HP Compaq",
                    "HP COMPAQ"]:
        return "Abbreviated And Full Name"
    else:
        print(value)

def detect_id_type(value):
    # Check if the string is empty or not provided
    if not value:
        return "Invalid ID"

    has_digit = False
    has_alpha = False
    has_special = False

    # Iterate over each character in the string
    for char in value:
        if char.isdigit():
            has_digit = True
        elif char.isalpha():
            has_alpha = True
        elif not char.isspace():  # Check for any character that is not a space and not alphanumeric
            has_special = True

    # Determine the type of ID based on the flags set
    if has_digit and not has_alpha and not has_special:
        return "Unique Numeric ID"
    elif has_digit and has_alpha and not has_special:
        return "Unique Alphanumeric ID"
    elif has_digit and not has_alpha and has_special:
        return "Unique Numeric ID with marks"
    elif (has_digit or has_alpha) and has_special:
        return "Unique Alphanumeric ID with marks"
    elif value == "PGHJG":
        return "Unique Alphanumeric ID"
    else:
        print(value)
        return "Invalid ID Format"
    
def determine_interface_data_type(value):
    if value in [
        "C-GbE2", "C -GBE2", "SAT-150 / SAS", 
        "U320 SCSI", "Ultra2/Ultra3 SCSI", "WU3", "Socket AM3+", 
        "U160 SCSI", "DIMM 240-pin", "Ultra320 LVD SCSI", "Male DB-9Connector to RJ-45", "Ultra320 SCSI",  "Ultra3 SCSI", "Ultra2 SCSI", "Ultra3", 
        "RJ-45", "PS/2 and VG", "Ultra SCSI-3", "SCSI Ultra 160"]:
        return "Abbreviated Name with Numeric"

    elif value in [
        "Serial Attached SCSI", "Fibre Channel", "Fiber Channel"
    ]:
        return "Name"
    
    elif value in ["AT IDE", "IDE", "LVD SE", "PCIe", "FC-AL", "PCI-X SAS", "PCI-X", "FCAL",  "SCSI LVD",  "Serial AT(SAT)", "Ultra SCSI", "HVD", "FAT", "SAT", "(LVD/SE) SCSI", "Ultra SC",
        "SAT/SAS", "FC", "SAS", "PCI", "LVD", "SCSI", "LVD/SE", "ATA", "SAS/SATA", "C-GBE",  "SATA", "C-GbE", "PCI- X", "SCSI/LVD", "Serial AT (SAT)", "PCI Express", "RJ-21 to RJ-45", "Ultra2"
    ]:
        return "Abbreviated Name"
    elif value in ["Serial Attached SCSI (SAS)", "PCI to Fibre Channel", "Fibre Channel AT (FAT)", "Serial Attached SCSI(SAS)", "Serial AttachedSCSI (SAS)"]:
        return "Abbreviated And Full Name"
    else:
        print(value)

def detect_splash_data_type(value):
    # Check if the string has numeric values
    if re.search(r'\d', value):
        # Check if it also contains a unit like inches
        if '"' in value or re.search(r'\d+\'\d+"', value) or 'inch' in value.lower():
            return "Free Text with Numeric and Unit"
        else:
            return "Free Text with Numeric"
    else:
        return "Free Text"
    
def determine_generation_data_type(value):
    if value in ["G2/G3", "DDR4", "Gen8 Gen9", "DDR", 
                 "G2HS", "G7", "Gen1-Gen7", "G4P", 
                 "AIT-3", "G4", "G3", "G8 G9", "G5p", 
                 "G2", "G1-G7", "G5", "G3/G4", "LTO2", 
                 "LTO 2", "G4p", "DLT1", "G6", "PII", 
                 "AIT-1", "DDS4", "AIT2", "G1", "G3 G4", 
                 "G4/G5", "DDR2", "DLT 1", "G1/G2", "G2 G3",
                 "Piledriver FX-4", "G8"]:
        return "Abbreviated Name with Numeric"
    else:
        return "Name with Numeric"
    


def determine_data_type(category, attribute, value):
    if category == 'Computers And Accessories':
        if attribute == 'Manufacturer':
            return categorize_manufacturer(value)
        elif attribute == 'Generation':
            return determine_generation_data_type(value)
        elif attribute == 'Part Number':
            return detect_id_type(value)
        elif attribute == "Capacity":
            return "Numeric with Unit"
        elif attribute == "Product Type":
            return "Free Text"
        elif attribute == "Interface":
            return determine_interface_data_type(value)
        elif attribute == "Cache":
            has_digit = any(char.isdigit() for char in value)
            has_alpha = any(char.isalpha() for char in value)  
            if has_digit and has_alpha:
                return 'Numeric with Unit'  
            elif has_digit:
                return "Numeric"
        elif attribute == "Processor Core":
            has_digit = any(char.isdigit() for char in value)
            has_alpha = any(char.isalpha() for char in value)    
            if has_digit and has_alpha:
                return 'Free Text with Numeric'  # Contains both letters and numbers
            elif has_alpha:
                return 'Free Text with Number as Word'
        elif attribute == "Processor Type":
            has_digit = any(char.isdigit() for char in value)
            has_alpha = any(char.isalpha() for char in value)    
            if has_digit and has_alpha:
                return 'Name With Numeric'  # Contains both letters and numbers
            elif has_alpha:
                return 'Name'
        elif attribute == "Processor Quantity":
            return "Numeric"
        elif attribute == "Bus Speed":
            return "Numeric with Unit"
        elif attribute == "Clock Speed":
            has_digit = any(char.isdigit() for char in value)
            has_alpha = any(char.isalpha() for char in value)  
            if has_digit and has_alpha:
                return 'Numeric with Unit'  
            elif has_digit:
                return "Numeric"
        elif attribute == "Rotational Speed":
            if 'rpm' in value.lower().replace(' ', ''):
                return 'Numeric with Unit'  
            elif 'k' in value.lower():
                return "Numeric (abbrevaited)" # Numeric + k 
        elif attribute == "Ports":
            has_digit = any(char.isdigit() for char in value)
            has_alpha = any(char.isalpha() for char in value)    
            if has_digit and has_alpha:
                return 'Free Text with Numeric'  # Contains both letters and numbers
            elif has_alpha:
                return 'Free Text with Number as Word'
        elif attribute == "Thermal Design Power":
            return "Numeric with Unit"
            
    if category == "Home And Garden":
        if attribute == 'Manufacturer':
            if value in ["3M", "HP"]:
                return 'Abbreviated Name'
            elif value in ["GE Monogram", "SC Johnson"]:
                return "Abbreviated And Full Name"
            else:
                return "Name"
        elif attribute == "Product Type":
            return "Free Text"
        elif attribute in ["Width", "Depth", "Height", "Length"]:
            return categorize_measurement(value)
        elif attribute == "Gauge":
            has_digit = any(char.isdigit() for char in value)
            has_alpha = any(char.isalpha() for char in value)  
            if has_digit and has_alpha:
                return 'Numeric with Unit'  
            elif has_digit:
                return "Numeric"
        elif attribute == "Material":
            has_digit = any(char.isdigit() for char in value)
            has_alpha = any(char.isalpha() for char in value)    
            if has_digit and has_alpha:
                return 'Free Text with Numeric'  # Contains both letters and numbers
            elif has_alpha:
                return 'Free Text'
        elif attribute == "Stainless Steel Series":
            return "Numeric"
        elif attribute == "Cooling":
            has_digit = any(char.isdigit() for char in value)
            has_alpha = any(char.isalpha() for char in value)    
            if has_digit and has_alpha:
                return 'Free Text with Numeric'  # Contains both letters and numbers
            elif has_alpha:
                return 'Free Text'
        elif attribute == "Splash":
            return detect_splash_data_type(value)
        elif attribute == "Shape":
            return 'Free Text'
        elif attribute == "Color":
            return 'Free Text'
        elif attribute == "Retail UPC":
            return detect_id_type(value)
        elif attribute == "Manufacturer Stock Number":
            return detect_id_type(value)
        elif attribute == "Heat":
            return 'Free Text'
        elif attribute == "Shelves":
            has_digit = any(char.isdigit() for char in value)
            has_alpha = any(char.isalpha() for char in value)    
            if has_digit:
                return 'Numeric'  # Contains both letters and numbers
            elif has_alpha:
                return 'Number as Word'
        elif attribute == "Base":
            return "Free Text"
        elif attribute == "Voltage":
            has_digit = any(char.isdigit() for char in value)
            has_alpha = any(char.isalpha() for char in value)  
            if has_digit and has_alpha:
                return 'Numeric with Unit'  
            elif has_digit:
                return "Numeric"
        elif attribute == "Capacity":
            return categorize_capacity(value)
        
    if category == "Office Products":
        if attribute == "Product Type":
            return "Free Text"
        elif attribute == "Brand":
            if value in ["3M", "HP", "ACCO"]:
                return 'Abbreviated Name'
            else:
                return "Name"
        elif attribute == "Color(s)":
            return "Free Text"
        elif attribute == "Retail UPC":
            return detect_id_type(value)
        elif attribute == "Manufacturer Stock Number":
            return detect_id_type(value)
        elif attribute == "Pack Quantity":
            has_digit = any(char.isdigit() for char in value)
            has_alpha = any(char.isalpha() for char in value)    
            if has_digit and has_alpha:
                return 'Free Text with Numeric'  # Contains both letters and numbers
            elif has_alpha:
                return 'Free Text'
            elif has_digit:
                return "Numeric"
        elif attribute == "Material":
            has_digit = any(char.isdigit() for char in value)
            has_alpha = any(char.isalpha() for char in value)    
            if has_digit and has_alpha:
                return 'Free Text with Numeric'  # Contains both letters and numbers
            elif has_alpha:
                return 'Free Text'
        elif attribute in ["Width", "Depth", "Height", "Length"]:
            return categorize_measurement(value)
        elif attribute == "Mounting":
            return 'Free Text'
        elif attribute == "Binding":
            return 'Free Text'
        elif attribute == "Closure":
            return 'Free Text'
        elif attribute == "Paper Weight":
            return "Numeric with Unit"
        elif attribute == "Sheet Perforation":
            return 'Free Text'
        elif attribute == "Capacity":
            return categorize_capacity(value)
        elif attribute == "Page Yield":
            if 'k' in value.lower():
                return "Numeric (abbrevaited)" # Numeric + k 
            else:
                return "Numeric"
        
    elif category == "Grocery And Gourmet Food":
        if attribute == "Brand":
            return "Name"
        elif attribute == "Product Type":
            return "Free Text"
        elif attribute == "Packing Type":
            return "Free Text"
        elif attribute == "Flavor":
            return "Free Text"
        elif attribute == "Size/Weight":
            return categorize_weight(value)
        elif attribute == "Pack Quantity":
            has_digit = any(char.isdigit() for char in value)
            has_alpha = any(char.isalpha() for char in value)    
            if has_digit and has_alpha:
                return 'Free Text with Numeric'  # Contains both letters and numbers
            elif has_alpha:
                return 'Free Text'
            elif has_digit:
                return "Numeric"
        elif attribute == "Manufacturer Stock Number":
            return detect_id_type(value)
        elif attribute == "Retail UPC":
            return detect_id_type(value)
    elif category == "Jewelry":
        if attribute == "Product Type":
            return "Free Text"
        elif attribute == "Gender":
            return "Free Text"
        elif attribute == "Stones Type":
            return "Free Text"
        elif attribute == "Stone Shape":
            return "Free Text"
        elif attribute == "Stones Setting":
            return "Free Text"
        elif attribute == "Metal Type":
            has_digit = any(char.isdigit() for char in value)
            has_alpha = any(char.isalpha() for char in value)    
            if has_digit and has_alpha:
                return 'Free Text with Numeric'  # Contains both letters and numbers
            elif has_alpha:
                return 'Free Text'
        elif attribute == "Stones Total Weight":
            return "Numeric with Unit"
        elif attribute == "Brand":
            return "Name"
        elif attribute == "Model Number":
            return detect_id_type(value)
    
    return "Unknown"

for product in data_list:
    category = product['category']
    for attribute, values in product['target_scores'].items():
        if values != {"n/a": 1}:
            for value_key, value_details in values.items():
                if 'pid' in value_details: 
                    value_details['datatype'] = determine_data_type(category, attribute, value_key)


aggregate_table = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
total_counts = defaultdict(int)
unique_values_per_attribute = defaultdict(lambda: defaultdict(set))

for data in data_list:
    category = data['category']
    attributes = data['target_scores']
    for attribute, values in attributes.items():
        if values != {"n/a": 1}:
            for value_key, value_data in values.items():
                if 'pid' in value_data: 
                    datatype = value_data.get('datatype', 'n/a')
                    aggregate_table[category][attribute][datatype] += 1
                    total_counts[(category, attribute)] += 1
                    unique_values_per_attribute[category][attribute].add(value_key)
        else:
            aggregate_table[category][attribute]["n/a"] += 1
            total_counts[(category, attribute)] += 1
            unique_values_per_attribute[category][attribute].add("n/a")

# Printing aggregated data with unique value counts
for category, attributes in aggregate_table.items():
    print(f'Category: {category}')
    for attribute, datatypes in attributes.items():
        print(f'  Attribute: {attribute}')
        unique_values_count = len(unique_values_per_attribute[category][attribute])
        print(f'    Unique Values: {unique_values_count}')
        for datatype, count in datatypes.items():
            relative_frequency = count / total_counts[(category, attribute)]
            print(f'    {datatype}: {count} (Relative Frequency: {relative_frequency:.2%})')

# Preparing DataFrame data
category_list = []
attribute_list = []
datatype_list = []
count_list = []
relative_frequency_list = []
unique_values_count_list = []

for category, attributes in aggregate_table.items():
    for attribute, datatypes in attributes.items():
        sorted_datatypes = sorted(datatypes.items(), key=lambda x: (x[0] == "n/a", x[0]))
        unique_values_count = len(unique_values_per_attribute[category][attribute])
        for datatype, count in sorted_datatypes:
            category_list.append(category)
            attribute_list.append(attribute)
            datatype_list.append(datatype)
            count_list.append(count)
            relative_frequency = count / total_counts[(category, attribute)]
            relative_frequency_list.append(relative_frequency)
            unique_values_count_list.append(unique_values_count)

data = {
    "Category": category_list,
    "Attribute": attribute_list,
    "Datatype": datatype_list,
    "Count": count_list,
    "Relative Frequency": relative_frequency_list,
    "Unique Values": unique_values_count_list,
}

df = pd.DataFrame(data)

grouped_df = df.groupby("Category").sum("Unique Values")

print(grouped_df)
print(df['Unique Values'].sum())

df["Relative Frequency"] = round(df["Relative Frequency"], 2)

df.to_csv("../data/stats/wdc/wdc_data_types_statistics.csv")