import gzip
import json
import os
from typing import Optional
import pandas as pd 
from pieutils.pydantic_models import ErrorClasses, ErrorClass
from pieutils.preprocessing import update_task_dict_from_test_set
import re
from typing import Dict, Any

from pydantic import create_model, Field

def prepare_example_task_prefixes(example, task_prefix):
    example['task_prefix'] = task_prefix.replace('[PLACEHOLDER]', example['attribute'])
    return example

def combine_example(example, pred, post_pred):
    """Format examples to save the predictions"""
    example['pred'] = pred
    example['post_pred'] = post_pred
    return example

def create_pydanctic_models_from_known_attributes(known_attributes): # description from local?
    """Create Pydantic models for the known attributes."""
    pydantic_models = {}
    for category in known_attributes:
        # Define field specs:
        if category[-1] == 's':
            # If the category name ends with an 's', it is a plural category, e.g. 'customers'.
            fields_spec = {attribute: (f'The {attribute} of {category}.', Optional[str]) for
                           attribute in known_attributes[category]}
        else:
            fields_spec = {attribute: (f'The {attribute} of a {category}.', Optional[str]) for
                           attribute in known_attributes[category]}
        model_description = f'Relevant customer information about a {category}.'
        pydantic_models[category] = create_pydanctic_model(category, _model_description=model_description, **fields_spec)

    return pydantic_models


def create_pydanctic_model(_model_name, _model_description='', **fields):
    # Create a dictionary to hold the fields for the dynamic model
    model_fields = {}

    # Convert the field specifications to Pydantic field declarations
    for field_name, (field_description, field_type) in fields.items():
        model_fields[field_name] = (field_type, Field(description=field_description))

    # Use the create_model function to create a dynamic Pydantic model
    dynamic_model_class = create_model(_model_name, **model_fields)
    dynamic_model_class.__doc__ = _model_description

    return dynamic_model_class

def create_pydanctic_model_with_examples(_model_name, _model_description='', **fields):
    # Create a dictionary to hold the fields for the dynamic model
    model_fields = {}

    # Convert the field specifications to Pydantic field declarations
    for field_name, (field_description, field_type, field_examples) in fields.items():
        model_fields[field_name] = (field_type, Field(description=field_description, examples=field_examples))

    # Use the create_model function to create a dynamic Pydantic model
    dynamic_model_class = create_model(_model_name, **model_fields)
    dynamic_model_class.__doc__= _model_description

    return dynamic_model_class


def create_pydanctic_model_from_pydantic_meta_model(pydantic_meta_model, known_attribute_values=None):
    # Create a dictionary to hold the fields for the dynamic model
    meta_model = dict(pydantic_meta_model)
    model_name = meta_model['name']
    model_description = meta_model['description']
    model_fields = {}
    with_examples = False
    for field in meta_model['attributes']:
        attribute = dict(field)
        if 'examples' in attribute:
            if 'type' in attribute and attribute['type'] == 'integer':
                model_fields[attribute['name']] = (attribute['description'], Optional[int], attribute['examples'])
            else:
                if known_attribute_values is not None and attribute['name'] in known_attribute_values:
                    model_fields[attribute['name']] = (attribute['description'], Optional[str], known_attribute_values[attribute['name']])
                else:
                    model_fields[attribute['name']] = (attribute['description'], Optional[str], attribute['examples'])
            with_examples = True
        else:
            if 'type' in attribute and attribute['type'] == 'integer':
                model_fields[attribute['name']] = (attribute['description'], Optional[int])
            else:
                model_fields[attribute['name']] = (attribute['description'], Optional[str])

    # Convert the model specifications to Pydantic model
    if with_examples:
        dynamic_model_class = create_pydanctic_model_with_examples(model_name, _model_description=model_description, **model_fields)
    else:
        dynamic_model_class = create_pydanctic_model(model_name, _model_description=model_description, **model_fields)
    return dynamic_model_class


def create_tabular_pydanctic_model_from_pydantic_meta_model(pydantic_meta_model):
    """Create a tabular Pydantic model from a Pydantic meta model."""
    dynamic_model_class = create_pydanctic_model_from_pydantic_meta_model(pydantic_meta_model)

    # Make this a tabular model
    model_name = f'{dynamic_model_class.__name__}s'
    model_description = f'A tabular representation of {dynamic_model_class.__name__}s.'
    model_fields = {'records': (f'A list of {model_name}.', list[dynamic_model_class])}
    tabular_dynamic_model_class = create_pydanctic_model(model_name, _model_description=model_description, **model_fields)

    return tabular_dynamic_model_class



def create_dict_of_pydanctic_product(pydantic_product):
    # Create a dictionary to hold the fields for the dynamic model
    schema = pydantic_product.schema()
    pydantic_model_schema = {'name': schema['title'], 'description': schema['description'], 'attributes': []}
    for field in schema['properties']:
        if 'examples' in schema['properties'][field] and schema['properties'][field]['examples'] is not None and len(schema['properties'][field]['examples']) > 0:
            pydantic_model_schema['attributes'].append({'name': field, 'description': schema['properties'][field]['description'], 'examples': schema['properties'][field]['examples']})
        else:
            pydantic_model_schema['attributes'].append({'name': field, 'description': schema['properties'][field]['description']})

    return pydantic_model_schema


def extract_attribute(answer, attribute):
    """Extract an attribute value for the open extraction."""
    if '\n' in answer:
        for part in answer.split('\n'):
            if attribute in part:
                if ':' in part:
                    return part.split(':')[1].strip()
    return "n/a"


def save_populated_task(task, task_dict):
    """Save the populated task to a file."""
    model_name = task_dict['model'].replace(':', '_').replace('/', '_')
    result_file = 'task_run_chat{}_{}_{}_{}.gz'.format(task, task_dict['dataset_name'], model_name, task_dict['timestamp'])
    path_to_result_file = 'prompts/runs/{}'.format(result_file)

    # Check if the path to the result file exists
    if not os.path.exists('prompts/runs'):
        os.makedirs('prompts/runs')

    with gzip.open(path_to_result_file, 'wt', encoding='utf-8') as fp:
        # set ensure_ascii=False to keep special chars
        json.dump(task_dict, fp, indent=4, ensure_ascii=False)

def save_populated_task_to_json(task, task_dict, title, description):
    """Save the populated task to a JSON file."""
    model_name = task_dict['model'].replace(':', '_').replace('/', '_')
    
    result_file = '{}_{}_{}_{}.json'.format(task, task_dict['dataset_name'], model_name, task_dict['timestamp'])
    path_to_result_file = 'prompts/runs/{}'.format(result_file)

    # Check if the path to the result file exists
    if not os.path.exists('prompts/runs'):
        os.makedirs('prompts/runs')

    with open(path_to_result_file, 'w', encoding='utf-8') as fp:
        # set ensure_ascii=False to keep special chars
        json.dump(task_dict, fp, indent=4, ensure_ascii=False)

def update_handlabelled_testset(source_path, target_path):
    loaded_dicts = []

    with open(source_path, 'r') as f:
        joint_lines = ''.join([line for line in f])
        json_dicts = joint_lines.split('}{')
        for json_dict in json_dicts:
            if json_dict[0] != '{':
                json_dict = '{' + json_dict
            if json_dict[-1] != '}':
                json_dict = json_dict + '}'
            loaded_dict = json.loads(json_dict)
            loaded_dicts.append(loaded_dict)

    with open(target_path, 'w+', encoding='utf-8') as f:
        for record in loaded_dicts:
            f.write('{}\n'.format(json.dumps(record)))


def convert_to_json_schema(pydantic_model, replace_description=False, schema_type='json_schema'):
    """Convert a Pydantic model to a JSON schema.
        schema_type: Schema to use - json_schema, json_schema_no_type or compact
        """
    schema = pydantic_model.schema()
    if schema_type in ("json_schema", "json_schema_no_type"):
        parameters = {'type': 'object', 'properties': {}}

        for property_name, property_schema in schema['properties'].items():
            if property_name in ("title"):
                continue
            if schema_type == "json_schema_no_type":
                parameters['properties'][property_name] = {k: v for k, v in property_schema.items() if k not in ("title", "type")}
            else:
                parameters['properties'][property_name] = {k: v for k, v in property_schema.items() if k not in ("title")}

        # Add required fields
        if 'required' in schema:
            parameters['required'] = schema['required']

        # Add additional definitions
        if 'definitions' in schema:
            parameters['definitions'] = {}
            for definition_name, definition_schema in schema['definitions'].items():
                parameters['definitions'][definition_name] = {'description': definition_schema['description'],
                                                              'type': 'object', 'properties': {}}

                for property_name, property_schema in definition_schema['properties'].items():
                    if property_name in ("title"):
                        continue
                    if schema_type == "json_schema_no_type":
                        parameters['definitions'][definition_name]['properties'][property_name] = {k: v for k, v in property_schema.items()
                                                                                                    if k not in ("title", "type")}
                    else:
                        parameters['definitions'][definition_name]['properties'][property_name] = {k: v for k, v in property_schema.items()
                                                                                if k not in ("title")}
                # Add required fields
                if 'required' in definition_schema:
                    parameters['definitions'][definition_name]['required'] = definition_schema['required']

        # Replace description
        if replace_description:
            schema[
                "description"] = f"Correctly extracted `{pydantic_model.__name__}` with all the required parameters with correct types."

        convert_schema = {
            "name": schema["title"],
            "description": schema["description"],
            "parameters": parameters,
        }
    elif schema_type == "compact":
        """Compact version of the JSON schema"""
        convert_schema = { }
        for property_name, property_schema in schema['properties'].items():
            if property_name in ("title"):
                continue
            if 'description' in property_schema:
                if 'examples' in property_schema:
                    convert_schema[property_name] = f"{property_schema['description']} - Examples: {', '.join(property_schema['examples'])}"
                else:
                    convert_schema[property_name] = property_schema['description']

    elif schema_type == "textual":
        """Textual version of the JSON schema"""
        introduction = f"A product offer from the product category {schema['title']} has the following attributes: {', '.join(schema['properties'].keys())}."
        attributes = []
        for property_name, property_schema in schema['properties'].items():
            if property_name in ("title"):
                continue
            if 'description' in property_schema:
                if 'examples' in property_schema:
                    attribute_text = f"The attribute {property_name} is defined as: {property_schema['description']} Known attribute values are {', '.join(property_schema['examples'])}."

                else:
                    attribute_text = f"The attribute {property_name} is defined as: {property_schema['description']}"
                attributes.append(attribute_text)
        convert_schema = introduction + '\n ' + '\n'.join(attributes)
    else:
        raise ValueError(f"Unknown schema type: {schema_type}")

    return convert_schema


def get_normalization_guidelines_from_csv(normalization_params, category, normalized_only):
    directory_path = f"data/descriptions/wdc/descriptions.csv"
    descriptions_csv = pd.read_csv(directory_path, sep=";")
    descriptions_csv = descriptions_csv[["Category", "Attribute", "Normalization_params", "Normalization_instruction"]] 

    # Remove square brackets from the Normalization_params strings
    descriptions_csv["Normalization_params"] = descriptions_csv["Normalization_params"].str.strip("[]").str.replace("'", "")

    if "Unit Conversion" in normalization_params and category in ["Home And Garden", "Office Products"]:
        dimensions = ['Width', 'Height', 'Depth', 'Length']
        if any(dim in descriptions_csv['Attribute'].values for dim in dimensions):
            combined_row = pd.DataFrame([{
                'Category': category, 
                'Attribute': 'Width/Height/Depth/Length', 
                'Normalization_params': 'Unit Conversion', 
                'Normalization_instruction': """
                Convert and standardize the Width/Height/Depth/Length measurements to centimeters (cm). 
                If the Width/Height/Depth/Length is not already in cm, convert it from inches, mm, meters, yards, or feet to cm.
                If no unit is specified, assume the measurement is in inches.
                For Width/Height/Depth/Length ranges, use the larger value in the range for conversion.
                Output the final Width/Height/Depth/Length value as a single numeric figure in cm, rounded to one decimal place.
                Exclude any unit indicators in the output."
                """
            }])
            descriptions_csv = pd.concat([descriptions_csv, combined_row], ignore_index=True) 
            # Remove individual dimension rows
            descriptions_csv = descriptions_csv[~descriptions_csv['Attribute'].isin(dimensions)]

    if normalized_only:
        guidelines = descriptions_csv[(descriptions_csv["Category"] == category) & (descriptions_csv["Normalization_params"].isin(normalization_params))]

    else:
        guidelines = descriptions_csv[(descriptions_csv["Category"] == category)]

    guidelines_str = ''
    attributes_without_instructions = []

    for _, row in guidelines.iterrows():
        attribute = row["Attribute"]
        instruction = row["Normalization_instruction"]

        if pd.isna(instruction):
            attributes_without_instructions.append(attribute)
        else:
            guidelines_str += f"{attribute}: {instruction}\n"

    # Creating a single entry for attributes without instructions
    if attributes_without_instructions:
        no_instruction_str = ", ".join(attributes_without_instructions) + ": Extract as is."
        guidelines_str += no_instruction_str

    return guidelines_str
    
def parse_llm_response_to_json(response):
    """Convert a response to a JSON object. This method is implemented for LLama 7B"""
    """Example response:  Stove
                            Fuel: Gas
                            Stove Type: Portable
                            Brand Name: Hewolf
                            Material: Stainless Steel
                            Applicable Seasoning Type: n/a
                            Model: n/a """

    # Split Response by new line
    response_parts = response.split('\n')
    response_parts = [part.strip() for part in response_parts if part.strip() != '']

    # Try to parse line as a JSON object
    for response_part in response_parts:
        parsed_response_part = response_part.replace('Human:', '').replace('AI:', '').replace('System:', '')\
            .replace('Algorithm:', '').replace('"s', "'s").replace("'", '"').strip()
        try:
            response_dict = json.loads(parsed_response_part)
            return response_dict
        except:
            print(parsed_response_part)
            pass

    # Try to parse line as a JSON object - 2nd attempt
    response_dict = {}
    # Parse response as a dictionary
    for response_part in response_parts:
        parsed_response_part = response_part.replace('Human:', '')\
            .replace('AI:', '').replace('System:', '').strip()
        if ':' in parsed_response_part:
            parsed_response_part = parsed_response_part.split(':')
            if len(parsed_response_part) == 2 and type(parsed_response_part[0]) == str and type(parsed_response_part[1]) == str:
                response_dict[parsed_response_part[0].replace('"', '').replace("'", "").strip()] = parsed_response_part[1].replace('"', '').replace("'", "").strip().rstrip(',!')

    return response_dict


def merge_attributes(dict1, dict2):
    # Combine two dictionaries, merging values for matching keys
    merged = {**dict1, **dict2}
    for key in dict1.keys() & dict2.keys():
        if isinstance(dict1[key], list) and isinstance(dict2[key], list):
            merged[key] = list(set(dict1[key] + dict2[key]))
        elif isinstance(dict1[key], dict) and isinstance(dict2[key], dict):
            merged[key] = merge_attributes(dict1[key], dict2[key])
    return merged


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
        # find all decimal values
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

def convert_tuple_keys_to_string(dictionary):
    if isinstance(dictionary, dict):
        converted_dict = {}
        for key, value in dictionary.items():
            if isinstance(key, tuple):
                string_key = '__'.join(map(str, key))  # Convert each element to string and join
                converted_dict[string_key] = convert_tuple_keys_to_string(value)
            else:
                converted_dict[key] = convert_tuple_keys_to_string(value)
        return converted_dict
    else:
        return dictionary

def detect_data_types(task_dict):
    for example in task_dict["examples"]:
        for attribute, values in example['target_scores'].items():
            if values != {"n/a": 1}:
                for value_key, value_details in values.items():
                    data_type_info = determine_data_type(example['category'], attribute, value_key)
                    values[value_key] = data_type_info

    return task_dict

def update_target_scores(target_scores, original_data_types):
    updated_scores = {}
    for value in target_scores:
        # If the value is 'n/a', skip data type assignment
        if value == "n/a":
            updated_scores[value] = 'n/a'
            continue

        data_type = next(iter(original_data_types), None)
        if data_type:
            updated_scores[value] = data_type
        else:
            updated_scores[value] = 'unknown'

    return updated_scores

def detect_data_types_for_normalized_data(task_dict_normalized, title, description, dataset):
    with open('prompts/task_template.json', 'r') as f:
        task_dict_original = json.load(f)

    task_dict_original["dataset_name"] = dataset
    task_dict_original = update_task_dict_from_test_set(task_dict_original, title, description)
    task_dict_original = detect_data_types(task_dict_original)

    # Sort both original and normalized task dicts by these fields to align their orders.
    task_dict_original['examples'].sort(key=lambda x: (x['input_title'], x['input_description']))
    task_dict_normalized['examples'].sort(key=lambda x: (x['input_title'], x['input_description']))

    for i in range(len(task_dict_normalized['examples'])):
        original_example = task_dict_original['examples'][i]
        normalized_example = task_dict_normalized['examples'][i]

        for attribute in normalized_example['target_scores']:
            if attribute in original_example['target_scores']:
                original_data_types = set(original_example['target_scores'][attribute].values())
                normalized_example['target_scores'][attribute] = update_target_scores(
                    normalized_example['target_scores'][attribute], original_data_types)

    return task_dict_normalized

def subset_task_dict(task_dict):
    directory_path = f"data/descriptions/wdc/descriptions.csv"
    descriptions_csv = pd.read_csv(directory_path, sep=";")

    # Filter out rows where Normalization_instruction is not null
    descriptions_csv = descriptions_csv[descriptions_csv['Normalization_instruction'].notnull()]

    category_attributes_with_guideline = {}

    for index, row in descriptions_csv.iterrows():
        category = row['Category']
        attribute = row['Attribute']
        if category not in category_attributes_with_guideline:
            category_attributes_with_guideline[category] = []
        if attribute not in category_attributes_with_guideline[category]:
            category_attributes_with_guideline[category].append(attribute)

    for category, attributes in task_dict['known_attributes'].items():
        if category in category_attributes_with_guideline:
            task_dict['known_attributes'][category] = [attr for attr in attributes if attr in category_attributes_with_guideline[category]]
        else:
            task_dict['known_attributes'][category] = []

    # Filter examples
    for example in task_dict['examples']:
        category = example['category']
        if category in category_attributes_with_guideline:
            valid_attributes = category_attributes_with_guideline[category]
            example['target_scores'] = {attr: scores for attr, scores in example['target_scores'].items() if attr in valid_attributes}
            
            for pred_key in ['pred', 'post_pred']:
                pred_str = example[pred_key]
                if pred_str:  
                    try:
                        pred = json.loads(pred_str)
                        example[pred_key] = json.dumps({attr: value for attr, value in pred.items() if attr in valid_attributes})
                    except json.JSONDecodeError:
                        print(f"Error decoding JSON for {pred_key}: {pred_str}")
                else:
                    example[pred_key] = json.dumps({})

    return task_dict