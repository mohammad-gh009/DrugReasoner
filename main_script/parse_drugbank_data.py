import pandas as pd  
import xml.etree.ElementTree as ET
from collections import defaultdict

def parse_drugbank(xml_file: str)-> pd.DataFrame:
    """
    input: the path to the XML file 
    output: pandas dataframe
    """
    ns = {'db': 'http://www.drugbank.ca'}
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    drugs = []
    
    for drug in root.findall('db:drug', ns):
        drug_data = defaultdict(lambda: None)
        # if count >= 5:
        #     break  
        # Basic information
        drug_data['drugbank_ids'] = [{
            'value': id_elem.text,
            'primary': id_elem.get('primary', 'false').lower() == 'true'
        } for id_elem in drug.findall('db:drugbank-id', ns)]
        
        drug_data['name'] = drug.findtext('db:name', namespaces=ns)
        drug_data['description'] = drug.findtext('db:description', namespaces=ns)
        drug_data['cas_number'] = drug.findtext('db:cas-number', namespaces=ns)
        drug_data['unii'] = drug.findtext('db:unii', namespaces=ns)
        
        # Physical properties
        drug_data['average_mass'] = parse_float(drug.findtext('db:average-mass', namespaces=ns))
        drug_data['monoisotopic_mass'] = parse_float(drug.findtext('db:monoisotopic-mass', namespaces=ns))
        drug_data['state'] = drug.findtext('db:state', namespaces=ns)
        
        # Groups and classifications
        drug_data['groups'] = [g.text for g in drug.find('db:groups', ns).findall('db:group', ns)] if drug.find('db:groups', ns) else []
        drug_data['classification'] = parse_classification(drug.find('db:classification', ns), ns)
        
        # References and properties
        drug_data['general_references'] = parse_references(drug.find('db:general-references', ns), ns)
        drug_data['calculated_properties'] = parse_properties(drug.find('db:calculated-properties', ns), ns)
        drug_data['experimental_properties'] = parse_properties(drug.find('db:experimental-properties', ns), ns)
        
        # Structural components
        drug_data['salts'] = parse_salts(drug.find('db:salts', ns), ns)
        drug_data['synonyms'] = parse_synonyms(drug.find('db:synonyms', ns), ns)
        drug_data['products'] = parse_products(drug.find('db:products', ns), ns)
        drug_data['patents'] = parse_patents(drug.find('db:patents', ns), ns)
        
        # Pharmacological data
        drug_data['indication'] = drug.findtext('db:indication', namespaces=ns)
        drug_data['pharmacodynamics'] = drug.findtext('db:pharmacodynamics', namespaces=ns)
        drug_data['mechanism_of_action'] = drug.findtext('db:mechanism-of-action', namespaces=ns)
        drug_data['toxicity'] = drug.findtext('db:toxicity', namespaces=ns)
        
        # PK/PD properties
        drug_data['metabolism'] = drug.findtext('db:metabolism', namespaces=ns)
        drug_data['half_life'] = drug.findtext('db:half-life', namespaces=ns)
        drug_data['protein_binding'] = drug.findtext('db:protein-binding', namespaces=ns)
        
        # Add more elements as needed following the same pattern
        
        # Attributes
        drug_data['type'] = drug.get('type')
        drug_data['created'] = drug.get('created')
        drug_data['updated'] = drug.get('updated')
        
        drugs.append(dict(drug_data))
        
    df = pd.DataFrame(drugs)
    return df

# Helper functions
def parse_float(value):
    try:
        return float(value) if value else None
    except ValueError:
        return None

def parse_classification(classification, ns):
    if classification is None:
        return None
    return {
        'description': classification.findtext('db:description', namespaces=ns),
        'direct_parent': classification.findtext('db:direct-parent', namespaces=ns),
        'kingdom': classification.findtext('db:kingdom', namespaces=ns),
        'superclass': classification.findtext('db:superclass', namespaces=ns),
        'class': classification.findtext('db:class', namespaces=ns),
        'subclass': classification.findtext('db:subclass', namespaces=ns),
        'alternative_parents': [p.text for p in classification.findall('db:alternative-parent', ns)],
        'substituents': [s.text for s in classification.findall('db:substituent', ns)]
    }

def parse_properties(properties, ns):
    if properties is None:
        return []
    return [{
        'kind': prop.findtext('db:kind', namespaces=ns),
        'value': prop.findtext('db:value', namespaces=ns),
        'source': prop.findtext('db:source', namespaces=ns)
    } for prop in properties.findall('db:property', ns)]

def parse_salts(salts, ns):
    if salts is None:
        return []
    return [{
        'name': salt.findtext('db:name', namespaces=ns),
        'unii': salt.findtext('db:unii', namespaces=ns),
        'cas_number': salt.findtext('db:cas-number', namespaces=ns),
        'inchikey': salt.findtext('db:inchikey', namespaces=ns)
    } for salt in salts.findall('db:salt', ns)]

def parse_synonyms(synonyms, ns):
    if synonyms is None:
        return []
    return [{
        'name': syn.text,
        'language': syn.get('language'),
        'coder': syn.get('coder')
    } for syn in synonyms.findall('db:synonym', ns)]

def parse_patents(patents, ns):
    if patents is None:
        return []
    return [{
        'number': pat.findtext('db:number', namespaces=ns),
        'country': pat.findtext('db:country', namespaces=ns),
        'approved': pat.findtext('db:approved', namespaces=ns),
        'expires': pat.findtext('db:expires', namespaces=ns)
    } for pat in patents.findall('db:patent', ns)]

def parse_products(products, ns):
    if products is None:
        return []
    return [{
        'name': prod.findtext('db:name', namespaces=ns),
        'labeller': prod.findtext('db:labeller', namespaces=ns),
        'ndc_id': prod.findtext('db:ndc-id', namespaces=ns),
        'dosage_form': prod.findtext('db:dosage-form', namespaces=ns),
        'strength': prod.findtext('db:strength', namespaces=ns)
    } for prod in products.findall('db:product', ns)]

def parse_references(references, ns):
    if references is None:
        return {}
    return {
        'articles': [{
            'pubmed_id': art.findtext('db:pubmed-id', namespaces=ns),
            'citation': art.findtext('db:citation', namespaces=ns)
        } for art in references.find('db:articles', ns).findall('db:article', ns)],
        'textbooks': [{
            'isbn': book.findtext('db:isbn', namespaces=ns),
            'citation': book.findtext('db:citation', namespaces=ns)
        } for book in references.find('db:textbooks', ns).findall('db:textbook', ns)]
    }

def convert_properties(original_list:list):
    """
    input: columns that contains list such as "experimental_properties", "calculated_properties" 
    
    fn: this will convert {kind:x , value:y , source:z} to {x:y}
    """ 
    if not isinstance(original_list, list):  # Handle null/non-list values
        return original_list
    
    converted = []
    for prop in original_list:
        try:
            converted.append({prop['kind']: prop['value']})
        except (KeyError, TypeError):  # Handle malformed entries
            continue  # or keep original as fallback
    return converted

def extract_smiles(properties):
    if isinstance(properties, list):  # Ensure the entry is a list
        for entry in properties:
            if "SMILES" in entry:
                return entry["SMILES"]
    return None  # Return None if no SMILES found or invalid format