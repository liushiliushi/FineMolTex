import os
from six.moves.urllib.request import urlretrieve
import ssl

def download_and_extract_compound_file(PubChem_datasets_home_folder, compound_file_name):
    compound_url = "https://ftp.ncbi.nlm.nih.gov/pubchem/Compound/CURRENT-Full/SDF/{}".format(compound_file_name)

    zipped_compound_file_path = "{}/{}".format(PubChem_datasets_home_folder, compound_file_name)
    ssl._create_default_https_context = ssl._create_unverified_context
    if not os.path.exists(zipped_compound_file_path):
        print("Downloading {} to {} ...".format(compound_url, zipped_compound_file_path))
        urlretrieve(compound_url, zipped_compound_file_path)
    return
