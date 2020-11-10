
import os
import requests
import tarfile
from glob import glob

if not os.path.exists('data'):
    os.mkdir('data')
if not os.path.exists(os.path.join('data', 'original_data')):
    os.mkdir(os.path.join('data', 'original_data'))
    os.mkdir(os.path.join('data', 'original_data', 'train'))
    os.mkdir(os.path.join('data', 'original_data', 'test'))
    

files_links = {'agnews':'https://drive.google.com/file/d/0Bz8a_Dbh9QhbUDNpeUdjb0wxRms/view?usp=sharing',
               'amazon':'https://drive.google.com/file/d/0Bz8a_Dbh9QhbUDNpeUdjb0wxRms/view?usp=sharing',
               'dbpedia':'https://drive.google.com/file/d/0Bz8a_Dbh9QhbUDNpeUdjb0wxRms/view?usp=sharing',
               'yelp':'https://drive.google.com/file/d/0Bz8a_Dbh9QhbUDNpeUdjb0wxRms/view?usp=sharing',
               'yahoo':'https://drive.google.com/file/d/0Bz8a_Dbh9Qhbd2JNdDBsQUdocVU/view?usp=sharing'}

# taken from this StackOverflow answer: https://stackoverflow.com/a/39225039
def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)    

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)

# download tarballs
for name,url in files_links.items():
    output = os.path.join('data', name+'.tar.gz')
    file_id = url.split('/')[5]
    download_file_from_google_drive(file_id, output)


# extract to train and test
for name in files_links.keys():
    with tarfile.open(os.path.join('data',name+'.tar.gz')) as tf:
        for m in tf.getmembers():
            if m.name[-4:] == '.csv':
                folder =  m.name.split('.')[0].split('/')[-1]
                m.name = name+'.csv'
                print(f'Extracting {m.name} {folder}')
                tf.extract(m, path=os.path.join('data', 'original_data', folder))

# delete tarballs
tars = glob(os.path.join('data','*.gz'))
[os.remove(t) for t in tars];



