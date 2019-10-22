import pandas as pd
import numpy as np
import re,pickle

TC_NUM_CLASSES = {
                    'yelp' : 5,
                    'yahoo': 10,
                    'amazon': 5,
                    'agnews': 4,
                    'dbpedia':14
            }
# dataset order for text classification
TC_ORDER = {
             1: ['yelp','agnews','dbpedia','amazon','yahoo'],
             2: ['dbpedia','yahoo','agnews','amazon','yelp'],
             3: ['yelp','yahoo','amazon','dbpedia','agnews'],
             4: ['agnews','yelp','amazon','yahoo','dbpedia']
        }
# dataset order for question answering
QA_ORDER = {
             1: ['quac','trweb','trwik','squad'],
             2: ['squad','trwik','quac','trweb'],
             3: ['trweb','trwik','squad','quac'],
             4: ['trwik','quac','trweb','squad']
        }
INDIVIDUAL_CLASS_LABELS = {
                            'yelp':{1:'1',2:'2',3:'3',4:'4',5:'5'},
                            'dbpedia':{1:'Company',2:'EducationalInstitution',3:'Artist',
                                       4:'Athlete',5:'OfficeHolder',6:'MeanOfTransportation',7:'Building',
                                       8:'NaturalPlace',9:'Village',10:'Animal',11:'Plant',12:'Album',
                                       13:'Film',14:'WrittenWork'},
                             'yahoo':{1:'Society & Culture',2:'Science & Mathematics',3:'Health',
                                      4:'Education & Reference',5:'Computers & Internet',6:'Sports',
                                      7:'Business & Finance',8:'Entertainment & Music',
                                      9:'Family & Relationships',10:'Politics & Government'},
                              'amazon':{1:'1',2:'2',3:'3',4:'4',5:'5'},
                              'agnews':{1:'World',2:'Sports',3:'Business',4:'Sci/Tech'}
                        }

# removes hyperlinks
preprocess = (lambda x: re.sub(r"https?:\/\/\S+\b|www\.(\w+\.)+\S*", "",str(x)))



def create_ordered_tc_data(order,base_location='data/original_data',save_location='data/ordered_data',split='test'):
    """
    creates ordered dataset for text classification with a maximum of 115,000 sequences
    and 7,600 sequences from each individual dataset for train and test data respectively
    i.e.,the size of the smallest training and test sets
    """
    dataset_sequence = TC_ORDER[order]
    ordered_dataset = {'labels':[],'content':[]}
    num_classes = 0
    max_samples = 115000 if split=='train' else 7600
    label_to_class = dict()

    for data in dataset_sequence:

        if data == 'yelp':
            df = pd.read_csv(base_location+'/'+split+'/'+data+'.csv',header=None,names=['labels','content'])
            labels = df.labels[:max_samples] + num_classes
            content= df.content[:max_samples].apply(preprocess)
            ordered_dataset['labels'].extend(list(labels))
            ordered_dataset['content'].extend(list(content))
            # Mapping new labels to classes
            for k,v in INDIVIDUAL_CLASS_LABELS[data].items():
                new_key = k + num_classes
                label_to_class[new_key] = v
            num_classes+=TC_NUM_CLASSES[data]

        elif data == 'yahoo':
            df = pd.read_csv(base_location+'/'+split+'/'+data+'.csv',header=None,names=['labels','title','content','answer'])
            labels = df.labels[:max_samples] + num_classes
            content= df.content[:max_samples].apply(preprocess)
            ordered_dataset['labels'].extend(list(labels))
            ordered_dataset['content'].extend(list(content))
            # Mapping new labels to classes
            for k,v in INDIVIDUAL_CLASS_LABELS[data].items():
                new_key = k + num_classes
                label_to_class[new_key] = v
            num_classes+=TC_NUM_CLASSES[data]
        else:
            df = pd.read_csv(base_location+'/'+split+'/'+data+'.csv',header=None,names=['labels','title','content'])
            labels = df.labels[:max_samples] + num_classes
            content= df.content[:max_samples].apply(preprocess)
            ordered_dataset['labels'].extend(list(labels))
            ordered_dataset['content'].extend(list(content))
            # Mapping new labels to classes
            for k,v in INDIVIDUAL_CLASS_LABELS[data].items():
                new_key = k + num_classes
                label_to_class[new_key] = v
            num_classes+=TC_NUM_CLASSES[data]

    ordered_dataframe = pd.DataFrame(ordered_dataset)
    ordered_dataframe.to_csv(save_location+'/'+split+'/'+str(order)+'.csv',index=False)
    with open(save_location+'/'+split+'/'+str(order)+'.pkl','wb') as f:
        pickle.dump(label_to_class,f)

create_ordered_tc_data(4)
