import os
import io
import codecs
import time
import collections
import nltk.tag
from nltk.tag.stanford import CoreNLPNERTagger
import nltk.tokenize.stanford as st
import re
import zipfile

def extract_entity_phrases(data, classes=[ 'LOCATION', 'PERSON']):

    # Extract entities of selected classes, add index to enable merge to phrases
    entities = [ (i, word, wclass)
        for (i, (word, wclass)) in enumerate(data) if classes is None or wclass in classes ]

    # Merge adjacent entities having the same classifier
    for i in range(len(entities) - 1, 0, -1):
        if entities[i][0] == entities[i - 1][0] + 1 and entities[i][2] == entities[i - 1][2]:
            entities[i - 1] = (entities[i - 1][0], entities[i - 1][1] + " " + entities[i][1], entities[i - 1][2])
            del entities[i]

    # Remove index in returned data
    return [ (word, wclass) for (i, word, wclass) in entities  ]

def extract_document_info(filename):
    document_name = os.path.basename(os.path.splitext(filename)[0])
    treaty_id, lang, *tail = document_name.split('_')
    return (document_name, treaty_id, lang)

def create_ner_tagger(options):
    return CoreNLPNERTagger(url=options['server_url'], encoding='utf8')

def create_tokenizer(options):
    return st.CoreNLPTokenizer(url=options['server_url'], encoding='utf8')

def read_file(filename):
    with codecs.open(filename, "r", "utf-8") as f:
        return f.read()

def create_statistics(entities):
    wc = collections.Counter()
    wc.update(entities)
    return wc

def serialize_content(stats, filename, token_count):
    document_name, treaty_id, lang = extract_document_info(filename)
    data = [ (document_name, treaty_id, lang, word, wclass, stats[(word, wclass)], token_count) for (word, wclass) in stats  ]
    content = '\n'.join(map(lambda x: ';'.join([str(y) for y in x]), data))
    return content

def write_content(outfile, content):
    if content != '':
        outfile.write(content)
        outfile.write('\n')

def main(options):

    nerrer = create_ner_tagger(options)
    tokenizer = create_tokenizer(options)
    outfile = os.path.join(options['output_folder'], "output_" + time.strftime("%Y%m%d_%H%M%S") + ".csv")
    tags = [ 'NUMBER', 'LOCATION', 'DATE', 'MISC', 'ORGANIZATION', 'DURATION', 'SET', 'ORDINAL', 'PERSON' ]

    for zip_source in options["zip_sources"]:
        with io.open(outfile, 'w', encoding='utf8') as o:
            with zipfile.ZipFile(zip_source) as pope_zip:
                for filename in pope_zip.namelist():
                    with pope_zip.open(filename) as pope_file:
                        try:
                            text = pope_file.read().decode("utf-8")
                            tokens = tokenizer.tokenize(text)
                            data = nerrer.tag(tokens)
                            entities = extract_entity_phrases(data, tags)  # [ 'LOCATION', 'PERSON', 'ORGANIZATION' ])
                            statistics = create_statistics(entities)
                            content = serialize_content(statistics, filename, len(tokens))
                            write_content(o, content)
                        except Exception as ex:
                            print('Failed: ' + filename)
if __name__ == "__main__":

    data_folder = '../data/'
    options = {
        "zip_sources": [ os.path.join(data_folder, "ner_test_data.zip") ],
        'server_url': 'http://localhost:9001',
        'output_folder': data_folder,
    }

    main(options)
