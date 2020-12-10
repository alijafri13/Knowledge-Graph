### Create virtual environment
python3 -m venv venv
source venv/bin/activate

echo "Downloading Packages..."
### Install packages
pip install wget
pip install bs4
wget https://maven.ceon.pl/artifactory/kdd-releases/pl/edu/icm/cermine/cermine-impl/1.13/cermine-impl-1.13-jar-with-dependencies.jar
pip install -r pip_libraries.txt
python -m nltk.downloader stopwords
python -m nltk.downloader wordnet
python -m nltk.downloader punkt
python -m nltk.downloader averaged_perceptron_tagger
python -m spacy download en_core_web_sm
#
git clone https://github.com/huggingface/neuralcoref.git
pip install -U spacy
python -m spacy download en
#
cd neuralcoref
pip install -r requirements.txt
pip install -e .
# #
cd ..

echo "All Packages Installed"

mkdir XML
mkdir TEXT
cp PDF/* XML/
echo converting PDF to XML
java -cp cermine-impl-1.13-jar-with-dependencies.jar pl.edu.icm.cermine.ContentExtractor -path XML

## Call text-scraping file (input: XML files, output: .txt files)
echo "Converting PDF's to Text..."
python Text_Scraper.py

### Call knowlegde graph (input: .txt files; output: dot file)
echo "Extracting Relationships..."
python Relation_Extraction.py

# deactivate
