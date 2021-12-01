# EDA for data summarization

Requires python 3.7+

Install required dependencies:

    python -m pip install -r requirements.txt

Unzip data:

    cd app/data/
    cat data.tar.gz.* | tar xzvf -


Unzip models:

    cd app/app_models/
    cat app_models.tar.gz.* | tar xzvf -

To run the app:
    
    python launcher.py

Then to run a training, from the root of the project:

    python RL-launcher.py
