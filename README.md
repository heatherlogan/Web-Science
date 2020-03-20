# Web-Science


Use from commandline - to set up requirements:
  
    pip3 freeze > requirements.txt 
  
    pip3 install -r requirements.txt 
  
  
Create local mongoDB instance. 
 
To run task1: Twitter Crawler 

    python3 tweep.py
   
To run task2: data Analytics (performed on sample of tweets stored in data/sample_tweets.py)
*disclaimer due sample data is from the same 10 minute period, therefore the graphs will only show one large bin. This is to display functionality. For full results, see report. 

    python3 data.py 

To run task3: Offensive language identification


    python3 offensive_lang.py EVALUATION_MODE MODEL_TYPE
    
    # where EVALUATION_MODE = True or False
    # and MODEL_TYPE = LR, MNB or RF
    
Output files are stored in /data/ folder.     
