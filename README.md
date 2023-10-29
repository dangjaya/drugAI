# Drug-MCTS

This is the implementation of Drug Design Generative Model called Drug-MCTS paper published on mdpi.com/XXXX

In order to run the codebase, you need to run pre-processing steps as follow :

- Run PIP install / conda install

  pip install -r requirements.txt

- Download BindingDB file from their website ( choose to download TSV file ) :

  https://www.bindingdb.org/rwd/bind/chemsearch/marvin/Download.jsp

  Place the file under data sub-directory

- Pre-process this new file by running jupyter notebook and run this notebook under notebooks sub-dir :

  Preprocess BindingDB for training.ipynb

  The notebook will produce a new file called preProcess.csv that will be used as input file for pyTorch training

# Train the model 

To begin training, run this code:
```
python train.py 
```

# Generate small molecules for a given protein sequence

To generate small molecules , run the code below :

```
python predict.py
```

e.q:

And enter the protein sequence when prompted :
Enter protein sequence  :
MKTPWKVLLGLLGAAALVTIITVPVVLLNKGTDDATADSRKTYTLTDYLKNTYRLKLYSLRWISDHEYLYKQENNILVFNAEYGNSSVFLENSTFDEFGHSINDYSISPDGQFILLEYNYVKQWRHSYTASYDIYDLNKRQLITEERIPNNTQWVTWSPVGHKLAYVWNNDIYVKIEPNLPSYRITWTGKEDIIYNGITDWVYEEEVFSAYSALWWSPNGTFLAYAQFNDTEVPLIEYSFYSDESLQYPKTVRVPYPKAGAVNPTVKFFVVNTDSLSSVTNATSIQITAPASMLIGDHYLCDVTWATQERISLQWLRRIQNYSVMDICDYDESSGRWNCLVARQHIEMSTTGWVGRFRPSEPHFTLDGNSFYKIISNEEGYRHICYFQIDKKDCTFITKGTWEVIGIEALTSDYLYYISNEYKGMPGGRNLYKIQLSDYTKVTCLSCELNPERCQYYSVSFSKEAKYYQLRCSGPGLPLYTLHSSVNDKGLRVLEDNSALDKMLQNVQMPSKKLDFIILNETKFWYQMILPPHFDKSKKYPLLLDVYAGPCSQKADTVFRLNWATYLASTENIIVASFDGRGSGYQGDKIMHAINRRLGTFEVEDQIEAARQFSKMGFVDNKRIAIWGWSYGGYVTSMVLGSGSGVFKCGIAVAPVSRWEYYDSVYTERYMGLPTPEDNLDHYRNSTVMSRAENFKQVEYLLIHGTADDNVHFQQSAQISKALVDVGVDFQAMWYTDEDHGIASSTAHQHIYTHMSHFIKQCFSLP


