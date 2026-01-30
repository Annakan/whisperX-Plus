The goal is to build an evaluator tool for WhisperX.


We will take some audio and a reference transcript and we will need to generate a transcript from the audio while varying the parameters of WhisperX and compare it to the reference transcript through various metrics, mostly based on Levenshtein distance, mostly using difflib and/or [diff-match-patch](https://github.com/google/diff-match-patch)

# First step

In the @directory:/home/gryom/SyncRoot/TFR/AIAN/WhispComp/SrcDataSet there is a set of directories with audio and text file ending in _REF.
Those are audio files and their associated reference transcripts.

there consist of several parts and we are, for now, interested only by the part after the two lines consecutive line of repeated `-` (X5+) followed immediately by a line of repeated `=` (X5+), till the end of the document

This last part is made of consecutive blocks of 3 lines :
- an **optional** line [<chars>] |  (digits:) (digit+ ms)
- a locuteur bloc `+A:` where `A` is locuteur identifier
- a transcript line containing the speech of the locuteur

We will need a function "parse_source" that given the path of such a reference transcript will produce a list of segments in the order they appear in the source document, where each segment is a dict with the following keys :
- locuteur
- text

We will need also a `prepare(<segments>, with_locuteur:bool:True, with_paragraphs:bool:True, lower_no_punct:bool:True)` function that process the structure resulting from "parse_source" and produce a continuous line/stream of words including the locuteur if with_locuteur is True and '\n' between segments if with_paragraphs is True.
Exemple given the following segments:

+A:
Ensuite, nous retrouverons notre présidente. Donc nous en étions à l'amendement 2008. Monsieur Cabrolier, je vous écoute.
+B:
Merci, madame la présidente. Donc cet amendement concerne, comme on en a parlé avant le dîner, 

Produce either (with_locuteur=True, with_paragraphs=True)
```
+A: Ensuite, nous retrouverons notre présidente. Donc nous en étions à l'amendement 2008. Monsieur Cabrolier, je vous écoute.
+B: Merci, madame la présidente. Donc cet amendement concerne, comme on en a parlé avant le dîner,
```
or (with_locuteur=False, with_paragraphs=True)
```
Ensuite, nous retrouverons notre présidente. Donc nous en étions à l'amendement 2008. Monsieur Cabrolier, je vous écoute.
Merci, madame la présidente. Donc cet amendement concerne, comme on en a parlé avant le dîner, 
```
or (with_locuteur=False, with_paragraphs=False)
```Ensuite, nous retrouverons notre présidente. Donc nous en étions à l'amendement 2008. Monsieur Cabrolier, je vous écoute.Merci, madame la présidente. Donc cet amendement concerne, comme on en a parlé avant le dîner,```

The lower_no_punct variation means that we remove all punctuation and lower case the text.


# Second step

We need to use our whisperX functionality to generate a transcript from the audio with varying parameters :
- `model`: The model to use for transcription. Can be one of "tiny", "small", "medium", or "large-v3".
- language is fixed to 'fr' # so we don't need to vary it nor include it in the output
- beam_size varies from 5 to 10 by step of 2
- patience varies along the discrete values [1,2,3]
- preprocess varies along the discrete values [0,1,2,3,4]

Each variation must produce a transcript file in a subdirectory of a given 'output' directory with the same name but with a suffix corresponding to the variation and a json file "params.json" containing the parameters used for the transcription and the source file name.

# Third step

We take each transcript produced, extract the same segments as the reference transcript, apply the prepare function to each transcript and measure the Levenshtein distance with the correspondingly prepared reference transcript and produce a compare-result file with the params and the comparison results in a csv format file.

# Fourth step

We aggregate the compare-result files and produce a final csv containing all data ready to be loaded into a Ipython or marimo notebook for analysis.

# How this should work practically

My current plan is to have a CLI tool that takes two parameters :
- source_data_dir: the directory containing the source data
- run_dir: should point to a directory containing a runplan.yaml file
- 
This runplan.yaml file will contain the following keys:


- to_process : a list of the subdirectories of source_data_dir to process (each subdirectory should contain an audio file and a reference transcript file)
- base_args : a dictionary containing the base arguments to be used for the transcription, those are the fixed arguments that would override the default but do not vary. The keys are the same as the ones in the whisperX CLI
- varying_args : for each variation the key is the name of the argument to vary and the value is a list of the values to vary it to. The keys names are the same as the ones in the whisperX CLI

Upon reading the runplan.yaml file, the CLI tool process each `to_process` subdirectory of source_data_dir and will copy the content (audio and transcript).
I will then create subdirectories inside this newly copied dir for each variation in the varying_args dict and store in it the result of the corresponding transcription and then generate the following variations through the prepare function:
- with_locuteur=True + with_paragraphs=True: VA
- with_locuteur=True + with_paragraphs=False + lower_no_punct=True: VB 
- with_locuteur=True + with_paragraphs=True + lower_no_punct=True: The textual version without punctuation and lowercased : VC
- with_locuteur=False + with_paragraphs=False + lower_no_punct=True: VD (The raw version)

And finally, a result.csv file containing the levenshtein distance for each variation with the identical transformation of the reference transcript: one line with the following columns :
- base directory name
- the variation name
- 4 columns with the levenshtein distance for each of the 4 variations.

Once all result.csv files are generated, the CLI tool will aggregate them and produce a final csv containing all data ready to be loaded into a Ipython or marimo notebook for analysis in the run_dir.

At each step the tool will compute the file name or the output it should produce at this step and check if this file already exists. If it does, it will skip this step and move to the next one.


The CLI tool will print a progress bar showing the progress of the process ([TQDM](https://github.com/tqdm/tqdm) or better if you have a suggestion).

 













