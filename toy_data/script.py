# python translate.py
#   --data_dir /Volumes/ML_Data/seq2seq --train_dir /Users/derekchen/Documents/active_projects/wmt/
#   --en_vocab_size=40000 --fr_vocab_size=40000
#   --size=256 --num_layers=2 --steps_per_checkpoint=50


# curl -s -o primary_debate03.csv https://raw.githubusercontent.com/gtadiparthi/debate-parser/master/output/rep_debate3_output.csv
# curl -o primary_debate04.csv https://raw.githubusercontent.com/gtadiparthi/debate-parser/master/output/rep_debate4_output.csv
# curl -o primary_debate05.csv https://raw.githubusercontent.com/gtadiparthi/debate-parser/master/output/rep_debate5_output.csv
# curl -o primary_debate06.csv https://raw.githubusercontent.com/gtadiparthi/debate-parser/master/output/rep_debate6_output.csv
# curl -o primary_debate07.csv https://raw.githubusercontent.com/gtadiparthi/debate-parser/master/output/rep_debate7_output.csv
# curl -o primary_debate08.csv https://raw.githubusercontent.com/gtadiparthi/debate-parser/master/output/rep_debate8_output.csv
# curl -o primary_debate09.csv https://raw.githubusercontent.com/gtadiparthi/debate-parser/master/output/rep_debate9_output.csv
# curl -o primary_debate10.csv https://raw.githubusercontent.com/gtadiparthi/debate-parser/master/output/rep_debate10_output.csv
# curl -o primary_debate11.csv https://raw.githubusercontent.com/gtadiparthi/debate-parser/master/output/rep_debate11_output.csv
# curl -o primary_debate12.csv https://raw.githubusercontent.com/gtadiparthi/debate-parser/master/output/rep_debate12_output.csv



data = []
with open ("encoded.txt", "r") as f:
    for line in f:
        data.append([int(char) for char in line.split()])
enc = data[:50]
dec = data[50:]

def inspectWord(word):
    print "".join(map(chr, word))
inspectWord(dec[29])