# Compare 2 files and output the difference to difference.txt

def compare(File1,File2):
    with open(File1,'r') as f:
        d=set(f.readlines())


    with open(File2,'r') as f:
        e=set(f.readlines())
	
	# Create file
    open('difference.txt','w').close()

    with open('difference.txt','a') as f:
        for line in list(d-e):
           f.write(line)

compare("spam-mail.tr.label","result.txt")