import os.path

here = os.path.dirname(os.path.realpath(__file__))
subdir = "data"
subdir2 = "0706"
filename = "myfile.txt"
filepath = os.path.join(here, subdir, subdir2, filename)

f=open(filepath,"w")

f.write("This is a test message")
f.close()

