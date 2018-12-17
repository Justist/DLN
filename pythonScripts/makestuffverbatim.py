import re
import sys

filename = sys.argv[1]
with open(filename, "r") as a:
	alllines = a.readlines()
	epoch = re.search(r"e(\d+)", filename).group(0)
with open(filename + ".verbatim", "w") as b:
	b.write("""
\\begin{figure}[!ht]
 \\begin{verbatim}
""")
	for line in alllines:
		b.write(line)
	b.write("""
 \\end{verbatim}
 \\caption{"""+ "The {} of epoch ${}$.".format(sys.argv[2], epoch[1:]) +"""}
\\end{figure}

""")