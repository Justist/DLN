import re
import sys

if len(sys.argv) is not 3:
    raise Exception("Wrong number of arguments given!")

filename = sys.argv[1]
with open(filename, "r") as a:
    alllines = a.readlines()
    epoch = re.search(r"e(\d+)", filename)[0]
with open(f"{filename}.verbatim", "w") as b:
    b.write("""
\\begin{figure}[!ht]
 \\begin{verbatim}
""")
    for line in alllines:
        b.write(line)
    b.write(
        (
            (
                """
 \\end{verbatim}
 \\caption{"""
                + f"The {sys.argv[2]} of epoch ${epoch[1:]}$."
            )
            + """}
\\end{figure}

"""
        )
    )
