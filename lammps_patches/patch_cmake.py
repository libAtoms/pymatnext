import re
in_sec = False
have_NS = False
with open("orig/cmake/CMakeLists.txt") as fin, open("new/cmake/CMakeLists.txt", "w") as fout:
    for l in fin:
        if in_sec:
            if re.search(r'\bNS\b', l):
                have_NS = True
            m = re.search(r'^(\s*)([^\s)]+)\)\s*$', l)
            if m and not have_NS:
                # add NS
                l = m.group(1) + m.group(2) + "\n"
                l += m.group(1) + "NS)\n"
                in_sec = False

        elif re.match(r'set\s*\(\s*STANDARD_PACKAGES', l):
            in_sec = True

        fout.write(l)
