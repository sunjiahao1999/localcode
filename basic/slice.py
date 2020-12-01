def trim(s):
    if len(s) == 0:
        return s
    while s[0] == ' ':
        s = s[1:]
        if len(s) == 0:
            return s
    while s[-1] == ' ':
        s = s[:-1]
        if len(s) == 0:
            return s
    return s
