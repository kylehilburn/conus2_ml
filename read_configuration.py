from ast import literal_eval

def read_configuration(filename='configuration.txt'):

    config = {}

    print('reading configuration file=',filename)

    f = open(filename,'r')

    for aline in f:

        sline = aline.strip()
        if sline.startswith('#'): continue  #skip comments
        if len(sline) == 0: continue  #skip blank lines

        #format of a line is: variable=value
        items = [anitem.strip() for anitem in sline.split('=')]
        if len(items) != 2: continue  #bad format
        akey = items[0]
        avalue = items[1]
  
        try:
            config[akey] = literal_eval(avalue)
        except ValueError:
            config[akey] = avalue  #string

    f.close()

    return config
