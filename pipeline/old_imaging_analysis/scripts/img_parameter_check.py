
# ToDo: Group checks, and remove checks which are taken over by snakemake

def check_parameters(config):
    pass
    return True


#check of Contour_Limit and num_measures
if len(Contour_Limit) != len(num_measures):
    print('ERROR: Contour_Limit and num_measures dimensions do not match')
    sys.exit(-1)

#check of lowcut and highcut with num_measures
if len(lowcut) != len(highcut):
    print('ERROR: Wrong lowcut and highcut dimensions')
    sys.exit(-1)

if len(zone) != len(lowcut):
    print ('ERROR: Wron zone and passband dimensions')
    sys.exit(-1)

if len(lowcut) != len(num_measures):
    print('ERROR: Wrong passband and num_measures dimensions')
    sys.exit(-1)

#check of lowcut and highcut components

for i in range(0,len(highcut)):
    try:
        if len(lowcut[i]) != len(highcut[i]):
            print('ERROR: Non-matching dimension for the %d element of lowcut and highcut' % i)
            sys.exit(-1)

        if len (zone[i]) != len (lowcut[i]):
            print ('ERROR: Non-matching dimension for %d element of zone and lowcut' % i)
            print ('Len zone is %d while Len lowcut is %d' % (len(zone[i]), len(lowcut[i])))
            sys.exit(-1)

        for l_el, h_el in zip (lowcut[i], highcut[i]):
            if l_el is None or h_el is None:
                continue

            if l_el > h_el:
                print ('ERROR: Lowcut[' + str(i) + '] element ' + str(l_el) + \
                       ' is greater than highcut[' + str(i) +'] element ' + str(h_el))
                sys.exit(-1)

    except TypeError:
        if lowcut[i] is None and highcut[i] is None and zone[i] is None:
            continue

        else:
            print('ERROR: Wrong syntax element %d' % i)
            sys.exit(-1)

#check directory
if os.path.exists(DATA_DIR) == False:
    print('ERROR:The selected DATA_DIR does not exist')
    sys.exit(-1)
