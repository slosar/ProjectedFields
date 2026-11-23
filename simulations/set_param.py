import sys

if __name__=='__main__':
    dir = sys.argv[1]
    n = int(sys.argv[2])
    print(f'Simulation {n}')
#    c = chr(ord('A')+n)
#    print(f'Simulation {n}')
    infile = '/astro/u/khong/g/'+dir+'/param_base.txt'
    outfile= '/astro/u/khong/g/'+dir+f'/param_{n:02}.txt'
    with open(infile, 'r') as f:
        param = f.readlines()

    for i, line in enumerate(param):
        ls = line.split()
        if len(ls)==0: continue

        if ls[0]=='OutputDir':
            ls[1] = f'./output_{n:02}'+'\n'
            param[i] = '    '.join(ls)

        elif ls[0]=='Seed':
            ls[1] = str(int(ls[1])+n)+'\n'
            param[i] = '    '.join(ls)
    
    with open(outfile, 'w') as f:
        f.writelines(param)
