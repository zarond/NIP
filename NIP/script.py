#prefix = "floatres/"
#prefix = "doubleres/"
#prefix = "floatrescluster/"
prefix = "doublerescluster/"
benchmarkfile = open(prefix+"benchmark.txt",'r')
benchmarkOMPfile = open(prefix+"benchmarkOMP.txt",'r')
benchmarkOMPAltfile = open(prefix+"benchmarkOMPAlt.txt",'r')
benchmarkKernelfile = open(prefix+"benchmarkKernel.txt",'r')
benchmarkKernelAltfile = open(prefix+"benchmarkKernelAlt.txt",'r')
text1 = benchmarkfile.readlines()
text2 = benchmarkOMPfile.readlines()
text3 = benchmarkOMPAltfile.readlines()
text4 = benchmarkKernelfile.readlines()
text5 = benchmarkKernelAltfile.readlines()
pos = 0

print('benchmark:')
for pos in range(len(text1)-1):
    el = text1[pos].split()
    #print(el)
    size = el[2]
    mean = el[7]
    sigma = el[11]
    min = el[13]
    max = el[14]
    print(size,mean,sigma,min,max)
print(text1[-1])

print('benchmark OMP:')
for pos in range(len(text2)-1):
    el = text2[pos].split()
    #print(el)
    size = el[2]
    numthreads = el[5]
    mean = el[8]
    sigma = el[12]
    min = el[14]
    max = el[15]
    print(size,numthreads,mean,sigma,min,max)
print(text2[-1])

print('benchmark OMP second algorythm:')
for pos in range(len(text3)-1):
    el = text3[pos].split()
    #print(el)
    size = el[2]
    numthreads = el[5]
    mean = el[8]
    sigma = el[12]
    min = el[14]
    max = el[15]
    print(size,numthreads,mean,sigma,min,max)
print(text3[-1])

print('benchmark kernel:')
for pos in range(len(text4)-1):
    el = text4[pos].split()
    #print(el)
    size = el[2]
    numthreads = el[5]
    mean = el[8]
    sigma = el[12]
    min = el[14]
    max = el[15]
    print(size,numthreads,mean,sigma,min,max)
print(text4[-1])

print('benchmark kernel second algorythm:')
for pos in range(len(text5)-1):
    el = text5[pos].split()
    #print(el)
    size = el[2]
    numthreads = el[5]
    mean = el[8]
    sigma = el[12]
    min = el[14]
    max = el[15]
    print(size,numthreads,mean,sigma,min,max)
print(text5[-1])
        


benchmarkfile.close()
benchmarkOMPfile.close()
benchmarkOMPAltfile.close()
benchmarkKernelfile.close()
benchmarkKernelAltfile.close()
