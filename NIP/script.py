benchmarkfile = open("benchmark.txt",'r')
benchmarkOMPfile = open("benchmarkOMP.txt",'r')
benchmarkOMPAltfile = open("benchmarkOMPAlt.txt",'r')
benchmarkKernelfile = open("benchmarkKernel.txt",'r')
benchmarkKernelAltfile = open("benchmarkKernelAlt.txt",'r')
text1 = benchmarkfile.readlines()
text2 = benchmarkOMPfile.readlines()
text3 = benchmarkOMPAltfile.readlines()
text4 = benchmarkKernelfile.readlines()
text5 = benchmarkKernelAltfile.readlines()
pos = 0
    
for pos in range(len(text1)-1):
    el = text1[pos].split()
    print(el)
    size = el[2]
    mean = el[7]
    sigma = el[11]
    min = el[13]
    max = el[14]
    print(size,mean,sigma,min,max)

for pos in range(len(text2)-1):
    el = text2[pos].split()
    print(el)
    size = el[2]
    numthreads = el[5]
    mean = el[8]
    sigma = el[12]
    min = el[14]
    max = el[15]
    print(size,numthreads,mean,sigma,min,max)

for pos in range(len(text3)-1):
    el = text3[pos].split()
    print(el)
    size = el[2]
    numthreads = el[5]
    mean = el[8]
    sigma = el[12]
    min = el[14]
    max = el[15]
    print(size,numthreads,mean,sigma,min,max)

for pos in range(len(text4)-1):
    el = text4[pos].split()
    print(el)
    size = el[2]
    numthreads = el[5]
    mean = el[8]
    sigma = el[12]
    min = el[14]
    max = el[15]
    print(size,numthreads,mean,sigma,min,max)
    
for pos in range(len(text5)-1):
    el = text5[pos].split()
    print(el)
    size = el[2]
    numthreads = el[5]
    mean = el[8]
    sigma = el[12]
    min = el[14]
    max = el[15]
    print(size,numthreads,mean,sigma,min,max)
        


benchmarkfile.close()
benchmarkOMPfile.close()
benchmarkOMPAltfile.close()
benchmarkKernelfile.close()
benchmarkKernelAltfile.close()
