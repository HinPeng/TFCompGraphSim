if __name__ == "__main__":
    recomp_srcs = dict()
    recomp_srcs_ops = dict()
    with open("recomp_srcsinfo.txt", 'r') as fin:
        lines = [line.strip('\n') for line in fin.readlines()]
        i = 0
        while True:
            line = lines[i].split('\t')
            assert len(line) == 2
            recomp_name = line[0]
            srcs_len = int(line[1])
            i+=1
            line = lines[i]
            ops_len = int(line)
            recomp_srcs[recomp_name] = list()
            for t in range(1,srcs_len+1):
                i+=1
                line = lines[i].split('\t')
                recomp_srcs[recomp_name].append((line[0], line[1]))
            recomp_srcs_ops[recomp_name] = list()
            for t in range(1, ops_len+1):
                i+=1
                recomp_srcs_ops[recomp_name].append(lines[i])
            i+=1
            if i>=len(lines):
                break
    with open("recomp_srcsinfo_new.txt", 'w') as fout:
        for recomp_name in recomp_srcs.keys():
            fout.write("%s\t%d\t%d\n" % (recomp_name, len(recomp_srcs[recomp_name]), len(recomp_srcs_ops[recomp_name])))
            for srcs in recomp_srcs[recomp_name]:
                fout.write("%d\t%s\n" % (int(srcs[0]), srcs[1]))
            for op in recomp_srcs_ops[recomp_name]:
                fout.write("%s\n" % op)
    
