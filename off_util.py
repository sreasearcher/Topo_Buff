from collections import defaultdict
import numpy as np
import copy


class Graph:
    def __init__(self, vertices):
        self.graph = defaultdict(list)
        self.V = vertices

    def addEdge(self, u, v):
        u-=1
        v-=1
        self.graph[u].append(v)

    def topologicalSortUtil(self, v, visited, stack):

        visited[v] = True

        for i in self.graph[v]:
            if visited[i] == False:
                self.topologicalSortUtil(i, visited, stack)

        stack.insert(0, v)

    def topologicalSort(self):
        import numpy as np
        visited = [False] * self.V
        stack = []

        for i in range(self.V):
            if visited[i] == False:
                self.topologicalSortUtil(i, visited, stack)
        stack=np.array(stack)

        # print(stack)
        return stack


class GS():
    def __init__(self, band=1):
        self.nin_f = np.array([0,
                               np.sum([105705600, 290400, 28168800, 290400, 28168800, 290400]),
                               290400,
                               np.sum([448084224, 186624, 47962368, 186624, 47962368, 186624]),
                               186624,
                               np.sum([149585280, 64896, 24984960, 64896, 24984960, 64896]),
                               64896,
                               np.sum([248832000, 36000, 72000000, 36000, 72000000, 36000]),
                               0]) / 1e10
        self.nin_o = np.array([0.14355469,
                               0.27694702,
                               0.06674194,
                               0.17797852,
                               0.04125977,
                               0.06188965,
                               0.01318359,
                               0.03433228,
                               0]) * 8
        nin_l = len(self.nin_f)
        self.nin_g = Graph(nin_l)
        self.nin_s = list(range(nin_l))
        for i in range(nin_l):
            self.nin_g.addEdge(i+1, i+2)
        self.nin_t = self.nin_o/band

        self.res_f = np.array([0,
                               np.sum([118013952, 1605632, 802816, 802816]),
                               np.sum([115605504, 401408, 200704]),
                               np.sum([115605504, 401408]),
                               np.sum([115605504, 401408, 200704]),
                               115605504 + 401408,
                               57802752 + 200704 + 100352,
                               115605504 + 200704,
                               115605504 + 200704 + 100352,
                               115605504 + 200704,
                               57802752 + 100352 + 50176,
                               115605504 + 100352,
                               115605504 + 100352 + 50176,
                               115605504 + 100352,
                               57802752 + 50176 + 25088,
                               115605504 + 50176,
                               115605504 + 50176 + 25088,
                               115605504 + 50176,
                               512000]) / 1e10
        self.res_f_down = np.array([6422528 + 200704,
                                    6422528 + 100352,
                                    6422528 + 50176]) / 1e10
        # self.res_f_down_idx = [6, 10, 14]
        self.res_o = np.array([1.1484375, 1.53125, 1.53125, 1.53125, 1.53125, 1.53125,
                               0.765625, 1.53125, 0.765625, 0.765625, 0.3828125, 0.3828125,
                               0.3828125, 0.3828125, 0.19140625, 0.19140625, 0.19140625,
                               0.19140625,0])
        self.res_o_d = np.array([0.765625, 0.3828125, 0.19140625])
        res_l = 18+1+3
        self.res_g=Graph(res_l)
        for i in range(res_l-1):
            self.res_g.addEdge(i+1,i+2)
        ex_next = [9,13,17]
        ex_before = [6,10,14]
        for i in range(3):
            idx = 19+i+1
            self.res_g.addEdge(ex_before[i],idx)
            self.res_g.addEdge(idx,ex_next[i])
        pair = [[2,5],[4,7],[8,11],[12,15],[16,19]]
        for i in range(len(pair)):
            self.res_g.addEdge(pair[i][0],pair[i][1])
        s_result = self.res_g.topologicalSort()
        self.res_s=np.array(s_result)
        s_result=np.array(s_result)
        self.res_t=self.res_o/band
        self.res_t_d=self.res_o_d/band
        self.res_f_final = dict()
        self.res_t_final = dict()
        self.res_next = dict()
        self.res_o_final = dict()
        for i in range(len(s_result)):
            if s_result[i]<19:
                self.res_f_final[s_result[i]] = self.res_f[s_result[i]]
                self.res_t_final[s_result[i]] = self.res_t[s_result[i]]
                self.res_o_final[s_result[i]] = self.res_o[s_result[i]]
            else:
                self.res_f_final[s_result[i]] = self.res_f_down[s_result[i]-19]
                self.res_t_final[s_result[i]] = self.res_t_d[s_result[i]-19]
                self.res_o_final[s_result[i]] = self.res_o_d[s_result[i]-19]
            self.res_next[s_result[i]] = self.res_g.graph[s_result[i]]

        self.vgg_f = np.array([
            0,
            89915392 + 3211264,
            1852899328 + 3211264 + 3211264,
            926449664 + 1605632,
            1851293696 + 1605632 + 1605632,
            925646848 + 802816,
            1850490880 + 802816,
            1850490880 + 802816 + 802816,
            925245440 + 401408,
            1850089472 + 401408,
            1850089472 + 401408 + 401408,
            462522368 + 100352,
            462522368 + 100352,
            462522368 + 100352 + 100352,
            102760448.0 + 4096,
            16777216 + 4096,
            4096000
        ]) / 1e10
        self.vgg_o = np.array([
            3 * 224 * 224,
            64 * 224 * 224,
            64 * 112 * 112,
            128 * 112 * 112,
            128 * 56 * 56,
            256 * 56 * 56,
            256 * 56 * 56,
            256 * 28 * 28,
            512 * 28 * 28,
            512 * 28 * 28,
            512 * 14 * 14,
            512 * 14 * 14,
            512 * 14 * 14,
            512 * 7 * 7,
            4096,
            4096,
            0
        ]) / 1024 / 1024 * 8
        self.vgg_t=self.vgg_o/band
        self.vgg_next=[]
        self.vgg_s=list(range(len(self.vgg_t)))
        for i in range(len(self.vgg_t)-1):
            self.vgg_next.append(i+1)


        self.ale_f = np.array([
            0,
            70470400 + 193600 + 193600,
            224088768 + 139968 + 139968,
            112205184 + 64896,
            149563648 + 43264,
            99723520 + 43264 + 43264,
            37748736 + 4096,
            16777216 + 4096,
            4096000
        ]) / 1e10
        self.ale_o = np.array([
            3 * 224 * 224,
            64 * 27 * 27,
            192 * 13 * 13,
            384 * 13 * 13,
            256 * 13 * 13,
            256 * 6 * 6,
            4096,
            4096,
            0
        ]) / 1024 / 1024 * 8
        self.ale_t = self.ale_o/band
        self.ale_g=Graph(8*2)
        self.ale_g.addEdge(1,2)
        self.ale_g.addEdge(1,10)
        for i in range(2,8):
            self.ale_g.addEdge(i,i+1)
            self.ale_g.addEdge(i+8,i+1+8)
        self.ale_g.addEdge(8,9)
        self.ale_g.addEdge(16,9)
        self.ale_g.addEdge(3,12)
        self.ale_g.addEdge(11,4)
        self.ale_g.addEdge(6,15)
        self.ale_g.addEdge(14,7)
        self.ale_g.addEdge(7,16)
        self.ale_g.addEdge(15,8)
        s_result=self.ale_g.topologicalSort()
        self.ale_s=np.array(s_result)
        self.ale_t_final={}
        self.ale_f_final = {}
        self.ale_o_final={}
        for i in range(len(s_result)):
            if s_result[i]==0:
                self.ale_t_final[0]=self.ale_t[0]
                self.ale_f_final[0]=self.ale_f[0]
                self.ale_o_final[0]=self.ale_o[0]
            elif 0<s_result[i]<8:
                self.ale_t_final[s_result[i]]=self.ale_t[s_result[i]]/2
                self.ale_f_final[s_result[i]]=self.ale_f[s_result[i]]/2
                self.ale_o_final[s_result[i]]=self.ale_o[s_result[i]]/2
            elif 8<s_result[i]<16:
                self.ale_t_final[s_result[i]]=self.ale_t[s_result[i]-8]/2
                self.ale_f_final[s_result[i]]=self.ale_f[s_result[i]-8]/2
                self.ale_o_final[s_result[i]]=self.ale_o[s_result[i]-8]/2
            elif s_result[i]==8:
                self.ale_t_final[s_result[i]]=self.ale_t[s_result[i]]
                self.ale_f_final[s_result[i]]=self.ale_f[s_result[i]]
                self.ale_o_final[s_result[i]]=self.ale_o[s_result[i]]
        self.ale_next={}
        for i in range(len(s_result)):
            self.ale_next[s_result[i]]=self.ale_g.graph[s_result[i]]


class Off():
    def __init__(self):
        self.trans = 1024/8
        self.power_1 = 104.17
        self.power_2 = 104.17
        self.buf = 2
        self.tau = 100/1000
        self.tau_= 100/1000
        self.gs = GS()
        self.n = 1000

    def off_one(self, f_f,o_f,s_f):
        com_1=[]
        com_2=[]
        trans=[]
        l_f=len(s_f)
        for i in range(l_f):
            com_1.append(f_f[s_f[i]]/self.power_1)
            com_2.append(f_f[s_f[i]]/self.power_2)
            trans.append(o_f[s_f[i]]/self.trans)

        com_1_l=copy.deepcopy(com_1)
        com_2_r=copy.deepcopy(com_2)
        for i in range(1,l_f):
            com_1_l[i]+=com_1_l[i-1]
        for i in range(l_f-2,-1,-1):
            com_2_r[i]+=com_2_r[i+1]
        com_2_r=np.array(com_2_r)-np.array(com_2)

        com_2_l=copy.deepcopy(com_2)
        com_1_r=copy.deepcopy(com_1)
        for i in range(1,l_f):
            com_2_l[i]+=com_2_l[i-1]
        for i in range(l_f-2,-1,-1):
            com_1_r[i]+=com_1_r[i+1]
        com_1_r=np.array(com_1_r)-np.array(com_1)

        min_time = float("inf")
        cut_point = -1
        status = 'l'
        for i in range(l_f):
            now_time = np.max([com_1_l[i],com_2_r[i],trans[i]])
            if now_time<min_time:
                cut_point=i
                min_time=now_time
        for i in range(l_f):
            now_time = np.max([com_2_l[i],com_1_r[i],trans[i]])
            if now_time<min_time:
                cut_point=i
                min_time=now_time
                status='r'

        if status=='l':
            rec_1=trans[0]
            cal_1=com_1_l[cut_point]
            pub_1=trans[cut_point]
            rec_2=pub_1
            cal_2=com_2_r[cut_point]
            pub_2=0
        else:
            rec_1=trans[0]
            cal_1=com_2_l[cut_point]
            pub_1=trans[cut_point]
            rec_2=pub_1
            cal_2=com_1_r[cut_point]
            pub_2=0
        theo_n=np.ceil(self.tau/np.max([rec_1,rec_2,cal_1,cal_2,pub_1,pub_2]))
        recs=np.array([rec_1,rec_2])+1e-8
        cals=np.array([cal_1,cal_2])+1e-8
        pubs=np.array([pub_1,pub_2])+1e-8
        avg_time=-1
        for i in range(2):
            r_num=0
            c_num=0
            p_num=0
            rc_buf=0
            cp_buf=0
            total_time=0
            rcver=Receiver(self.tau,self.buf,recs[i])
            clter=Calculator(self.tau,self.buf,cals[i])
            puber=Publisher(self.tau,self.buf,pubs[i])
            while p_num<self.n:
                if r_num<self.n:
                    recvd_num, rc_buf, run_time=rcver.receive(rc_buf)
                    r_num+=recvd_num
                    total_time+=run_time
                if c_num<self.n:
                    cal_num, rc_buf, cp_buf, run_time=clter.calculate(rc_buf,cp_buf)
                    c_num+=cal_num
                    total_time+=run_time
                if p_num<self.n:
                    pub_num, cp_buf, run_time=puber.publish(cp_buf)
                    p_num+=pub_num
                    total_time+=run_time
                total_time+=self.tau_
            if total_time>avg_time:
                avg_time=total_time
        avg_time/=self.n
        return avg_time,theo_n, min_time

class Receiver():
    def __init__(self,tau,buf,rec_one):
        self.tau=tau
        self.now_rec=0
        self.buf = buf
        self.rec_one=rec_one

    def receive(self,rc_buf):
        rec_one=self.rec_one
        rest = self.buf-rc_buf
        if rest<1:
            return 0,rc_buf,0
        total = int((self.now_rec+self.tau)/rec_one)
        # recvd_num = 0
        if total<=rest:
            self.now_rec=self.now_rec+self.tau-total*rec_one
            recvd_num=total
            rc_buf+=total
            run_time=self.tau
        else:
            recvd_num=rest
            rc_buf=self.buf
            run_time=rest*rec_one-self.now_rec
            self.now_rec=0
        return recvd_num, rc_buf, run_time

class Calculator():
    def __init__(self,tau,buf,cal_one):
        self.tau=tau
        self.buf=buf
        self.now_cal=0
        self.cal_one=cal_one

    def calculate(self,rc_buf,cp_buf):
        cal_one=self.cal_one
        rest=min(rc_buf,self.buf-cp_buf)
        if rest<1:
            return 0,rc_buf,cp_buf,0
        total = int((self.now_cal+self.tau)/cal_one)
        if total==0:
            self.now_cal+=self.tau
            return 0,rc_buf,cp_buf,self.tau
        elif total<=rest:
            rc_buf-=total
            cp_buf+=total
            cal_num=total
            run_time=self.tau
            self.now_cal=self.tau-(total*cal_one-self.now_cal)
        else:
            cal_num=rest
            run_time=rest*cal_one-self.now_cal
            rc_buf-=rest
            cp_buf+=rest
            self.now_cal=0
        return cal_num,rc_buf,cp_buf,run_time

class Publisher():
    def __init__(self,tau,buf,pub_one):
        self.tau=tau
        self.buf=buf
        self.pub_one=pub_one
        self.now_pub=0

    def publish(self,cp_buf):
        pub_one=self.pub_one
        if cp_buf<1:
            return 0,cp_buf,0
        total=int((self.now_pub+self.tau)/pub_one)
        if total==0:
            self.now_pub+=self.tau
            return 0,cp_buf,0
        if total<=cp_buf:
            cp_buf-=total
            pub_num=total
            run_time=self.tau
            self.now_pub=self.tau-(total*pub_one-self.now_pub)
        else:
            pub_num=cp_buf
            run_time=pub_num*pub_one-self.now_pub
            cp_buf=0
            self.now_pub=0
        return pub_num,cp_buf,run_time


