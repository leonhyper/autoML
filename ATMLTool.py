import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import scale
from sklearn.preprocessing import OneHotEncoder
from sklearn.neighbors import KNeighborsClassifier
from scipy.spatial.distance import pdist, squareform
from numba import jit

def readData(path):
    ds=pd.read_csv(path,engine='python')
    return ds

def judge(ds):
    print(ds.dtypes)

def clean(train,test,target,textcolumn=[]):
    clist= train.columns.values.tolist()
    droplist=[]
    for name in clist: 
        if(name!=target): 
            # 将空缺值数量大于总体80%的特征列删除
            if(train[name].isnull().sum()/len(train)>0.8):
                train.drop([name],axis=1,inplace=True)
                droplist.append(name)
            else:
                # 将连续型特征的空缺值用均值填补
                if(train[name].dtype==float):
                    train[name].fillna(np.mean(train[name]),inplace = True)
                    test[name].fillna(np.mean(test[name]),inplace = True)

    train.dropna(inplace=True)

    for name in clist:
        if(name!=target):
            if name not in textcolumn:
                # 清理离散型数据
                if(train[name].dtype==object or train[name].dtype==np.int64):
                    # 删除离散唯一特征列和最大类别成员数量大于总体80%的列
                    if(max(train[name].value_counts().values)/len(train)>0.8):
                        train.drop([name],axis=1,inplace=True)
                        droplist.append(name)
    
    #对测试集进行相同的删除操作
    for name in droplist:
        test.drop([name],axis=1,inplace=True)
    test.dropna(inplace=True)
    
#以一维list的格式提取特诊矩阵中的列
def columnXtrac1(X,n=0):
    c=[]
    for i in X:
        c.append(i[n])
    return c

#以二维list的格式提取特诊矩阵中的列
def columnXtrac2(X,n=0):
    c=[]
    for i in X:
        c.append([i[n]])
    return c

def matrixing(train,test,target,textcolumn=[]):
    clist= train.columns.values.tolist()
    t=pd.concat([train,test])
    # 处理目标列，类别转化为数字类型
    if(train[target].dtype==object):
        lecoder=LabelEncoder()
        lecoder.fit(t[target])
        train[target]=lecoder.transform(train[target])
        test[target]=lecoder.transform(test[target])
    YTR=train[[target]].values
    YTE=test[[target]].values

    
    XTR=train[[clist[0]]].values
    XTE=test[[clist[0]]].values
    
    for name in clist: 
        if name not in textcolumn:
            if(name!=target):
                # 将连续型数据归一化
                if(train[name].dtype==np.float64):
                    train[name]=scale(train[name])
                    test[name]=scale(test[name])
                    XTR=np.append(XTR,train[[name]].values,axis=1)
                    XTE=np.append(XTE,test[[name]].values,axis=1)

                # 将离散型数据用独热编码处理
                if(train[name].dtype==np.int64 or train[name].dtype==object ):
                    lecoder=LabelEncoder()
                    t[name]=lecoder.fit_transform(t[name])
                    train[name]=lecoder.transform(train[name])
                    test[name]=lecoder.transform(test[name])
                    XTR=np.append(XTR,train[[name]].values,axis=1)
                    XTE=np.append(XTE,test[[name]].values,axis=1)
                  
    XTR=np.delete(XTR,0,axis=1)
    XTE=np.delete(XTE,0,axis=1)
    return XTR,YTR,XTE,YTE



# 计算特征列与目标值的距离相关性
@jit
def distcorr(X, Y):
    X = np.atleast_1d(X)
    Y = np.atleast_1d(Y)
    if np.prod(X.shape) == len(X):
        X = X[:, None]
    if np.prod(Y.shape) == len(Y):
        Y = Y[:, None]
    X = np.atleast_2d(X)
    Y = np.atleast_2d(Y)
    n = X.shape[0]
    if Y.shape[0] != X.shape[0]:
        raise ValueError('Number of samples must match')
    a = squareform(pdist(X))
    b = squareform(pdist(Y))
    A = a - a.mean(axis=0)[None, :] - a.mean(axis=1)[:, None] + a.mean()
    B = b - b.mean(axis=0)[None, :] - b.mean(axis=1)[:, None] + b.mean()
    
    dcov2_xy = (A * B).sum()/float(n * n)
    dcov2_xx = (A * A).sum()/float(n * n)
    dcov2_yy = (B * B).sum()/float(n * n)
    dcor = np.sqrt(dcov2_xy)/np.sqrt(np.sqrt(dcov2_xx) * np.sqrt(dcov2_yy))
    return dcor

# 将数组缩放到0-1范围内
def rescale_linear(array):
    minimum, maximum = np.min(array), np.max(array)
    m = 1 / (maximum - minimum)
    b = 0 - m * minimum
    return m *np.asarray(array) + b

# 使用平均特征组合进行特征选择
def ACSelection(XTR,YTR,XTE,YTE):
    distdic={}
    for i in range(len(XTR[0])):
        distdic[i]=distcorr(columnXtrac1(XTR,i),columnXtrac1(YTR))

    d_order=sorted(distdic.items(),key=lambda x:x[1],reverse=True)
    k=[]
    for t in d_order:
        k.append(t[0])

    XTemp=columnXtrac2(XTR,k[0])
    XTTemp=columnXtrac2(XTE,k[0])
    scoredic={}

    for i in range(len(k)):
        if(i==0):
            # 用knn测试正确率,正确率对应循环次数存入字典
            # {2:0.8}表示第三次循环中加入了相关性排名第三的特征列，此时正确率为0.8
            KNN=KNeighborsClassifier(n_neighbors=3,weights='uniform')
            KNN.fit(XTemp,YTR.ravel())
            scoredic[i]=KNN.score(XTTemp,YTE.ravel())
        else:
            # 用knn测试正确率,正确率对应循环次数存入字典
            XTemp=np.append(XTemp,columnXtrac2(XTR,k[i]),axis=1)
            XTTemp=np.append(XTTemp,columnXtrac2(XTE,k[i]),axis=1)
            KNN=KNeighborsClassifier(n_neighbors=3,weights='uniform')
            KNN.fit(XTemp,YTR.ravel())
            scoredic[i]=KNN.score(XTTemp,YTE.ravel())
            
    print(scoredic)
    s_order=sorted(scoredic.items(),key=lambda x:x[1],reverse=True)
    XTRS=columnXtrac2(XTR,k[0])
    XTES=columnXtrac2(XTE,k[0])
    k2=[]
    for t in s_order:
        k2.append(t[0])
    
    if(k2[0]==0):
        return XTRS,XTES
    else:
        for i in range(k2[0]+1):
            if(i!=0):
                XTRS=np.append(XTRS,columnXtrac2(XTR,k[i]),axis=1)
                XTES=np.append(XTES,columnXtrac2(XTE,k[i]),axis=1)
        return XTRS,XTES

#变异
def mutate(pop,rate):
    newpop=[]
    for p in pop:
        temp=list(p)
        for i in range(len(temp)):
            if(rate[i]>=random.random()):
                temp[i]='1'
            else:
                temp[i]='0'
        newpop.append(''.join(temp))
    return newpop

# 交叉
def cross(ma,fa):
    c=[]
    m=list(ma)
    f=list(fa)
    for i in range(len(m)):
        if(random.random()<=0.5):
            c.append(m[i])
        else:
            c.append(f[i])
    return ''.join(c)

# 适应性函数
# 以KNN的预测正确率作为适应度
def evafit(s,XTR,YTR,XTE,YTE):
    l=list(s)
    if '1' not in l:
        return 0
    else:
        X=columnXtrac2(XTR)
        XR=columnXtrac2(XTE)
        for i in range(len(l)):
            if(l[i]=='1'):
                X=np.append(X,columnXtrac2(XTR,i),axis=1)
                XR=np.append(XR,columnXtrac2(XTE,i),axis=1)
        X=np.delete(X,0,axis=1)
        XR=np.delete(XR,0,axis=1)
        KNN=KNeighborsClassifier(n_neighbors=3,weights='uniform')
        KNN.fit(X,YTR.ravel())
        return KNN.score(XR,YTE.ravel())

# 对整个种群测定适应度，返回适应度列表
def fitpop(pop,XTR,YTR,XTE,YTE):
    l=[]
    for p in pop:
        l.append(evafit(p,XTR,YTR,XTE,YTE))
    return l

# 按照概率抽取样本
def rand_pick(seq , probabilities):
    x = np.random.uniform(0 ,1)
    cumprob = 0.0
    for item , item_pro in zip(seq , probabilities):
        cumprob += item_pro
        if x < cumprob:
            break
    return item

# 筛选列表中最大的前十个数的index
def topN(l,n):
    nums=l
    temp=[]
    Inf = 0
    for i in range(n):
        temp.append(nums.index(max(nums)))
        nums[nums.index(max(nums))]=Inf
    temp.sort()
    return temp

# 根据基因字符串提取矩阵
def exract(X,S):
    l=list(S)
    XR=columnXtrac2(X)
    for i in range(len(l)):
        if(l[i]=='1'):
            XR=np.append(XR,columnXtrac2(X,i),axis=1)
    XR=np.delete(XR,0,axis=1)
    return XR    

# 使用进化算法进行特征选择
def GASelection(XTR,YTR,XTE,YTE):
    # 初始化突变率列表
    mutrate=[]
    for i in range(len(XTR[0])):
        mutrate.append(distcorr(columnXtrac1(XTR,i),columnXtrac1(YTR)))
    mutrate=rescale_linear(mutrate)
    # 初始化第一代种群
    pop=[]
    fitnesslist=[]
    while(len(pop)<10):
        s=[]
        for i in range(len(XTR[0])):
            s.append('0')
        str=''.join(s)
        pop.append(str)
    pop=mutate(pop,mutrate)
    fitnesslist=fitpop(pop,XTR,YTR,XTE,YTE)

    # return(0)

    # 计数器，记录迭代次数，最高适应度，最高适应度个体和终止条件
    loops=1
    topfit=max(fitnesslist)
    topindi=pop[fitnesslist.index(topfit)]
    endnote=0
    # 子代种群
    newpop=[]
    # 子代适应度列表
    newfitlist=[]

    # 进化算法
    while(endnote<=10):
        # 轮盘赌抽样。计算抽样概率列表
        prop=[]
        sumup=np.sum(fitnesslist)
        for i in fitnesslist:
            prop.append(i/(sumup+0.000001))

        while(len(newpop)<10):
            mother=''
            father=''
            # 抽取亲本个体
            mother=rand_pick(pop,prop)
            father=rand_pick(pop,prop)
            # while(mother==father):
            #     father=rand_pick(pop,prop)
            # 交叉
            newpop.append(cross(mother,father))
        
    
        # 对子代种群进行群体变异
        newpop=mutate(newpop,mutrate)
        # 计算子代适应度
        newfitlist=fitpop(newpop,XTR,YTR,XTE,YTE)

        # 合并亲代子代选出适应度前10的个体作为新的亲代
        pop=pop+newpop
        tempfit=fitnesslist+newfitlist
        temp=topN(fitnesslist,10)

        newpop=[]
        newfitlist=[]
        for i in temp:
            newpop.append(pop[i])
            newfitlist.append(tempfit[i])
        pop=newpop
        fitnesslist=newfitlist

        # 更新计数器
        loops+=1
        if(max(fitnesslist)>topfit):
            topfit=max(fitnesslist)
            topindi=pop[fitnesslist.index(topfit)]
            endnote=0
        else:
            endnote+=1
    
    # 输出最高适应度和对应个体以及迭代次数
    print(loops,topindi,topfit)
    # 根据最高适应度个体基因提取相应特征列
    XTRS=exract(XTR,topindi)
    XTES=exract(XTE,topindi)
    return XTRS,XTES



            


