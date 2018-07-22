import numpy as np
from math import log
import matplotlib.pyplot as plt
#load data
Traindata=np.loadtxt(r"C:\Users\Administrator\Desktop\MNIST\decisiontree\Iris.txt",delimiter=',',usecols=(0,1,2,3),dtype=float)
Trainlabel=np.loadtxt(r"C:\Users\Administrator\Desktop\MNIST\decisiontree\Iris.txt",delimiter=',',usecols=(4),dtype=str)
feaname=["#0","#1","#2","#3"]
feanamecopy=["#0","#1","#2","#3"]
average=np.mean(Traindata,axis=0).tolist()#get the average
averagecopyy=np.mean(Traindata,axis=0).tolist()
#print(average,averagecopyy)
#calculate entropy
def calentropy(Trainlabel):
    n=len(Trainlabel) #the number of sambles
    count={} #creat dictionary "count"
    for vec in Trainlabel:
        if vec not in count.keys():
            count[vec]=0  
        count[vec]+=1 #统计有多少个类及每个类的数量
    entropy=0
    for key in count:
        pxi=float(count[key])/n #notice transfering to float first
        entropy-=pxi*log(pxi,2)
    return entropy 
#按照某个特征分类后的数据
def splitTraindata(Traindata,axis,average):
    index_less=[]
    index_greater=[]
    n=len(Traindata)
    for index in range(n):
        d=Traindata[index]
        if np.all(d[axis]<average[axis]):
            #print(type(d[axis]))
            index_less.append(index)
        else:
            index_greater.append(index)
    return index_less,index_greater
def index2data(Traindata,Trainlabel,split_index,axis):
    indexl=split_index[0]
    indexg=split_index[1]
    datal=[]
    datag=[]
    labell=[]
    labelg=[]
    for i in indexl:
            reduceddatal=np.append(Traindata[i][:axis],Traindata[i][axis+1:])
            datal.append(reduceddatal)
    for i in indexg:
            educeddatag=np.append(Traindata[i][:axis],Traindata[i][axis+1:])
            datag.append(reduceddatal)
    labell=Trainlabel[indexl]
    labelg=Trainlabel[indexg]
    return datal,datag,labell,labelg
#根据最大信息增益选择最佳feature
def choosebest_splitnode(Traindata,Trainlabel,average):
    #print(Traindata[0])
    n_feature=len(Traindata[0])#feature的个数
    n_label=len(Trainlabel)#多少组数据
    base_entropy=calentropy(Trainlabel)#计算熵
    best_gain=-1
    for feature_i in range(n_feature):#calculate entropy under each splitting feature
        cur_entropy=0
        indexset_less,indexset_greater=splitTraindata(Traindata,feature_i,average)
        #print(indexset_less,indexset_greater,len(indexset_greater),len(indexset_less))
        prob_less=float(len(indexset_less))/n_label
        prob_greater=float(len(indexset_greater))/n_label
        cur_entropy+=prob_less*calentropy(Trainlabel[indexset_less])
        cur_entropy+=prob_greater*calentropy(Trainlabel[indexset_greater])
        info_gain=base_entropy-cur_entropy
        if (info_gain>best_gain):
            best_gain=info_gain
            best_index=feature_i
        return best_index
#递归构建决策树
def buildtree(Traindata,Trainlabel,feaname,average):
    #空数据
    if Trainlabel.size==0:
        return "NULL"
    listlabel=Trainlabel.tolist()
    #stop when all samples in this subset belongs to one class
    #只有一种分类标签
    if listlabel.count(Trainlabel[0])==Trainlabel.size:
        return Trainlabel[0]
    #return the majority of samples' label in this subset if no extra features available
    #特征名列表为空时时
    #按类别数量作判定
    if len(feaname)==0:
        labelcount={}
        for vote in Trainlabel:
            if vote not in labelcount.keys():
                labelcount[vote]=0
            labelcount[vote]+=1
        maxx=-1
        for key in labelcount.keys():
            if maxx<labelcount[key]:
                maxx=labelcount[key]
                maxlabel=key
        return maxlabel
    bestsplit_feature=choosebest_splitnode(Traindata,Trainlabel,average)
    n_len=len(Traindata[0])
    #print(bestsplit_feature,n_len,type(Traindata))
    cur_feaname=feaname[bestsplit_feature]
    #print(cur_feaname)
    node_fea={cur_feaname:{}}
    del(feaname[bestsplit_feature])
    #print(len(feaname))
    split_index=splitTraindata(Traindata,bestsplit_feature,average)
    #print(split_index[0],split_index[1])
    data_less,data_greater,label_less,label_greater=index2data(Traindata,Trainlabel,split_index,bestsplit_feature)
    #print(data_less,data_greater,len(data_less),len(data_greater),label_less[0],label_greater[0])
    del(average[bestsplit_feature])
    #print(feaname)
    #print(cur_feaname)
    node_fea[cur_feaname]["<"]=buildtree(data_less,label_less,feaname,average)
    node_fea[cur_feaname][">"]=buildtree(data_greater,label_greater,feaname,average)
    return node_fea
#样本分类 classify a new sample
def classify(mytree,testdata,feaname,average):
    cc={}
    if type(mytree)!= type(cc):
        return mytree
    fea_name=list(mytree.keys())
    #print(fea_name,feaname)
    fea_index=feaname.index(fea_name[0])
    #print(fea_index,average)
    value=testdata[fea_index] 
    nextbranch=mytree[fea_name[0]]
    if value<average[fea_index]:
        nextbranch=nextbranch["<"]
    else:
        nextbranch=nextbranch[">"]
    return classify(nextbranch,testdata,feaname,average )

#mytree可视化
def plotNode(ax1,Nodename,centerpt,parentpt,nodeType,arrow_args):
    ax1.annotate(Nodename,xy=parentpt,xycoords="axes fraction",xytext
                            =centerpt,textcoords="axes fraction",va="center",ha=
                            "center",bbox=nodeType,arrowprops=arrow_args)
def plotmidtext(ax1,centerpt,parentpt,textname):
    xmid=(parentpt[0]-centerpt[0])/2+centerpt[0]
    ymid=(parentpt[1]-centerpt[1])/2+centerpt[1]
    ax1.text(xmid,ymid,textname,va="center",ha="center",rotation=30)
def getNumleafs(mytree):
    Numleafs=0
    firstNode=list(mytree.keys())[0]
    seconddict=mytree[firstNode]
    for key in seconddict.keys():
        if type(seconddict[key]) is dict:
            Numleafs+=getNumleafs(seconddict[key])
        else:
            Numleafs+=1
    return Numleafs
def getTreedepth(mytree):
    maxdepth=0
    firstNode=list(mytree.keys())[0]
    seconddict=mytree[firstNode]
    for key in seconddict.keys():
        if type(seconddict[key]) is dict:
            thisdepth=1+getTreedepth(seconddict[key])
        else:
            thisdepth=1
        if thisdepth>maxdepth:
            maxdepth=thisdepth
    return maxdepth
def plotTree(ax1,totalW,totalD,xoff,yoff,mytree,parentpt,textname,decisionNode,leafNode,arrow_args):
    Numleafs=getNumleafs(mytree)
    depth=getTreedepth(mytree)
    firstNode=list(mytree.keys())[0]
    centerpt=(xoff+(1.0+float(Numleafs))/(2*totalW),yoff)
    plotmidtext(ax1,centerpt,parentpt,textname)
    plotNode(ax1,firstNode,centerpt,parentpt,decisionNode,arrow_args)
    seconddict=mytree[firstNode]
    yoff=yoff-1.0/totalD
    for key in seconddict.keys():
              if type(seconddict[key]) is dict:
                  plotTree(ax1,totalW,totalD,xoff,yoff,seconddict[key],parentpt,str(key),decisionNode,leafNode,arrow_args)
              else:  
                  xoff=xoff+1/totalW
                  plotNode(ax1,seconddict[key],(xoff,yoff),centerpt,leafNode,arrow_args)
                  plotmidtext(ax1,(xoff,yoff),centerpt,str(key))
    #yoff =yoff + 1.0/totalD
              
def createPlot(mytree,decisionNode,leafNode,arrow_args):
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    #axprops = dict(xticks=[], yticks=[])
    ax1 = plt.subplot(111, frameon=False)    #no ticks
    totalW = float(getNumleafs(mytree))
    totalD = float(getTreedepth(mytree))
    xoff = -0.5/totalW
    yoff = 1.0
    plotTree(ax1,totalW,totalD,xoff,yoff,mytree, (0.5,1.0), '',decisionNode,leafNode,arrow_args)
    plt.show()
              
    
    
    

decisionNode=dict(boxstyle="sawtooth",fc="0.8")
leafNode=dict(boxstyle="round4",fc="0.8")
arrow_args=dict(arrowstyle="<-")
mytree=buildtree(Traindata,Trainlabel,feaname,average)
ss=list(mytree.keys())[0]
print(mytree,ss)
testdata=[0,0,10,0]
#print(feanamecopy,averagecopyy)
xx=classify(mytree ,testdata,feanamecopy,averagecopyy)
#print(testdata,mytree,xx)
print(xx,Traindata[0])
createPlot(mytree,decisionNode,leafNode,arrow_args)

