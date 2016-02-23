import numpy as np
from numpy.random import randint
from time import time

class svmlitepy:
    def __init__(self,C=0.05,tol=0.001,kernel='rbf',coef=0.0 , gamma='auto',degree=3 , verbose=0):
        self.N=-1
        self.d=-1
        self.C=C
        self.tol=tol
        self.eps=0.001
        self.coef=coef
        self.gamma=gamma
        self.degree=degree
        self.verbose=verbose
        self.b=0
        self.TrainX=None
        self.TrainY=None
        self.alpha=None
        self.error_cache=None
        self.precomputed_self_dot_product=None
        if kernel=='rbf':
            self.kernel_func=self.rbf_kernel
        elif kernel=='linear':
            self.kernel_func=self.linear_kernel
        elif kernel=='poly':
            self.kernel_func=self.poly_kernel
        elif kernel=='sigmoid':
            self.kernel_func=self.sigmoid_kernel
        else:
            raise ValueError("Invalid Kernel Type")

    def fit(self,TrainX,TrainY):
        self.TrainX=TrainX
        self.TrainY=TrainY
        self.N=self.TrainX.shape[0]
        self.d=self.TrainX.shape[1]
        if self.gamma=='auto':
            self.gamma=1.0/self.d
        self.alpha=np.zeros(self.N)
        self.error_cache=np.zeros(self.N)
        self.precomputed_self_dot_product=np.zeros(self.N)
        for i in range(0,self.N):
            self.precomputed_self_dot_product[i]=np.dot(self.TrainX[i,:],self.TrainX[i,:])
        ExamineAll=True
        NumChanged=0
        loop=1
        while(NumChanged>0 or ExamineAll):
            t0=time()
            NumChanged=0
            if(ExamineAll):
                for i in range(0,self.N):
                    NumChanged+=self.examineExample(i)
            else:
                for i in range(0,self.N):
                    alpha=self.alpha[i]
                    if alpha != 0 and alpha != self.C:
                        NumChanged+=self.examineExample(i)
            if ExamineAll:
                ExamineAll=False
            elif NumChanged == 0:
                ExamineAll=True
            if self.verbose==1:
                alpha0=0
                alphaC=0
                alphamid=0
                for i in range(0,self.N):
                    alphatemp=self.alpha[i]
                    if alphatemp==0:
                        alpha0+=1
                    elif alphatemp==self.C:
                        alphaC+=1
                    else:
                        alphamid+=1
                print "Took %0.3f secs for loop %d with number of lagrange multipliers for alpha=0, 0< alpha < C and alpha=C being %d %d %d" % ((time()-t0),loop,alpha0,alphamid,alphaC)
            loop+=1


    def examineExample(self,index):
        Y1=self.TrainY[index]
        alpha1=self.alpha[index]
        E1=0
        if alpha1>0 and alpha1<self.C:
            E1=self.error_cache[index]
        else:
            E1=self.learned_func(index)-Y1

        R1=E1*Y1
        if (R1 < -self.tol and alpha1 < self.C) or ( R1> self.tol and alpha1 >0):
            pos=-1
            tmax=0
            for k in range(0,self.N):
                if self.alpha[k]>0 and self.alpha[k]<self.C:
                    E2=self.error_cache[k]
                    temp=np.fabs(E1-E2)
                    if temp>tmax:
                        tmax=temp
                        pos=k

            if pos>=0:
                if self.takeStep(index,pos)==1:
                    return 1
            temp=np.random
            temp=randint(0,self.N)
            for k in range(temp,temp+self.N):
                pos=k%self.N
                if self.alpha[pos]>0 and self.alpha[pos]<self.C:
                    if self.takeStep(index,pos)==1:
                        return 1

            temp=randint(0,self.N)
            for k in range(temp,temp+self.N):
                pos=k%self.N
                if self.takeStep(index,pos)==1:
                    return 1

        return 0

    def takeStep(self,i1,i2):
        if i1==i2:
            return 0
        alpha1=self.alpha[i1]
        alpha2=self.alpha[i2]
        Y1=self.TrainY[i1]
        Y2=self.TrainY[i2]
        E1=0.0
        E2=0.0
        L=0.0
        H=0.0
        a1=0.0
        a2=0.0
        bnew=0.0
        if alpha1>0 and alpha1<self.C:
            E1=self.error_cache[i1]
        else:
            E1=self.learned_func(i1)-Y1

        if alpha2>0 and alpha2<self.C:
            E2=self.error_cache[i2]
        else:
            E2=self.learned_func(i2)-Y2

        S=Y1*Y2
        if S==1:
            Gamma=alpha1+alpha2
            if Gamma>self.C:
                L=Gamma-self.C
                H=self.C
            else:
                L=0
                H=Gamma
        else:
            Gamma=alpha1-alpha2
            if Gamma>0:
                L=0
                H=self.C-Gamma
            else:
                L=-Gamma
                H=self.C

        if L==H:
            return 0

        k11=self.kernel_func(i1,i1)
        k12=self.kernel_func(i1,i2)
        k22=self.kernel_func(i2,i2)
        eta=2*k12-k11-k22
        if eta<0:
            a2=alpha2+Y2*(E2-E1)/eta
            if a2<L:
                a2=L
            elif a2>H:
                a2=H
        else:
            c1=eta/2
            c2=Y2*(E1-E2)-eta*alpha2
            Lobj=c1*L*L+c2*L
            Hobj=c1*H*H+c2*H
            if Lobj > (Hobj+self.eps):
                a2=L
            elif Lobj<Hobj-self.eps :
                a2=H
            else:
                a2=alpha2

        if np.fabs(a2-alpha2) < self.eps*(a2+alpha2+self.eps):
            return 0

        a1=alpha1-S*(a2-alpha2)
        if a1<0:
            a2+=S*a1
            a1=0
        elif a1>self.C:
            a2+=S*(a1-self.C)
            a1=self.C

        if a1>0 and a1<self.C:
            bnew=self.b+E1+Y1*(a1-alpha1)*k11+Y2*(a2-alpha2)*k12
        else:
            if a2>0 and a2<self.C:
                bnew=self.b+E2+Y1*(a1-alpha1)*k12+Y2*(a2-alpha2)*k22
            else:
                b1=self.b+E1+Y1*(a1-alpha1)*k11+Y2*(a2-alpha2)*k12
                b2=self.b+E2+Y1*(a1-alpha1)*k12+Y2*(a2-alpha2)*k22
                bnew=(b1+b2)/2

        delta_b=bnew-self.b
        b=bnew
        t1=Y1*(a1-alpha1)
        t2=Y2*(a2-alpha2)

        for i in range(0,self.N):
            if self.alpha[i]>0 and self.alpha[i]<self.C:
                self.error_cache[i]=self.error_cache[i]+(t1*self.kernel_func(i1,i)+t2*self.kernel_func(i2,i) -delta_b)

        self.error_cache[i1]=0.0
        self.error_cache[i2]=0.0
        self.alpha[i1]=a1
        self.alpha[i2]=a2
        return 1

    def learned_func(self,index):
        S=0.0
        for i in range(0,self.N):
            S+=self.alpha[i]*self.TrainY[i]*self.kernel_func(i,index)
        S=S-self.b
        return S

    def linear_kernel(self,index1,index2):
        return np.dot(self.TrainX[index1,:],self.TrainX[index2,:])

    def rbf_kernel(self,index1,index2):
        S=self.precomputed_self_dot_product[index1]+self.precomputed_self_dot_product[index2]-2*np.dot(self.TrainX[index1,:],self.TrainX[index2,:])
        S=S*-1.0
        return np.exp(S*self.gamma)


    def poly_kernel(self,index1,index2):
        return  pow(self.gamma*np.dot(self.TrainX[index1,:],self.TrainX[index2,:])+self.coef,self.degree)

    def sigmoid_kernel(self,index1,index2):
        return np.arctanh(self.gamma*np.dot(self.TrainX[index1,:],self.TrainX[index2,:])-self.coef)
