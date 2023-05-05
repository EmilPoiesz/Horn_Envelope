from sympy import *
import functools
import random
import timeit

number_of_variables = 4

def define_variables(number):
    s = "".join(['v'+str(i)+',' for i in range(number)])
    V = [e for e in symbols(s)]
    return V

# V = define_variables(number_of_variables)



def generate_target(V, n_clauses,n_body_literals=-1):
    T = set()
    for i in range(n_clauses):
        if n_body_literals < 0:
            V_sample = random.sample(V, random.randint(1, len(V)-1))
        else:
            V_sample = random.sample(V, n_body_literals)
        clause = functools.reduce(lambda x,y: x & y, V_sample)
        V_implies = [item for item in V if item not in V_sample]
        x = random.randint(0, len(V)+1)
        target = ((clause) >> false if x == 0
                  else true >> (clause) if x == len(V)+1 and len(V_sample) == 1
                  else (clause) >> random.choice(V_implies))
        T.add(target)
    return T


def evaluate(formula, x, V):
    if formula == true:
        return True
    if formula == false:
        return False
    a = {V[i]: x[i] for i in range(len(V))}
    for i in range(len(V)):
        a[V[i]] = (True if x[i] == 1
                        else False)
    return True if formula.subs(a) == True else False

def set2theory(set):
    tempt = True
    for e in set:
        tempt = tempt & e
    return tempt

def entails(T,clause,V):
    '''
    Checks if T entails a horn clause c.
    Returns an assignment that falsifies c and satisfies T
     if the check is false.
    '''
    T1 = set2theory(T)
    assignment =satisfiable(T1 & ~clause)
    res =[0 for i in range(len(V))]
    if assignment != False :
        for e in assignment:
            idx = V.index(e)
            res[idx] =  1 if assignment[e] == True else 0
        return res
    return True


def EQ(H, V, target):
    for c in H:
        answer = entails(target,c,V)
        if  answer != True:
            #here a positive counterexample is returned
            return answer
    for c in target:
        answer = entails(H,c,V)
        if  answer != True:
            #here a negative counterexample is returned
            return answer
    return True

def MQ(assignment, V, target):
    t = set2theory(target)
    return evaluate(t,assignment,V)

def get_hypothesis(S, V,bad_nc,background):
    H = set()
    for a in [a for a in S if a not in bad_nc]:
        L = [V[index] for index,value in enumerate(a) if a[index] ==1 ] + [true]
        R = [V[index] for index,value in enumerate(a) if a[index] ==0 ] + [false]
        for r in R:
            clause = functools.reduce(lambda x,y: x & y, L)
            clause = (clause) >> r
            H.add(clause)
    H = H.union(background)
    return H

def get_body(clause):
    if type(clause.args[0]) == Symbol:
        return (clause.args[0],)
    else: return clause.args[0].args

def positive_check_and_prune(H,S,Pos,V,bad_nc):
    for pos in Pos:
        for clause in H.copy():
            if (evaluate(clause, pos, V) == False):
                H.remove(clause)
                # identify_problematic_nc(H,S,bad_nc)

    identify_problematic_nc(H,S,bad_nc,V)

    # for clause in [c for c in H.copy() if type(c) == Implies]:
    #     for c2 in [c for c in H.copy() if  type(c) == Not]:
    #         if set(get_body(clause)).issubset(set(get_body(clause))):
    #             H.discard(clause)

    return H

def checkduplicates(x,S,enabled):
    if x in S and enabled:
        # print('ah! I am trying to add {} to S but the negative counterexample {} is already present in S.'.format(x,x))
        raise Exception('I am trying to add {} to S but the negative counterexample {} is already present in S.\nIf you are learning from examples classified by a neural network, it means that it is not encoding a horn theory.'.format(x,x))

def identify_problematic_nc(H,S,bad_nc,V):
    h=set2theory(H)
    for a in [a for a in S if a not in bad_nc]:
        if (evaluate(h, a, V) == True):
            bad_nc.append(a)

def isgointobeduplicate(list,a,bad_nc):
    if a in list:
        bad_nc.append(a)
        return True
    else: return False

def learn(V,MQ,EQ, bad_nc,bad_pc,background= {}, verbose = False,iterations=-1,guard=False):
    terminated = False
    metadata = []
    average_samples = 0
    eq_done = 0
    H = set()
    H = H.union(background)
    S = []
    #remember positive counterexamples
    Pos = []
    #list of negative counterexamples that cannot produce a rule (according to positive counterexamples)
    #bad_nc =[]
    while True and iterations!=0:
        data = {}
        start = timeit.default_timer()
        #Ask for H
        eq_res = EQ(H)
        if type(eq_res) == bool and eq_res:
            terminated = True
            stop = timeit.default_timer()
            data['runtime'] = stop-start
            metadata.append(data)
            with open('output.txt', 'a') as f:
                f.write("=== TERMINATED ===\n")
            break
        else:
            (x,i) = eq_res
        data['sample'] = i
        pos_ex=False
        if verbose ==2:
            print('Iteration {}\n\nhypothesis is {}.\n\ncounterexample: {}\n\nS: {}\nbad_nc is{}\n\nbad_pc is{}\n\n\n\n\n'.format(abs(iterations),H,x,S,bad_nc,bad_pc))
            # input()
        elif verbose == 1:
            print('Iteration {}     '.format(abs(iterations)),end='\r')
        #If x is pos ce
        for clause in H.copy():
            if (evaluate(clause, x,V) == False):
                if clause in background:
                    bad_pc.append(x)
                else:
                    H.remove(clause)
                    Pos.append(x)
                    #check if a nc in S does not falsify a clause in H
                    identify_problematic_nc(H,S,bad_nc,V)
                    pos_ex = True
        if not pos_ex:
            #If x is neg ce
            replaced = False
            for s in S:
                s_intersection_x = [1 if s[index] ==1 and x[index] == 1 else 0 for index in range(len(V))]
                A = {index for index,value in enumerate(s_intersection_x) if value ==1}
                B = {index for index,value in enumerate(s) if value ==1}
                if A.issubset(B) and not B.issubset(A): # A properly contained in B
                    idx = S.index(s)
                    if MQ(s_intersection_x) == False and s_intersection_x not in bad_nc:
                        checkduplicates(s_intersection_x,S,guard)
                        if not isgointobeduplicate(S,s_intersection_x,bad_nc):
                            S[idx] = s_intersection_x
                            replaced = True
                        break
            if not replaced:
                checkduplicates(x,S,guard)
                S.append(x)

            H = get_hypothesis(S,V,bad_nc,background)
            #small optimisation. Refine hypo. with known positive counterexamples.
            H = positive_check_and_prune(H,S,Pos,V,bad_nc)
        iterations-=1
        stop = timeit.default_timer()
        data['runtime'] = stop-start
        metadata.append(data)
        if iterations % 5 == 0:
            sentence = "iteration = {eq}\tlen(H) = {h}\truntime = {rt}\n".format(eq = 5000 - iterations, h=len(H), rt = data['runtime'])
            with open('output.txt', 'a') as f:
                f.write(sentence)
    return (terminated, metadata, H)

#this is the target
# T = {((V[0] & V[1]) >> V[2]), (V[0] & V[3]) >> False}
# A difficult target
# T = {((V[0] & V[1]) >> V[2]), (V[0] & V[3]) >> False, V[1]}
#
# mq = lambda a : MQ(a,V, T)
# eq = lambda a : EQ(a, V, T)
# print('hypothesis found!\n',learn(V,mq,eq))


# v = define_variables(4)
# r = positive_check_and_prune({(v[0] >> v[1]), (~v[0])},[],v,[])
# print(r)
