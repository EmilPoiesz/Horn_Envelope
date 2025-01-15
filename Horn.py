from sympy import *
import functools
import random
import timeit

def define_variables(number):
    """
    Generate a list of symbolic variables.

    This function creates a list of symbolic variables named 'v0', 'v1', ..., 'v(number-1)'.

    Parameters:
    number (int): The number of symbolic variables to generate.

    Returns:
    list: A list of symbolic variables.
    """
    s = "".join(['v'+str(i)+',' for i in range(number)])
    return list(symbols(s))

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
    # What does it mean here to say that a formula is true or false?
    # Related to free symbols?
    if formula == true:
        return True
    if formula == false:
        return False
    a = {V[i]: x[i] for i in range(len(V))}
    for i in range(len(V)):
        a[V[i]] = (True if x[i] == 1
                        else False)
    return True if formula.subs(a) == True else False

def from_set_to_theory(set):
    """
    Converts a set of boolean values to a single boolean value using logical AND.

    Args:
        set (iterable): An iterable of boolean values.

    Returns:
        bool: The result of performing a logical AND operation on all elements in the set.
    """
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
    T1 = from_set_to_theory(T)
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
    t = from_set_to_theory(target)
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
    # Is nc -> Non-norn Clause?

    # Current hypothesis
    h=from_set_to_theory(H)
    # List of negative? counterexamples that cannot produce a rule (according to positive counterexamples)?
    for a in [a for a in S if a not in bad_nc]: # Why bad? What does bad mean here?
        # If they are true, then are they negative counterexamples?
        # This feels like something that is correlated to NON-HORN section of the algorithm?
        if (evaluate(h, a, V) == True):
            bad_nc.append(a)

def isgointobeduplicate(list,a,bad_nc):
    if a in list:
        bad_nc.append(a)
        return True
    else: return False


def learn(V, ask_membership_oracle, ask_equivalence_oracle, bad_nc, bad_pc, background= {}, verbose = False,iterations=-1,guard=False):
    
    # V is the set of all variables
    # ask_membership_oracle is the membership query function
    # ask_equivalence_oracle is the equivalence query function
    # background is the background knowledge

    # H is the hypothesis
    # S is the set of examples
    # Pos is the set of positive counterexamples

    # eq_res is the result of the equivalence query, that is a counterexample

    terminated = False
    metadata = []
    average_samples = 0
    eq_done = 0
    HYPOTHESIS = set()
    HYPOTHESIS = HYPOTHESIS.union(background)
    # is S in this case negative counterexamples?
    S = []
    #remember positive counterexamples
    Pos = []
    #list of negative counterexamples that cannot produce a rule (according to positive counterexamples)
    #bad_nc =[]
    while True and iterations!=0:
        data = {}
        start = timeit.default_timer()
        
        # Ask for a counterexample with respect to the current hypothesis
        # The counter example is using PAC learning. Asking for a set amount of samples, 
        # and if no sample is found to contradict the hypothesis, then the hypothesis is considered correct.
        counter_example = ask_equivalence_oracle(HYPOTHESIS)

        # If the result is True, then the hypothesis is correct
        if type(counter_example) == bool and counter_example:
            terminated = True
            stop = timeit.default_timer()
            data['runtime'] = stop-start
            metadata.append(data)
            with open('output.txt', 'a') as f:
                f.write("=== TERMINATED ===\n")
            break
        
        (counter_example_assignment,sample_number) = counter_example
        data['sample'] = sample_number
        pos_ex=False
        
        if verbose ==2:
            print(f'\nIteration: {abs(iterations)}\n\n' + 
                  f'Hypothesis: {sorted([str(h) for h in HYPOTHESIS])}.\n\n' +
                  f'Hypothesis length: {len(HYPOTHESIS)}.\n\n' +
                  f'Counterexample: {counter_example_assignment}\n\n'+
                  f'Samples: {S}\n' +
                  f'bad_nc:  {bad_nc}\n' +
                  f'bad_pc:  {bad_pc}\n\n\n\n')
        elif verbose == 1:
            print(f'Iteration {abs(iterations)}', end='\r')
        
        # If x is positive counterexample
        # Which means that the counterexample should be true but the hypothesis says it is false.
        for clause in HYPOTHESIS.copy():
            # Find the clause in H that the counter example breaks
            if (evaluate(clause, counter_example_assignment,V) == False):
                # If the clause is a background restriction clause then the counter example is a bad example.
                # The background restrictions is just ensuring one-hot-encoding of the variables.
                if clause in background:
                    bad_pc.append(counter_example_assignment)
                
                # If the clause is not a background restriction clause then remove it from H
                # In the algorithm, the counterexample is added to a list of positive counter examples,
                # then the hypothesis is reconstructed with all the positive counter examples gathered so far.
                # Why do we simply remove the clause from the hypothesis here?
                else:
                    HYPOTHESIS.remove(clause)
                    # Add the counter example to positive counter examples
                    # This counter example is added multiple times. Once for every clause in the
                    # hypothesis that it breaks. Isn't this redundant?
                    Pos.append(counter_example_assignment)

                    # TODO: This is a bit confusing. What is the purpose of this?
                    # Doesn't nc mean negative counterexample? This is in the positive counterexample loop.
                    # The c maybe means a clause?

                    #check if a nc in S does not falsify a clause in H
                    # This updates the bad_nc list but we cannot see this from the code. 
                    # Possible refactor?
                    identify_problematic_nc(HYPOTHESIS,S,bad_nc,V)
                    
                    # Flag that the counter example is a positive counter example
                    pos_ex = True

        if not pos_ex:
            # If x is negative counterexample
            # This means that the counter example is false but the hypothesis says it is true so we should change the hypothesis.
            replaced = False
            for s in S:
                # Here we are looking for intersections of assignments between the negative counterexample and previous samples in S.
                s_intersection_x = [1 if s[index] ==1 and counter_example_assignment[index] == 1 else 0 for index in range(len(V))]
                # A contains everywhere there is intersection between the sample s and the counter example.
                # B contains everywhere there is a 1 in the sample s.
                A = {index for index,value in enumerate(s_intersection_x) if value ==1}
                B = {index for index,value in enumerate(s) if value ==1}
                if A.issubset(B) and not B.issubset(A): # A properly contained in B
                    # We refine a negative counterexample.
                    # This is the first such instance where a negative counter example is properly explained by a smaller counterexample. Line 7 in the original algorithm.
                    idx = S.index(s)
                    # We ask MQ if response is no. Line 6 in the algorithm.
                    if ask_membership_oracle(s_intersection_x) == False and s_intersection_x not in bad_nc:
                        # Did we already find it? We cannot control what counterexamples the LLM will give us.
                        # this simply does a print that the intersection is already in S. It does not do anything else.
                        checkduplicates(s_intersection_x,S,guard)
                        
                        # The name of this method is confusing. Is it is_go_into_be_duplicate?
                        # is_going_to_be_duplicate? (missing g)
                        # It checks if the intersection is already in S, if so add it to bad_nc.

                        # Potential issues: You are modifying the bad_nc list in a if check. 
                        # This is not clear from the code. TODO: potential refactor.
                        if not isgointobeduplicate(S,s_intersection_x,bad_nc):
                            S[idx] = s_intersection_x
                            replaced = True
                        break
            if not replaced:
                # We weren't able to refine a already present negative counter example.
                # Again the checkduplicates method is called. The method still doesnt do anything other than raise an exception.
                checkduplicates(counter_example_assignment,S,guard)
                # What gets added to S here?
                S.append(counter_example_assignment)

            # Reconstruct the hypothesis. Line 14 in the algorithm.
            HYPOTHESIS = get_hypothesis(S,V,bad_nc,background)
            #small optimisation. Refine hypo. with known positive counterexamples.
            HYPOTHESIS = positive_check_and_prune(HYPOTHESIS,S,Pos,V,bad_nc)
        iterations-=1
        stop = timeit.default_timer()
        data['runtime'] = stop-start
        metadata.append(data)
        if iterations % 5 == 0:
            sentence = "iteration = {eq}\tlen(H) = {h}\truntime = {rt}\n".format(eq = 5000 - iterations, h=len(HYPOTHESIS), rt = data['runtime'])
            with open('output.txt', 'a') as f:
                f.write(sentence)
    return (terminated, metadata, HYPOTHESIS)

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
