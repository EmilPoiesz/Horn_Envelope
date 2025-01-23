from sympy import *
import functools
import random
import timeit
import Binarizer

def define_variables(number):
    """
    Creates a list of symbolic variables named 'v0', 'v1', ..., 'v(number-1)'.

    Parameters:
    number (int): The number of symbolic variables to generate.

    Returns:
    list: A list of symbolic variables.
    """
    return list(symbols("".join(['v'+str(i)+',' for i in range(number)])))

#TODO: This is not used?
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

def evaluate(clause, x, V):
    
    #Check if clause is tautology or contradiction
    if clause == True or clause == False: return clause 

    a = {V[i]: x[i] for i in range(len(V))}
    #for i in range(len(V)):
    #    a[V[i]] = (True if x[i] == 1
    #                    else False)
    return clause.subs(a)

def from_set_to_theory(set):
    """
    Converts a set of boolean clauses to a single boolean expression using logical AND.

    Args:
        set (iterable): An iterable of boolean clauses.

    Returns:
        Expr: The result of performing a logical AND operation on all elements in the set.
    """
    #TODO: Can we use functools.reduce(lambda: ... ) like in get_hypothesis?
    #combined_clause = True
    #for clause in set:
    #    combined_clause = combined_clause & clause
    #return combined_clause
    return functools.reduce(lambda x,y: x & y, set)

#TODO: This is not used?
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

#TODO: This is not used?
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

#TODO: This is not used?
def MQ(assignment, V, target):
    t = from_set_to_theory(target)
    return evaluate(t,assignment,V)

def get_hypothesis(negative_counterexamples, V, excluded_negative_counterexamples, background):
    H = set()
    for assignment in [example for example in negative_counterexamples if example not in excluded_negative_counterexamples]:
        # This seems like a weird mix of horn(e) and quasi(e)
        # Each assignment is in negative counterexamples meaning that it seems like we are in line 14, creating H.
        L = [V[index] for index,value in enumerate(assignment) if assignment[index] ==1 ] + [true]
        R = [V[index] for index,value in enumerate(assignment) if assignment[index] ==0 ] + [false]
        for r in R:
            # But here we take each true value and imply each false value in turn. 
            # Same as conjunction of true -> disjunction of false?
            # But this is the method for the quasi section which should be used on non-horn examples
            # Are bad_nc/excluded_negative_counterexamples the same as non-horn?
            # This is a bit confusing.
            clause = functools.reduce(lambda x,y: x & y, L)
            clause = (clause) >> r
            H.add(clause)
    H = H.union(background)
    return H

#TODO: this is not used?
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
    # I am not sure what this does as it is not part of the algorithm so I have commented out the one use of this method.

    # Is nc -> Non-norn Clause?

    # Current hypothesis
    h = from_set_to_theory(H)
    # List of negative? counterexamples that cannot produce a rule (according to positive counterexamples)?
    for a in [a for a in S if a not in bad_nc]: # Why bad? What does bad mean here?
        # If they are true, then are they negative counterexamples?
        # This feels like something that is correlated to NON-HORN section of the algorithm?
        if (evaluate(h, a, V) == True):
            bad_nc.append(a)

def is_going_to_be_duplicate(negative_counterexamples,example_intersection,bad_nc):
    if example_intersection in negative_counterexamples:
        bad_nc.append(example_intersection)
        return True
    else: return False

def is_subset(subset, superset):
    return all(x <= y for x, y in zip(subset, superset))

def intersect_all_lists(lists):
    return functools.reduce(lambda x, y: [a & b for a, b in zip(x, y)], lists)

def union_of_lists(list_of_lists):
    return functools.reduce(lambda x, y: [a | b for a, b in zip(x, y)], list_of_lists)

def learn(V, ask_membership_oracle, ask_equivalence_oracle, bad_nc, bad_pc, binarizer:Binarizer, background= {}, verbose = False,iterations=-1,guard=False):
    
    metadata = []
    current_hypothesis = set()
    current_hypothesis = current_hypothesis.union(background)
    
    negative_counterexamples = []
    positive_counterexamples = []
    non_horn_counterexamples = []

    while iterations!=0:
        start = timeit.default_timer()
        iteration_data = {}
        #iteration_data['Clauses removed'] = []
        
        counterexample_response = ask_equivalence_oracle(current_hypothesis)

        # Check if the hypothesis is correct
        if type(counterexample_response) == bool and counterexample_response:
            stop = timeit.default_timer()
            iteration_data['runtime'] = stop-start
            metadata.append(iteration_data)
            with open('output.txt', 'a') as f:
                f.write("=== TERMINATED ===\n")
            break
        
        (counterexample, sample_number) = counterexample_response
        iteration_data['sample'] = sample_number
        
        # Positive counterexample check
        positive_counterexample_flag = False
        for antecedent in current_hypothesis:#.copy():
            
            if evaluate(antecedent, counterexample,V): continue
            assert antecedent not in background
            
            #current_hypothesis.remove(clause) #This is different from the algorithm
            #iteration_data['Clauses removed'].append(clause)

            # Only add the counterexample once
            #if counterexample not in positive_counterexamples:
            positive_counterexamples.append(counterexample)
            
            # Flag that the counter example is a positive counter example
            positive_counterexample_flag = True
            break

            #check if a nc in S does not falsify a clause in H
            #identify_problematic_nc(current_hypothesis, negative_counterexamples, bad_nc, V)

        # Negative counterexample check
        if not positive_counterexample_flag:

            replaced_flag = False
            for negative_counterexample in negative_counterexamples:

                example_intersection = [negative_counterexample[i] & counterexample[i] for i in range(len(V))]
                if  (example_intersection != negative_counterexample) and \
                    (not ask_membership_oracle(example_intersection)) and \
                    (example_intersection not in non_horn_counterexamples): #TODO: intersection models Q?
                    
                    idx = negative_counterexamples.index(negative_counterexample)
                    negative_counterexamples[idx] = example_intersection
                    replaced_flag = True
                    break

            if not replaced_flag: negative_counterexamples.append(counterexample)

        #I added this section for moving negative counterexamples to non-Horn examples
        for neg_counterexample in negative_counterexamples:
            positive_superset = [pos_counterexample for pos_counterexample in positive_counterexamples if is_subset(neg_counterexample, pos_counterexample)]
            if positive_superset == []: continue
            if neg_counterexample == intersect_all_lists(positive_superset):
                negative_counterexamples.remove(neg_counterexample)
                non_horn_counterexamples.append(neg_counterexample)
        
        H = set()
        for neg_counterexample in negative_counterexamples:
            positive_superset = [pos_counterexample for pos_counterexample in positive_counterexamples if is_subset(neg_counterexample, pos_counterexample)]
            antecedent = And(*[V[i] for i, val in enumerate(neg_counterexample) if val == 1])
            if len(positive_superset) == 0:
                consequent = False
                implication = Implies(antecedent, consequent)
                H.add(implication)
            else:
                consequent = And(*[V[i] for i, val in enumerate(union_of_lists(positive_superset)) if val == 1])
                implication = Implies(antecedent, consequent)
                H.add(implication)

        Q = set()
        for nh_counterexample in non_horn_counterexamples:
            antecedent = And(*[V[i] for i, val in enumerate(nh_counterexample) if val == 1])
            consequent = Or(*[V[i] for i, val in enumerate(nh_counterexample) if val == 0])
            implication = Implies(antecedent, consequent)
            Q.add(implication)
        
        current_hypothesis = H.union(Q, background)
        # Is the hypothesis already the union of Horn and non-Horn clauses?
        #current_hypothesis = get_hypothesis(negative_counterexamples, V, bad_nc, background)
        #small optimisation. Refine hypo. with known positive counterexamples.
        #current_hypothesis = positive_check_and_prune(current_hypothesis,negative_counterexamples,positive_counterexamples,V,bad_nc)
        
        if verbose ==2:
            signed_counterexample = '+' if positive_counterexample_flag else '-'
            #clauses_removed = iteration_data['Clauses removed']
            print(f'\nIteration: {abs(iterations)}\n\n' + 
                  f'({sample_number}) Counterexample: ({signed_counterexample}) {binarizer.sentence_from_binary(counterexample)}\n\n'+
                  f'New Hypothesis: {sorted([str(h) for h in current_hypothesis if h not in background])}\n\n' +
                  f'New Hypothesis length: {len(current_hypothesis)-len(background)} + background: {len(background)}\n\n' +
                  f'total positive counterexamples:  {len(positive_counterexamples)}\n' +
                  f'total negative counterexamples:  {len(negative_counterexamples)} ({sum(sum(lst) for lst in negative_counterexamples)})\n' +
                  f'total non-horn counterexamples:  {len(non_horn_counterexamples)}\n\n\n\n')
        elif verbose == 1:
            print(f'Iteration {abs(iterations)}', end='\r')
        
        iterations-=1
        stop = timeit.default_timer()
        iteration_data['runtime'] = stop-start
        metadata.append(iteration_data)
        if iterations % 5 == 0:
            sentence = "iteration = {eq}\tlen(H) = {h}\truntime = {rt}\n".format(eq = 5000 - iterations, h=len(current_hypothesis), rt = iteration_data['runtime'])
            with open('output.txt', 'a') as f:
                f.write(sentence)
    
    # Did we terminate or exhaust the limit?
    terminated = iterations != 0  
    return (terminated, metadata, current_hypothesis)

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
