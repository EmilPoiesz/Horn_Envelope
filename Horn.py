from sympy import *
import functools
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

def evaluate(clause, x, V):
    
    #tautology or contradiction
    if clause == True or clause == False: return clause 

    assignment = {V[i]: x[i] for i in range(len(V))}
    return clause.subs(assignment)

def from_set_to_theory(set):
    """
    Converts a set of boolean clauses to a single boolean expression using logical AND.

    Args:
        set (iterable): An iterable of boolean clauses.

    Returns:
        Expr: The result of performing a logical AND operation on all elements in the set.
    """

    return functools.reduce(lambda x,y: x & y, set)

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

    H = ()
    Q = ()

    while iterations!=0:
        start = timeit.default_timer()
        iteration_data = {}
        
        # Issue with EQ oracle, checking H is fine, checking Q is not.
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
        for clause in current_hypothesis:
            
            if evaluate(clause, counterexample, V): continue
            
            assert clause not in background
            positive_counterexamples.append(counterexample)
            positive_counterexample_flag = True
            break

        # Negative counterexample check
        if not positive_counterexample_flag:

            replaced_flag = False
            for example in negative_counterexamples:

                intersection_of_counterexamples = [example[i] & counterexample[i] for i in range(len(V))]
                if  (intersection_of_counterexamples != example) and \
                    (not ask_membership_oracle(intersection_of_counterexamples)) and \
                    (intersection_of_counterexamples not in non_horn_counterexamples): #TODO: intersection models Q?
                    
                    idx = negative_counterexamples.index(example)
                    negative_counterexamples[idx] = intersection_of_counterexamples
                    replaced_flag = True
                    break

            if not replaced_flag: negative_counterexamples.append(counterexample)

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
    
    terminated = iterations != 0  
    return (terminated, metadata, current_hypothesis)
